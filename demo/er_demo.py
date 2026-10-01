"""End-to-end demo of the entity-resolution pipeline on SYNTHETIC business records.

Everything here is made up: no challenge data is used. The demo runs the real
pipeline logic at small scale:
  1. normalization (core.py from the repository)
  2. the fast R10 retrieval profile: hashed binary TF-IDF name-char / address-word /
     joint routes + exact-name / suffix-name / address-key joins, top 10 per route
  3. the exact 80 pair features used in training
  4. the trained LightGBM base model -> score-context meta model -> threshold 0.70
  5. one-owner-per-target conflict resolution

Scores on a few hundred synthetic records are illustrative, not a benchmark: the real
models were trained with IDF statistics and name frequencies from a ~5M-record corpus.
"""
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
from anyascii import anyascii
from rapidfuzz import fuzz
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

_HERE = Path(__file__).resolve().parent
# Works from the Hugging Face / Kaggle folder (core.py alongside) or from the GitHub repo's demo/ folder.
for _p in (_HERE, _HERE.parent / 'code/business_entity_resolution/src'):
    if (_p / 'core.py').exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from core import LEGAL, containment, jaccard, view  # noqa: E402
from inference import META_THRESHOLD, OWNERSHIP_MARGIN, context, BASE_COLS, resolve_ownership  # noqa: E402

RANK = 10
_MODEL_FILES = {'base': ('base_lightgbm.txt', '../artifacts/v2/run/lightgbm.txt'),
                'meta': ('meta_lightgbm.txt', '../distributed/recovery/meta_r10_lightgbm.txt')}


def _model_path(model_dir, kind):
    for name in _MODEL_FILES[kind]:
        path = Path(model_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(f'{kind} model not found under {model_dir}')

# --------------------------------------------------------------------------------------
# Synthetic data: designed hard cases + generated filler businesses for realistic IDF.
# --------------------------------------------------------------------------------------
QUERIES = [
    ('S1-demo-01', 'Sharma Textiles Private Limited', 'Plot 14, MIDC Industrial Area, Andheri East, Mumbai, Maharashtra 400093', 'India'),
    ('S1-demo-02', 'Krishna Enterprises', '12 MG Road, Indore, Madhya Pradesh 452001', 'India'),
    ('S1-demo-03', 'Blue Ridge Coffee Roasters LLC', '1200 Main St Ste 4, Asheville, NC 28801', 'US'),
    ('S1-demo-04', 'Nova Robotics Inc', '77 Harbor Blvd, Oakland, CA 94607', 'US'),
    ('S1-demo-05', 'Ganesh Stores', 'Shop 3, Laxmi Road, Pune, Maharashtra 411030', 'India'),
    ('S1-demo-06', 'Sunrise Medical Center', '45 Park Avenue, New York, NY 10016', 'US'),
    ('S1-demo-07', 'Boulangerie Dupont SARL', '12 Rue de la République, 69002 Lyon', 'France'),
    ('S1-demo-08', 'Precision Auto Parts Co', '5400 W Commerce Dr, Tulsa, OK 74107', 'US'),
]
# (entity_id, name, address, country, true S1 owner or None, case description)
TARGETS = [
    ('S2-demo-101', 'SHARMA TEXTILES PVT LTD', 'PLOT NO 14 MIDC ANDHERI (E) MUMBAI MAHARASHTRA 400093', 'India', 'S1-demo-01', 'abbreviated legal suffix + address'),
    ('S3-demo-102', 'Sharma Textiles', '14, MIDC Industrial Area, Andheri East, Mumbai', 'India', 'S1-demo-01', 'suffix dropped, pincode missing'),
    ('S2-demo-103', 'Sharma Textile Traders', '22 Linking Road, Bandra West, Mumbai, Maharashtra 400050', 'India', None, 'similar name, different business'),
    ('S2-demo-201', 'कृष्णा एंटरप्राइजेज', '12 एमजी रोड, इंदौर, मध्य प्रदेश 452001', 'India', 'S1-demo-02', 'Devanagari script'),
    ('S3-demo-202', 'Krishna Enterprises', '', 'India', 'S1-demo-02', 'missing address'),
    ('S3-demo-203', 'Krishna Enterprises', '8 Park Street, Kolkata, West Bengal 700016', 'India', None, 'same name, other city'),
    ('S2-demo-301', 'Blue Ridge Coffee Roasters', '1200 Main Street Suite 4, Asheville, NC 28801', 'US', 'S1-demo-03', 'abbreviation expansion'),
    ('S3-demo-302', 'BLUE RIDGE COFFEE', '1200 MAIN ST, ASHEVILLE, NC', 'US', 'S1-demo-03', 'truncated name, upper case'),
    ('S2-demo-303', 'Main Street Dental Care', '1200 Main St Ste 6, Asheville, NC 28801', 'US', None, 'co-located business'),
    ('S2-demo-401', 'Nova Robotics Solutions', '9100 Research Blvd, Austin, TX 78758', 'US', None, 'similar name, other state (S1 has no match)'),
    ('S2-demo-501', 'Ganesh Stores', 'Shop No. 3, Laxmi Rd, Pune 411030', 'India', 'S1-demo-05', 'common name, same address'),
    ('S3-demo-502', 'Ganesh Stores', '41 Anna Salai, Chennai, Tamil Nadu 600002', 'India', None, 'common name, other city'),
    ('S2-demo-601', 'Sunrise Med Ctr', 'Park Ave 45, New York NY 10016', 'US', 'S1-demo-06', 'abbreviations + reordered address'),
    ('S3-demo-602', 'Sunrise Medical Centre', '45 Park Ave, Manhattan, New York 10016', 'US', 'S1-demo-06', 'British spelling'),
    ('S2-demo-701', 'BOULANGERIE DUPONT', '12 RUE DE LA REPUBLIQUE LYON 69002', 'France', 'S1-demo-07', 'accents removed'),
    ('S3-demo-801', 'Precison Auto Prts', '5400 West Commerce Drive, Tulsa, OK 74107', 'US', 'S1-demo-08', 'typos + expanded address'),
    ('S3-demo-802', 'Precision Auto Glass', '1150 S Memorial Dr, Tulsa, OK 74112', 'US', None, 'similar name, same city'),
]

_IN_FIRST = 'Agarwal Bansal Chopra Desai Gupta Iyer Jain Kapoor Kumar Mehta Nair Patel Rao Reddy Singh Verma Joshi Pillai Das Bose'.split()
_IN_WORD = 'Traders Industries Exports Foods Pharma Electricals Motors Builders Jewellers Agencies Chemicals Logistics Steels Plastics Garments'.split()
_IN_SUFFIX = ['Private Limited', 'Pvt Ltd', 'Limited', 'LLP', '', '']
_IN_STREET = 'MG Road|Station Road|Nehru Nagar|Gandhi Chowk|Ring Road|Link Road|Industrial Estate|Sector 18|Civil Lines|Market Yard'.split('|')
_IN_CITY = [('Mumbai', 'Maharashtra', '400'), ('Pune', 'Maharashtra', '411'), ('Delhi', 'Delhi', '110'), ('Chennai', 'Tamil Nadu', '600'),
            ('Bengaluru', 'Karnataka', '560'), ('Ahmedabad', 'Gujarat', '380'), ('Jaipur', 'Rajasthan', '302'), ('Kolkata', 'West Bengal', '700')]
_US_FIRST = 'Summit Pioneer Liberty Evergreen Golden Harbor Cedar Maple Eagle Silver Northstar Riverside Crescent Atlas Union'.split()
_US_WORD = 'Plumbing|Dental|Bakery|Logistics|Consulting|Auto Repair|Realty|Fitness|Pharmacy|Printing|Hardware|Insurance|Catering|Roofing|Florist'.split('|')
_US_SUFFIX = ['LLC', 'Inc', 'Co', 'Corp', '', '']
_US_STREET = 'Main St|Oak Ave|Elm St|Broadway|Market St|Washington Blvd|Lake Dr|Pine Rd|2nd Ave|Highland Ave'.split('|')
_US_CITY = [('Austin', 'TX', '787'), ('Denver', 'CO', '802'), ('Portland', 'OR', '972'), ('Columbus', 'OH', '432'),
            ('Tampa', 'FL', '336'), ('Phoenix', 'AZ', '850'), ('Boston', 'MA', '021'), ('Seattle', 'WA', '981')]


def _fillers(src, n=350, seed=7):
    rng = random.Random(seed * 10 + src)
    out = []
    for i in range(n):
        if rng.random() < .55:
            city, state, pin = rng.choice(_IN_CITY)
            name = ' '.join(x for x in (rng.choice(_IN_FIRST), rng.choice(_IN_WORD), rng.choice(_IN_SUFFIX)) if x)
            addr = f'{rng.randint(1, 300)}, {rng.choice(_IN_STREET)}, {city}, {state} {pin}{rng.randint(1, 99):03d}'
            country = 'India'
        else:
            city, state, zp = rng.choice(_US_CITY)
            name = ' '.join(x for x in (rng.choice(_US_FIRST), rng.choice(_US_WORD), rng.choice(_US_SUFFIX)) if x)
            addr = f'{rng.randint(10, 9999)} {rng.choice(_US_STREET)}, {city}, {state} {zp}{rng.randint(1, 99):02d}'
            country = 'US'
        out.append((f'S{src}-fill-{i:04d}', name, addr, country))
    return out


def corpus(src):
    designed = [t[:4] for t in TARGETS if t[0].startswith(f'S{src}-')]
    return designed + _fillers(src)


# --------------------------------------------------------------------------------------
# Retrieval: same vectorizer definitions as v2/hashed_index.py, fast R10 route profile.
# --------------------------------------------------------------------------------------
def augment(s):
    if s.isascii():
        return s
    t = anyascii(s).lower()
    return s + ' ' + t if t != s else s


def retrieval_text(s, address=False):
    s = augment(s)
    if address:
        s = re.sub(r'\d+', lambda m: str(int(m[0])), s)
    return s


class HashTfidf:
    def __init__(self, route):
        self.route = route
        self.idf = None
        self.vectorizer = HashingVectorizer(
            n_features=2**20, alternate_sign=False, norm=None, binary=True,
            analyzer='char' if 'char' in route else 'word', ngram_range=(3, 5) if route == 'name_char' else (1, 1),
            lowercase=False, token_pattern=r'(?u)\b\w+\b', dtype=np.float32)

    def raw(self, texts):
        return self.vectorizer.transform([retrieval_text(t, self.route.startswith('address')) for t in texts])

    def fit(self, texts):
        m = self.raw(texts)
        d = np.bincount(m.indices, minlength=2**20)
        n = m.shape[0]
        idf = (1 + np.log((n + 1) / (d + 1))).astype(np.float32)
        idf[d > n * (.03 if 'char' in self.route else .1)] = 0
        idf[d == 0] = 0
        self.idf = idf
        return self

    def transform(self, texts):
        m = self.raw(texts)
        m.data *= self.idf[m.indices]
        m.eliminate_zeros()
        normalize(m, copy=False)
        return m


def topk(q, m, k, threshold):
    sims = (q @ m.T).toarray()
    out = []
    for row in sims:
        ix = [i for i in np.lexsort((np.arange(len(row)), -row)) if row[i] >= threshold][:k]
        out.append([(int(i), float(row[i])) for i in ix])
    return out


def retrieve(query_views, target_views):
    names = HashTfidf('name_char').fit([v['nf'] for v in target_views])
    addrs = HashTfidf('address_word').fit([v['af'] for v in target_views])
    tn = names.transform([v['nf'] for v in target_views])
    ta = addrs.transform([v['af'] for v in target_views])
    qn = names.transform([v['nf'] for v in query_views])
    qa = addrs.transform([v['af'] for v in query_views])
    routes = {'name_char': topk(qn, tn, RANK, .015), 'address_word': topk(qa, ta, RANK, .015),
              'joint_name_address': topk(sparse.hstack([qn * .5, qa * .5], format='csr'),
                                         sparse.hstack([tn, ta], format='csr'), RANK, .08)}
    freq = []
    for route, key in (('exact_name', lambda v: v['nf']), ('suffix_name', lambda v: v['s']),
                       ('address_key', lambda v: ' '.join(sorted(v['at'])))):
        hits = []
        for i, q in enumerate(query_views):
            h = [(j, fuzz.token_set_ratio(q['af'], t['af']) / 100) for j, t in enumerate(target_views)
                 if key(q) and key(t) == key(q)]
            if route != 'address_key':
                if len(freq) <= i:
                    freq.append([0, 0])
                freq[i][0 if route == 'exact_name' else 1] = len(h)
            hits.append(sorted(h, key=lambda x: (-x[1], x[0]))[:RANK])
        routes[route] = hits
    candidates = []
    for i in range(len(query_views)):
        found = {}
        for route, per_query in routes.items():
            for rank, (j, score) in enumerate(per_query[i], 1):
                found.setdefault(j, {})[route] = (rank, score)
        candidates.append(found)
    return candidates, freq


# --------------------------------------------------------------------------------------
# The 80 pair features: verbatim logic of src/retrieval.py features() + v2 extras().
# --------------------------------------------------------------------------------------
V1_ROUTES = ['exact_name', 'suffix_name', 'rare_name_token', 'name_char_tfidf', 'address_token', 'address_key', 'numeric_address', 'address_char']
ROUTE_MAP = {'name_char': 'name_char_tfidf', 'address_word': 'address_token', 'exact_name': 'exact_name',
             'suffix_name': 'suffix_name', 'address_key': 'address_key'}
RESCUES = ['translit_char', 'missing_name_char', 'joint_name_address']


def cosine(a, b, idf):
    if not a or not b:
        return 0.0
    dot = sum(idf.get(t, 1.)**2 for t in a & b)
    return dot / math.sqrt(sum(idf.get(t, 1.)**2 for t in a) * sum(idf.get(t, 1.)**2 for t in b))


def ratio(a, b):
    return min(len(a), len(b)) / max(len(a), len(b), 1)


def edit(a, b):
    return fuzz.ratio(a, b) / 100 if a and b else 0.


def acronym(v):
    return ''.join(t[0] for t in v['s'].split())


def v1_features(q, c, routes, density, nfreq, sfreq, src, idf):
    nj = jaccard(q['ng'], c['ng']); aj = jaccard(q['ag'], c['ag']); ne = float(q['n'] == c['n'])
    nm = c['missing']; nc = float(bool(q['nums'] and c['nums'] and not q['nums'] & c['nums']))
    pe = float(bool(q['postal'] & c['postal'])); pc = float(bool(q['postal'] and c['postal'] and not pe))
    z = [ne, float(bool(q['s']) and q['s'] == c['s']), nj, cosine(q['ng'], c['ng'], idf), jaccard(q['nt'], c['nt']), containment(q['nt'], c['nt']), edit(q['nf'], c['nf']),
         ratio(q['nf'], c['nf']), abs(len(q['nt']) - len(c['nt'])), q['nf'][:4] == c['nf'][:4], q['nf'][-4:] == c['nf'][-4:],
         float(bool(acronym(q)) and (acronym(q) == acronym(c) or acronym(q) == c['s'].replace(' ', '') or acronym(c) == q['s'].replace(' ', ''))),
         bool(q['script'] & c['script']), math.log1p(nfreq), math.log1p(sfreq), bool(q['a']) and q['a'] == c['a'], aj, jaccard(q['at'], c['at']), containment(q['at'], c['at']), ratio(q['af'], c['af']),
         jaccard(q['nums'], c['nums']), nc, pe, pc, q['missing'], nm, nj * aj, ne * nc, nj * nm, (1 - nj) * aj,
         q['raw']['country'] == c['raw']['country'], src == 3, math.log1p(density), q['nf'] == c['nf'], q['unicode_name'] == c['unicode_name']]
    for r in V1_ROUTES:
        rank, score = routes.get(r, (0, 0))
        z += [bool(rank), 1 / rank if rank else 0, score]
    return z


def extras(q, c):
    a = ' '.join(t for t in q['nf'].split() if t not in LEGAL); b = ' '.join(t for t in c['nf'].split() if t not in LEGAL)
    qa = anyascii(q['nf']).lower(); ca = anyascii(c['nf']).lower()
    return [fuzz.token_sort_ratio(q['nf'], c['nf']) / 100, fuzz.partial_ratio(q['nf'], c['nf']) / 100,
            fuzz.ratio(a, b) / 100 if a and b else 0, containment(set(a.split()), set(b.split())),
            fuzz.ratio(qa, ca) / 100, fuzz.token_set_ratio(qa, ca) / 100,
            fuzz.token_set_ratio(q['af'], c['af']) / 100 if c['af'] else 0, fuzz.partial_ratio(q['af'], c['af']) / 100 if c['af'] else 0,
            fuzz.token_set_ratio(' '.join(t for t in q['at'] if not t.isnumeric()), ' '.join(t for t in c['at'] if not t.isnumeric())) / 100 if c['af'] else 0,
            max((fuzz.ratio(x, y) / 100 for x in q['nums'] for y in c['nums']), default=0), 0, 0]


def pair_features(q, c, hits, density, nfreq, sfreq, src, idf):
    routes = {ROUTE_MAP[r]: v for r, v in hits.items() if r in ROUTE_MAP}
    if 'address_word' in hits and q['nums'] & c['nums']:
        routes['numeric_address'] = hits['address_word']
    rescue = []
    for route in RESCUES:
        rank, score = hits.get(route, (0, 0))
        rescue.extend([bool(rank), 1 / rank if rank else 0, score])
    return v1_features(q, c, routes, density, nfreq, sfreq, src, idf) + extras(q, c) + rescue


def _record(t):
    return dict(zip(['entity_id', 'business_name', 'business_address', 'country'], t[:4]))


def run_demo(model_dir=_HERE):
    base_model = lgb.Booster(model_file=str(_model_path(model_dir, 'base')))
    meta_model = lgb.Booster(model_file=str(_model_path(model_dir, 'meta')))
    qviews = [view(_record(q)) for q in QUERIES]
    truth = {q[0]: {t[0] for t in TARGETS if t[4] == q[0]} for q in QUERIES}
    # Designed target S2-demo-1xx belongs to query S1-demo-01, 2xx to S1-demo-02, ...
    case = {(f'S1-demo-{int(t[0][-3:]) // 100:02d}', t[0]): t[5] for t in TARGETS}
    X, qidx, source, pairs = [], [], [], []
    stats = {}
    for src in (2, 3):
        tviews = [view(_record(t)) for t in corpus(src)]
        gd = Counter()
        for v in tviews:
            gd.update(v['ng'])
        idf = {g: 1 + math.log((len(tviews) + 1) / (d + 1)) for g, d in gd.items()}
        candidates, freq = retrieve(qviews, tviews)
        stats[src] = {'targets': len(tviews), 'pairs': sum(map(len, candidates))}
        for qi, found in enumerate(candidates):
            begin = len(X)
            for j, hits in sorted(found.items()):
                X.append(pair_features(qviews[qi], tviews[j], hits, len(found), freq[qi][0], freq[qi][1], src, idf))
                qidx.append(qi); source.append(src); pairs.append((QUERIES[qi], tviews[j]['raw'], hits))
            if len(X) > begin:  # relative evidence within the query's candidates (features 69/70)
                bestn = max(x[2] for x in X[begin:]); besta = max(x[16] for x in X[begin:])
                for x in X[begin:]:
                    x[69] = x[2] - bestn; x[70] = x[16] - besta
    X = np.asarray(X, dtype=np.float32)
    qidx = np.asarray(qidx); source = np.asarray(source)
    base = base_model.predict(X)
    meta = meta_model.predict(np.concatenate([X[:, BASE_COLS], context(qidx, source, base)], axis=1))
    links = [(p[0][0], p[1]['entity_id'], s) for p, s in zip(pairs, meta) if s >= META_THRESHOLD]
    final = resolve_ownership(links, OWNERSHIP_MARGIN)
    rows = []
    for (qrec, trec, hits), b, m in zip(pairs, base, meta):
        tid = trec['entity_id']
        rows.append({'query_id': qrec[0], 'query_name': qrec[1], 'candidate_id': tid,
                     'candidate_name': trec['business_name'], 'candidate_address': trec['business_address'],
                     'routes': ','.join(sorted(hits)), 'base_score': float(b), 'meta_score': float(m),
                     'predicted': tid in final.get(qrec[0], []), 'true_match': tid in truth[qrec[0]],
                     'case': case.get((qrec[0], tid), 'other record')})
    per_query = []
    for q in QUERIES:
        t, p = truth[q[0]], set(final.get(q[0], []))
        tp = len(t & p)
        f = 1.0 if not t and not p else (1.25 * tp / (.25 * len(t) + len(p)) if p else 0.0)
        per_query.append({'query_id': q[0], 'name': q[1], 'country': q[3], 'true': len(t), 'predicted': len(p),
                          'correct': tp, 'f0_5': f})
    return {'pairs': rows, 'per_query': per_query, 'stats': stats,
            'macro_f0_5': sum(r['f0_5'] for r in per_query) / len(per_query), 'feature_matrix': X}


if __name__ == '__main__':
    result = run_demo()
    print('corpus:', result['stats'])
    for r in result['per_query']:
        print(f"{r['query_id']}  {r['name'][:32]:<32} true={r['true']} pred={r['predicted']} correct={r['correct']} F0.5={r['f0_5']:.2f}")
    print('macro F0.5 on synthetic queries: %.3f' % result['macro_f0_5'])
    for r in sorted(result['pairs'], key=lambda r: -r['meta_score'])[:25]:
        print(f"{r['query_id']} -> {r['candidate_name'][:28]:<28} base={r['base_score']:.3f} meta={r['meta_score']:.3f} "
              f"pred={int(r['predicted'])} true={int(r['true_match'])}  [{r['case']}]")
