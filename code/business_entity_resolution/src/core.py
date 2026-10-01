"""Local entity resolution primitives. No network or test-inference entry point."""
from __future__ import annotations
import csv
import hashlib
import math
import re
import unicodedata as ud
from collections import Counter
from pathlib import Path

VERSION = 'v1.0'
SEED = 2026
LEGAL = frozenset('ltd limited pvt private llc inc incorporated corp corporation company co llp pllc'.split())

def rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed TSV: {path}')
            yield row

def parse_ids(text, valid=None):
    ids = [] if not text.strip() else [x.strip() for x in text.split(',')]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate ID')
    if any(not x.startswith(('S2-', 'S3-')) for x in ids):
        raise ValueError('Only S2/S3 target IDs are allowed')
    if valid is not None and set(ids) - valid:
        raise ValueError('Unknown target ID')
    return frozenset(ids)

def read_sets(path, column, required=None, valid=None):
    with open(path,encoding='utf-8') as f:
        if f.readline().rstrip('\r\n').split('\t') != ['source1_entity_id',column]:
            raise ValueError('Wrong or missing TSV header')
    out = {}
    for row in rows(path):
        if set(row) != {'source1_entity_id', column}:
            raise ValueError('Wrong TSV header')
        eid = row['source1_entity_id']
        if not eid.startswith('S1-') or eid in out:
            raise ValueError('Invalid or duplicate S1 ID')
        out[eid] = parse_ids(row[column], valid)
    if required is not None and set(out) != set(required):
        raise ValueError('S1 coverage mismatch')
    return out

def entity_metric(truth, pred):
    t, p = set(truth), set(pred)
    if not t:
        return (1.0, 1.0, 1.0) if not p else (0.0, 0.0, 0.0)
    tp = len(t & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(t)
    return precision, recall, 1.25 * tp / (0.25 * len(t) + len(p))

def metrics(truth, pred):
    if set(truth) != set(pred) or not truth:
        raise ValueError('Metric requires identical nonempty entity coverage')
    prf = [entity_metric(t, pred[e]) for e, t in truth.items()]
    n = len(truth)
    empty = [e for e,t in truth.items() if not t]
    return dict(n_entities=n, macro_precision=sum(x[0] for x in prf)/n,
                macro_recall=sum(x[1] for x in prf)/n, macro_f0_5=sum(x[2] for x in prf)/n,
                singleton_accuracy=sum(not pred[e] for e in empty)/len(empty) if empty else None,
                singleton_count=len(empty), zero_match_rate=sum(not p for p in pred.values())/n,
                average_links=sum(map(len,pred.values()))/n,
                prediction_counts=dict(Counter(str(min(5,len(p))) for p in pred.values())),
                true_counts=dict(Counter(str(min(5,len(t))) for t in truth.values())),
                false_positive_links=sum(len(pred[e]-t) for e,t in truth.items()),
                false_negative_links=sum(len(t-pred[e]) for e,t in truth.items()))

def unicode_text(text):
    return ud.normalize('NFKC', text).lower().strip()

def norm(text):
    # Preserve combining marks, which are part of Indic words, unlike regex \w.
    text = unicode_text(text)
    return ' '.join(''.join(c if ud.category(c)[0] in 'LNM' else ' ' for c in text).split())

def fold(text):
    # Strip Latin accents only; retain Indic vowel signs and other non-Latin marks.
    out=[]; latin=False
    for c in ud.normalize('NFD', text):
        if not ud.combining(c):
            latin='LATIN' in ud.name(c,'')
        if not (ud.combining(c) and latin): out.append(c)
    return ud.normalize('NFC',''.join(out))

def suffix(text):
    tokens=text.split()
    while tokens and tokens[-1] in LEGAL: tokens.pop()
    return ' '.join(tokens)

def grams(text, low=3, high=5):
    s=' '+text+' '
    return frozenset(s[i:i+k] for k in range(low,high+1) for i in range(len(s)-k+1)) if text else frozenset()

def scripts(text):
    return frozenset(ud.name(c,'UNKNOWN').split()[0] for c in text if ud.category(c).startswith('L'))

def view(raw):
    n,a=norm(raw['business_name']),norm(raw['business_address'])
    nf,af=fold(n),fold(a)
    return dict(raw=raw, unicode_name=unicode_text(raw['business_name']), unicode_address=unicode_text(raw['business_address']),
        n=n,a=a,nf=nf,af=af,s=suffix(nf),nt=frozenset(nf.split()),at=frozenset(af.split()),
        ng=grams(nf),ag=grams(af),nums=frozenset(str(int(x)) for x in re.findall(r'\d+',a)),
        postal=frozenset(re.findall(r'(?<!\d)(?:\d{6}|\d{5}(?:-\d{4})?)(?!\d)',a)),
        script=scripts(n),missing=not bool(a))

def stable(text):
    return int.from_bytes(hashlib.blake2b((str(SEED)+'|'+text).encode(),digest_size=8).digest(),'big')

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def jaccard(a,b):
    return len(a&b)/len(a|b) if a or b else 0.0

def containment(a,b):
    return len(a&b)/min(len(a),len(b)) if a and b else 0.0

def write_sets(path, mapping, column):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n',quoting=csv.QUOTE_NONE)
        w.writerow(['source1_entity_id',column])
        for eid, ids in sorted(mapping.items()): w.writerow([eid,','.join(sorted(ids))])
