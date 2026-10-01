"""Compare frozen direct V2 output with distributed adapter on test sample."""
import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'distributed'
sys.path.insert(0, str(DIST))
from worker import configure, preflight, score_source, shard_records, manifest_rows, hardware, combine, run_features


def evaluate(n=12):
    import common
    import lightgbm as lgb
    import numpy as np
    pre = preflight()
    records = shard_records('0000', manifest_rows()['0000'])[:n]
    sparse, pair = configure()
    model = lgb.Booster(model_file=str(ROOT / 'artifacts/v2/run/lightgbm.txt'))
    paths = [DIST / 'equivalence/frozen_direct', DIST / 'equivalence/distributed_adapter']
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    results = []
    for mode, path in enumerate(paths):
        timing = {'retrieval_seconds': 0., 'feature_seconds': 0., 'scoring_seconds': 0., 'serialization_seconds': 0.}
        all_candidates = []
        all_features = []
        all_scores = []
        all_decisions = []
        for src in (2, 3):
            t = time.perf_counter()
            sparse.retrieve(records, src, path, k=100, policy='tiered')
            timing['retrieval_seconds'] += time.perf_counter()-t
            t = time.perf_counter()
            run_features(records, src, path, pair, 1 if mode == 0 else 4)
            timing['feature_seconds'] += time.perf_counter()-t
            with gzip.open(path / f's{src}_candidates.jsonl.gz', 'rt', encoding='utf-8') as f:
                all_candidates.extend([{'source': src, **json.loads(line)} for line in f])
            for npz in sorted(path.glob(f's{src}_[0-9][0-9][0-9][0-9][0-9].npz')):
                with np.load(npz) as z:
                    X, q, eid = z['X'], z['q'], z['eid']
                t = time.perf_counter()
                score = model.predict(X, num_threads=4).astype(np.float32) if len(X) else np.empty(0, np.float32)
                timing['scoring_seconds'] += time.perf_counter()-t
                all_features.append((src, q.copy(), eid.copy(), X.copy()))
                all_scores.append((src, q.copy(), eid.copy(), score.copy()))
                all_decisions.extend((src, int(q[i]), str(eid[i])) for i in range(len(q)) if score[i] >= 0.642)
            if mode == 1:
                t = time.perf_counter()
                score_source(src, records, path, model, 4)
                timing['serialization_seconds'] += time.perf_counter()-t
        results.append((all_candidates, all_features, all_scores, all_decisions, timing))
    a, b = results
    if a[0] != b[0]:
        raise ValueError('Candidate sets or route metadata differ')
    if len(a[1]) != len(b[1]):
        raise ValueError('Feature chunk count differs')
    for x, y in zip(a[1], b[1]):
        if x[0] != y[0] or not np.array_equal(x[1], y[1]) or not np.array_equal(x[2], y[2]) or not np.array_equal(x[3], y[3]):
            raise ValueError('Feature vectors differ')
    for x, y in zip(a[2], b[2]):
        if x[0] != y[0] or not np.array_equal(x[1], y[1]) or not np.array_equal(x[2], y[2]) or not np.array_equal(x[3], y[3]):
            raise ValueError('LightGBM scores differ')
    if a[3] != b[3]:
        raise ValueError('Threshold decisions differ')
    for src in (2, 3):
        candidate_lines = (paths[1] / f's{src}_candidate.tsv').read_text(encoding='utf-8').splitlines()
        matching_lines = (paths[1] / f's{src}_matching.tsv').read_text(encoding='utf-8').splitlines()
        for qi in range(n):
            candidates = set(candidate_lines[qi].split('\t')[1].split(',')) - {''}
            matches = set(matching_lines[qi].split('\t')[1].split(',')) - {''}
            direct_candidates = {str(eid) for ss, q, es, _ in a[1] if ss == src for ix, eid in zip(q, es) if ix == qi}
            direct_matches = {eid for ss, ix, eid in a[3] if ss == src and ix == qi}
            if candidates != direct_candidates or matches != direct_matches:
                raise ValueError('Final candidate or matched IDs differ')
    from merge import validate_pair
    combine(records, paths[1], 'candidate', paths[1] / 'candidate_part.tsv')
    combine(records, paths[1], 'matching', paths[1] / 'matching_part.tsv')
    validate_pair(paths[1] / 'matching_part.tsv', paths[1] / 'candidate_part.tsv', {r['entity_id'] for r in records})
    pair_count = sum(len(x[2]) for x in a[1])
    seconds = sum(b[4].values())
    disk_bytes = sum(p.stat().st_size for p in paths[1].rglob('*') if p.is_file())
    metrics = {'status': 'PASS', 'sample_source1_count': n, 'candidate_pairs': pair_count,
               'matched_links': len(a[3]), 'candidate_sets_identical': True, 'features_bitwise_identical': True,
               'scores_bitwise_identical': True, 'threshold_decisions_identical': True,
               'final_ids_identical': True, 'hardware': hardware(), 'direct_timings': a[4],
               'distributed_timings': b[4], 'distributed_sample_seconds': seconds,
               'sample_source1_per_second': n / seconds, 'sample_candidate_pairs_per_second': pair_count / seconds,
               'sample_disk_bytes': disk_bytes,
               'sample_peak_ram_bytes': __import__('prepare').memory().get('peak_rss_bytes'),
               'full_test_hours_naive_sample_extrapolation': seconds * 1732544 / n / 3600}
    (DIST / 'equivalence_report.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=int, default=12)
    a = p.parse_args()
    evaluate(a.sample)
