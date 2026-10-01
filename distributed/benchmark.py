"""Measured test-target benchmark without completing any production shard."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'distributed'
sys.path.insert(0, str(DIST))
from worker import configure, preflight, score_source, shard_records, manifest_rows, hardware, combine, run_features


def run(n):
    import common
    import lightgbm as lgb
    from prepare import memory
    from merge import validate_pair
    preflight()
    records = shard_records('0000', manifest_rows()['0000'])[:n]
    sparse, pair = configure()
    model = lgb.Booster(model_file=str(ROOT / 'artifacts/v2/run/lightgbm.txt'))
    out = DIST / f'benchmark_{n}'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    timings = {'retrieval_seconds': 0., 'feature_seconds': 0., 'scoring_seconds': 0., 'serialization_seconds': 0.}
    feature_worker_counts = {}
    pairs = links = 0
    peak_bound = 0
    threads = hardware()['native_threads']
    start = time.perf_counter()
    for src in (2, 3):
        t = time.perf_counter()
        sparse.retrieve(records, src, out, k=100, policy='tiered')
        timings['retrieval_seconds'] += time.perf_counter()-t
        t = time.perf_counter()
        available = hardware()['available_ram']
        feature_workers = (4 if (hardware()['logical_cpus'] or 1) >= 8 and available >= 6 * 2**30 else
                           2 if (hardware()['logical_cpus'] or 1) >= 4 and available >= int(4.5 * 2**30) else 1)
        feature_worker_counts[str(src)] = feature_workers
        peak_bound = max(peak_bound, run_features(records, src, out, pair, feature_workers))
        timings['feature_seconds'] += time.perf_counter()-t
        t = time.perf_counter()
        result = score_source(src, records, out, model, threads)
        timings['scoring_seconds'] += time.perf_counter()-t
        pairs += result['pairs']
        links += result['links']
    t = time.perf_counter()
    combine(records, out, 'candidate', out / 'candidate_part.tsv')
    combine(records, out, 'matching', out / 'matching_part.tsv')
    checked = validate_pair(out / 'matching_part.tsv', out / 'candidate_part.tsv', {r['entity_id'] for r in records})
    if checked != (pairs, links):
        raise ValueError('Benchmark output count mismatch')
    timings['serialization_seconds'] += time.perf_counter()-t
    elapsed = time.perf_counter()-start
    report = {'status': 'PASS', 'sample_source1_count': n, 'candidate_pairs': pairs, 'predicted_links': links,
              **timings, 'end_to_end_seconds': elapsed, 'source1_per_second': n / elapsed,
              'candidate_pairs_per_second': pairs / elapsed,
              'peak_ram_bytes': max(peak_bound, memory().get('peak_rss_bytes', 0)),
              'disk_usage_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
              'hardware': hardware(), 'feature_workers_by_source': feature_worker_counts,
              'note': 'Sample fixed overhead makes naive full-test extrapolation pessimistic'}
    (DIST / 'performance_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=int, default=1000)
    a = p.parse_args()
    run(a.sample)
