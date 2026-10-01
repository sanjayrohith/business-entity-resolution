"""One-shard-at-a-time adapter around unchanged V2 retrieval and features."""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'distributed'
INDEX = DIST / 'shared_test_indexes'
V2 = ROOT / 'code/business_entity_resolution/v2'
sys.path.insert(0, str(V2))
from shards import file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha_text(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def hardware():
    import common  # Supplies the pinned V2 package path on the original validation host.
    import psutil
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    gpu = bool(shutil.which('nvidia-smi'))
    # Sparse CPU route is frozen; GPU detection is diagnostic only.
    threads = max(1, min(4, (os.cpu_count() or 1) // 2, int(max(1, (ram.available - 2 * 2**30) // 2**30))))
    return {'logical_cpus': os.cpu_count(), 'physical_ram': ram.total,
            'available_ram': ram.available, 'free_disk': disk.free,
            'gpu_available': gpu, 'native_threads': threads}


def configure():
    import common
    import sparse_retrieval as sparse
    import targeted
    import pair_features as pair
    import sqlite3
    def connection(src):
        db = INDEX / f's{src}.sqlite'
        return sqlite3.connect('file:' + db.resolve().as_posix() + '?mode=ro', uri=True)
    common.connection = connection
    common.V2 = INDEX
    # Keep original training-only IDF for the exact 80 feature semantics.
    pair.connection = connection
    pair.V1 = ROOT / 'artifacts/v1'
    sparse.V2 = INDEX
    sparse.connection = connection
    targeted.V2 = INDEX
    return sparse, pair


def preflight(verify_files=True):
    import common
    import importlib.metadata as metadata
    import lightgbm as lgb
    from pair_features import FEATURES
    from core import VERSION
    frozen_dir = ROOT / 'artifacts/v2/run'
    artifacts = read(frozen_dir / 'artifact_manifest.json')
    for name in ('lightgbm.txt', 'frozen_selection.json'):
        if file_hash(frozen_dir / name) != artifacts[name]['sha256']:
            raise ValueError(f'Frozen {name} checksum mismatch')
    frozen = read(frozen_dir / 'frozen_selection.json')
    if frozen['selected'] != 'lightgbm' or frozen['models']['lightgbm']['thresholds'] != [0.642, 0.642]:
        raise ValueError('Frozen model or threshold mismatch')
    model = lgb.Booster(model_file=str(frozen_dir / 'lightgbm.txt'))
    if len(FEATURES) != 80 or FEATURES != frozen['features'] or model.feature_name() != FEATURES:
        raise ValueError('Frozen feature names/order mismatch')
    codes = read(frozen_dir / 'code_fingerprints.json')
    for rel in ('common.py', 'sparse_retrieval.py', 'hashed_index.py', 'targeted.py', 'pair_features.py'):
        path = V2 / rel
        if file_hash(path) != codes[str(path.relative_to(ROOT)).replace('/', '\\')]:
            raise ValueError(f'Frozen V2 code changed: {rel}')
    v1codes = read(ROOT / 'artifacts/v1/run_6000/code_fingerprints.json')
    for rel in ('core.py', 'retrieval.py', 'prepare.py'):
        path = ROOT / 'code/business_entity_resolution/src' / rel
        if file_hash(path) != v1codes[str(path.relative_to(ROOT)).replace('/', '\\')]:
            raise ValueError(f'Frozen shared code changed: {rel}')
    requirements = (V2 / 'requirements.txt').read_text().splitlines()
    for line in requirements:
        if line.strip():
            name, version = line.split('==')
            if metadata.version(name) != version:
                raise ValueError(f'Dependency version mismatch: {name}')
    portable = read(INDEX / 'portable_manifest.json')
    if portable['normalization_version'] != VERSION:
        raise ValueError('Normalization version mismatch')
    if portable['retrieval_config_fingerprint'] != sha_text([portable['sources'][str(s)]['retrieval_config'] for s in (2, 3)]) or portable['normalization_config_fingerprint'] != sha_text([portable['sources'][str(s)]['normalization_config'] for s in (2, 3)]):
        raise ValueError('Portable index configuration fingerprint mismatch')
    if portable['transliteration_config_fingerprint'] != sha_text({'library': 'anyascii', 'version': '0.3.3', 'routes': ['translit_char', 'missing_name_char', 'joint_name_address']}):
        raise ValueError('Transliteration configuration mismatch')
    for src in (2, 3):
        path = ROOT / f'dataset/test/test_source{src}.tsv'
        expected = portable['sources'][str(src)]['dataset_fingerprint']
        if path.stat().st_size != expected['size'] or file_hash(path) != expected['sha256']:
            raise ValueError(f'Official test S{src} dataset mismatch')
        with open(path, encoding='utf-8') as f:
            if f.readline().rstrip('\r\n').split('\t') != ['entity_id', 'business_name', 'business_address', 'country']:
                raise ValueError(f'Test S{src} schema mismatch')
        hashed = read(INDEX / f'index2_s{src}/manifest.json')
        base = read(INDEX / f'index_s{src}/manifest.json')
        if hashed['config'] != portable['sources'][str(src)]['retrieval_config'] or base['config'] != portable['sources'][str(src)]['normalization_config']:
            raise ValueError(f'Test S{src} index configuration mismatch')
    if verify_files:
        for rel, info in portable['files'].items():
            path = INDEX / rel
            if not path.is_file() or path.stat().st_size != info['bytes'] or file_hash(path) != info['sha256']:
                raise ValueError(f'Test index file mismatch: {rel}')
    source1 = ROOT / 'dataset/test/test_source1.tsv'
    shard_check = read(DIST / 'shard_verification.json')
    if file_hash(source1) != shard_check['source_sha256']:
        raise ValueError('Official test S1 fingerprint mismatch')
    for src in (2, 3):
        if not (ROOT / f'artifacts/v1/s{src}_idf.json').is_file():
            raise ValueError('Frozen training IDF missing')
    return {'model_checksum': artifacts['lightgbm.txt']['sha256'],
            'threshold': 0.642,
            'feature_config_checksum': sha_text({'names': FEATURES, 'pair_code': file_hash(V2 / 'pair_features.py'), 'shared_code': file_hash(ROOT / 'code/business_entity_resolution/src/retrieval.py')}),
            'retrieval_config_checksum': sha_text({'indexes': [portable['sources'][str(s)]['retrieval_config'] for s in (2, 3)],
                                                   'sparse_code': file_hash(V2 / 'sparse_retrieval.py'), 'targeted_code': file_hash(V2 / 'targeted.py')}),
            'normalization_config_checksum': sha_text({'indexes': [portable['sources'][str(s)]['normalization_config'] for s in (2, 3)],
                                                       'core_code': file_hash(ROOT / 'code/business_entity_resolution/src/core.py')}),
            'dataset_fingerprints': {'1': shard_check['source_sha256'], **{str(s): portable['sources'][str(s)]['dataset_fingerprint']['sha256'] for s in (2, 3)}},
            'test_index_fingerprint': file_hash(INDEX / 'portable_manifest.json')}


def manifest_rows():
    with open(DIST / 'shard_manifest.tsv', encoding='utf-8', newline='') as f:
        return {row['shard_id']: row for row in csv.DictReader(f, delimiter='\t')}


def shard_records(sid, meta):
    from core import rows
    path = DIST / f's1_shards/shard_{sid}.tsv'
    records = list(rows(path))
    h = hashlib.sha256()
    for row in records:
        h.update(row['entity_id'].encode() + b'\n')
    if len(records) != int(meta['source1_count']) or h.hexdigest() != meta['source1_fingerprint']:
        raise ValueError(f'Shard {sid} S1 fingerprint mismatch')
    return records


def score_source(src, records, work, model, threads):
    import numpy as np
    t = time.perf_counter()
    n = len(records)
    candidates = work / f's{src}_candidate.tsv'
    matches = work / f's{src}_matching.tsv'
    count_pairs = count_links = 0
    with open(candidates, 'w', encoding='utf-8', newline='') as cf, open(matches, 'w', encoding='utf-8', newline='') as mf:
        for start in range(0, n, 25):
            path = work / f's{src}_{start:05d}.npz'
            with np.load(path) as z:
                X, q, eid = z['X'], z['q'], z['eid']
            scores = model.predict(X, num_threads=threads) if len(X) else np.empty(0)
            for qi in range(start, min(n, start+25)):
                ix = np.flatnonzero(q == qi)
                all_ids = [str(eid[j]) for j in ix]
                selected = [str(eid[j]) for j in ix if scores[j] >= 0.642]
                if len(set(all_ids)) != len(all_ids) or not set(selected) <= set(all_ids):
                    raise ValueError('Candidate or prediction duplication')
                cf.write(records[qi]['entity_id'] + '\t' + ','.join(all_ids) + '\n')
                mf.write(records[qi]['entity_id'] + '\t' + ','.join(selected) + '\n')
                count_pairs += len(all_ids)
                count_links += len(selected)
            path.unlink()
            path.with_suffix('.features.json').unlink()
    return {'seconds': time.perf_counter()-t, 'pairs': count_pairs, 'links': count_links}


def combine(records, work, column, final):
    with open(work / f's2_{column}.tsv', encoding='utf-8') as a, open(work / f's3_{column}.tsv', encoding='utf-8') as b, open(final, 'w', encoding='utf-8', newline='') as out:
        out.write('source1_entity_id\t' + ('candidate_entity_ids' if column == 'candidate' else 'matched_entity_ids') + '\n')
        for row in records:
            ea, va = a.readline().rstrip('\n').split('\t', 1)
            eb, vb = b.readline().rstrip('\n').split('\t', 1)
            if ea != row['entity_id'] or eb != ea:
                raise ValueError('Source output alignment mismatch')
            ids = [x for x in (va + ',' + vb).split(',') if x]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate cross-source ID')
            out.write(ea + '\t' + ','.join(ids) + '\n')
        if a.readline() or b.readline():
            raise ValueError('Excess source rows')


def run_features(records, src, work, pair, feature_workers):
    if feature_workers == 1:
        pair.run(records, src, work)
        return 0
    import psutil
    records_path = work / 'records.json'
    records_path.write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    children = []
    for part in range(feature_workers):
        child_log = open(work / f's{src}_feature_{part}.log', 'w', encoding='utf-8')
        process = subprocess.Popen([sys.executable, str(DIST / 'feature_child.py'), '--records', str(records_path),
                                    '--output', str(work), '--source', str(src), '--part', str(part),
                                    '--parts', str(feature_workers)], stdout=child_log, stderr=subprocess.STDOUT)
        children.append((process, child_log))
    codes = [process.wait() for process, _ in children]
    for _, child_log in children:
        child_log.close()
    records_path.unlink()
    if any(codes):
        raise RuntimeError(f'Feature child failed for S{src}: {codes}')
    child_peaks = [read(work / f's{src}_features_complete_{part}.json')['memory'].get('peak_rss_bytes', 0) for part in range(feature_workers)]
    return psutil.Process().memory_info().rss + sum(child_peaks)


def verified(sid, pre, meta):
    folder = DIST / f'results/shard_{sid}'
    mf = folder / 'shard_manifest.json'
    if not mf.exists():
        return False
    m = read(mf)
    if m.get('status') != 'VERIFIED_COMPLETE' or m['source1_fingerprint'] != meta['source1_fingerprint']:
        return False
    for key in ('model_checksum', 'threshold', 'feature_config_checksum', 'retrieval_config_checksum',
                'normalization_config_checksum', 'dataset_fingerprints', 'test_index_fingerprint'):
        if m.get(key) != pre[key]:
            raise ValueError(f'Completed shard {sid} has mismatched {key}')
    for name in ('matching_part.tsv', 'candidate_part.tsv'):
        if file_hash(folder / name) != m[name + '_sha256']:
            raise ValueError(f'Completed shard {sid} checksum failed')
    return True


def run_shard(sid, worker_name, pre, meta, threads):
    import lightgbm as lgb
    import psutil
    from prepare import memory
    sparse, pair = configure()
    from sparse_dot_topn import sp_matmul_topn
    # Frozen sparse function and arguments, with a bounded native thread count.
    def bounded(*args, **kwargs):
        available = psutil.virtual_memory().available
        kwargs['n_threads'] = 1 if available < 3 * 2**30 else threads
        if available < 2 * 2**30:
            raise MemoryError('Stopped sparse route before unsafe paging')
        return sp_matmul_topn(*args, **kwargs)
    sparse.sp_matmul_topn = bounded
    import targeted
    targeted.sp_matmul_topn = bounded
    records = shard_records(sid, meta)
    working_root = DIST / 'results/.working'
    working_root.mkdir(parents=True, exist_ok=True)
    work = working_root / f'shard_{sid}'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    log = open(work / 'worker.log', 'w', encoding='utf-8')
    model = lgb.Booster(model_file=str(ROOT / 'artifacts/v2/run/lightgbm.txt'))
    timings = {'retrieval': 0., 'feature': 0., 'scoring': 0., 'serialization': 0.}
    pairs = links = 0
    peak_bound = 0
    start = time.perf_counter()
    try:
        for src in (2, 3):
            t = time.perf_counter()
            sparse.retrieve(records, src, work, k=100, policy='tiered')
            timings['retrieval'] += time.perf_counter()-t
            t = time.perf_counter()
            available = psutil.virtual_memory().available
            feature_workers = (4 if (os.cpu_count() or 1) >= 8 and available >= 6 * 2**30 else
                               2 if (os.cpu_count() or 1) >= 4 and available >= int(4.5 * 2**30) else 1)
            peak_bound = max(peak_bound, run_features(records, src, work, pair, feature_workers))
            timings['feature'] += time.perf_counter()-t
            s = score_source(src, records, work, model, threads)
            timings['scoring'] += s['seconds']
            pairs += s['pairs']
            links += s['links']
            # Intermediate retrieval artifacts are not needed after exact candidates are scored.
            for path in work.glob(f's{src}_*.npz'):
                path.unlink()
            for path in work.glob(f's{src}_*.json'):
                path.unlink()
            (work / f's{src}_candidates.jsonl.gz').unlink()
            gc.collect()
        t = time.perf_counter()
        combine(records, work, 'candidate', work / 'candidate_part.tsv')
        combine(records, work, 'matching', work / 'matching_part.tsv')
        for src in (2, 3):
            for column in ('candidate', 'matching'):
                (work / f's{src}_{column}.tsv').unlink()
        from merge import validate_pair
        checked_pairs, checked_links = validate_pair(work / 'matching_part.tsv', work / 'candidate_part.tsv', {r['entity_id'] for r in records}, INDEX)
        if checked_pairs != pairs or checked_links != links:
            raise ValueError('Shard score/output count mismatch')
        timings['serialization'] = time.perf_counter()-t
        m = {'shard_id': sid, 'worker_name': worker_name, 'source1_count': len(records),
             'source1_fingerprint': meta['source1_fingerprint'], 'candidate_pair_count': pairs,
             'predicted_link_count': links, 'retrieval_runtime': timings['retrieval'],
             'feature_runtime': timings['feature'], 'scoring_runtime': timings['scoring'],
             'serialization_runtime': timings['serialization'], 'end_to_end_runtime': time.perf_counter()-start,
             'peak_ram': max(peak_bound, memory().get('peak_rss_bytes', psutil.Process().memory_info().rss)), **pre,
             'matching_part.tsv_sha256': file_hash(work / 'matching_part.tsv'),
             'candidate_part.tsv_sha256': file_hash(work / 'candidate_part.tsv'),
             'status': 'VERIFIED_COMPLETE'}
        (work / 'shard_manifest.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
        log.write(json.dumps(m, indent=2) + '\n')
        log.close()
        final = DIST / f'results/shard_{sid}'
        if final.exists():
            raise ValueError('Result directory already exists')
        work.replace(final)
        print('VERIFIED_COMPLETE', sid, pairs, links, flush=True)
    finally:
        if not log.closed:
            log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker-name', required=True)
    p.add_argument('--shard-ids')
    p.add_argument('--worker-index', type=int)
    p.add_argument('--worker-count', type=int)
    p.add_argument('--max-shards', type=int)
    a = p.parse_args()
    if a.shard_ids is None and (a.worker_index is None or a.worker_count is None):
        p.error('Provide --shard-ids or --worker-index and --worker-count')
    info = hardware()
    print('hardware', json.dumps(info), flush=True)
    if info['available_ram'] < 4 * 2**30 or info['free_disk'] < 2 * 2**30:
        raise RuntimeError('Insufficient available RAM or disk headroom')
    pre = preflight()
    meta = manifest_rows()
    ids = [x.zfill(4) for x in a.shard_ids.split(',')] if a.shard_ids else [x for x in sorted(meta) if int(x) % a.worker_count == a.worker_index]
    if any(x not in meta for x in ids):
        raise ValueError('Unknown shard ID')
    completed = 0
    for sid in ids:
        if verified(sid, pre, meta[sid]):
            print('skip VERIFIED_COMPLETE', sid, flush=True)
            continue
        if a.max_shards is not None and completed >= a.max_shards:
            break
        available = hardware()['available_ram']
        if available < 4 * 2**30:
            raise RuntimeError('Memory pressure unsafe; stop before next shard')
        threads = min(info['native_threads'], max(1, int((available - 2 * 2**30) // 2**30)))
        run_shard(sid, a.worker_name, pre, meta[sid], threads)
        completed += 1
        gc.collect()


if __name__ == '__main__':
    main()
