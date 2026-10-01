"""Checkpointed single-machine fast submission profile using verified V2 assets.

The original V2 code/model is untouched. This module omits the expensive name-word,
address-character and two targeted rescue routes, retaining V2's name-character,
address-word, joint, exact-name, suffix-name, and address-key routes.
"""
import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import sqlite3
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/v2/packages'))
sys.path.insert(0, str(ROOT / 'distributed'))
from worker import INDEX, configure, preflight, combine, file_hash, hardware

ROUTES = ('name_char', 'address_word', 'joint_name_address', 'exact_name', 'suffix_name', 'address_key')
PROFILE = 'V2_FAST_DEADLINE_R10'
RANK = 10
EXPECTED_ROWS = 1_732_544
RUN = ROOT / 'distributed/deadline'
CHECKPOINTS = RUN / 'checkpoints'


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def fast_retrieve(records, src, out, threads):
    """V2 sparse products and joins for the measured deadline route subset."""
    import numpy as np
    import sparse_retrieval as sr
    from scipy import sparse
    from sparse_dot_topn import sp_matmul_topn
    start = time.perf_counter()
    n = len(records)
    index = INDEX / f'index2_s{src}'
    meta = sr.read(index / 'manifest.json')
    views = [sr.view(r) for r in records]
    vectors = {}
    queries = {}
    for route in ('name_char', 'address_word'):
        with open(index / f'{route}.pkl', 'rb') as f:
            vectors[route] = pickle.load(f)
        queries[route] = vectors[route].transform([v['nf' if route == 'name_char' else 'af'] for v in views])
    joint_query = sparse.hstack([queries['name_char'] * .5, queries['address_word'] * .5], format='csr')
    best = {r: (np.empty((n, 0), np.int32), np.empty((n, 0), np.float32)) for r in ROUTES[:3]}
    route_seconds = {'sparse': 0., 'exact': 0., 'union': 0.}
    t = time.perf_counter()
    for name_path in sorted(index.glob('*_name_char.npz')):
        offset = int(name_path.name.split('_')[0])
        name_matrix = sparse.load_npz(name_path)
        address_matrix = sparse.load_npz(index / f'{offset:08d}_address_word.npz')
        for route, query, matrix, k, threshold in (
            ('name_char', queries['name_char'], name_matrix, RANK, .015),
            ('address_word', queries['address_word'], address_matrix, RANK, .015),
            ('joint_name_address', joint_query, sparse.vstack([name_matrix, address_matrix], format='csr'), RANK, .08),
        ):
            if route == 'joint_name_address':
                product = sp_matmul_topn(query, matrix, top_n=k, threshold=threshold, sort=True, n_threads=threads)
            else:
                product = sp_matmul_topn(query, matrix, top_n=k, threshold=threshold, sort=True, n_threads=threads)
            ids = np.zeros((n, k), np.int32)
            scores = np.zeros((n, k), np.float32)
            for i in range(n):
                lo, hi = product.indptr[i:i+2]
                length = hi - lo
                ids[i, :length] = product.indices[lo:hi] + offset + 1
                scores[i, :length] = product.data[lo:hi]
            best[route] = sr.top_merge(*best[route], ids, scores, k)
            del product, ids, scores
            if route == 'joint_name_address':
                del matrix
        del name_matrix, address_matrix
        gc.collect()
    route_seconds['sparse'] = time.perf_counter() - t
    del queries, joint_query, vectors
    gc.collect()

    t = time.perf_counter()
    con = sqlite3.connect('file:' + (INDEX / f's{src}.sqlite').resolve().as_posix() + '?mode=ro', uri=True)
    con.execute('CREATE TEMP TABLE queries(q INTEGER,n TEXT,s TEXT,ak TEXT)')
    con.executemany('INSERT INTO queries VALUES(?,?,?,?)',
                    [(i, v['nf'], v['s'], ' '.join(sorted(v['at']))) for i, v in enumerate(views)])
    from rapidfuzz import fuzz
    exact = {}
    freq = np.zeros((n, 2), np.int32)
    for route, col in (('exact_name', 'n'), ('suffix_name', 's'), ('address_key', 'ak')):
        hits = [[] for _ in records]
        statement = f"SELECT q.q,r.rowid,r.a FROM queries q JOIN r ON r.{col}=q.{col} WHERE q.{col}<>'' AND r.rowid<=?"
        for qi, rid, addr in con.execute(statement, (meta['rows'],)):
            hits[qi].append((rid, fuzz.token_set_ratio(views[qi]['af'], addr) / 100))
        for i, hit in enumerate(hits):
            if route != 'address_key':
                freq[i, 0 if route == 'exact_name' else 1] = len(hit)
            hit.sort(key=lambda x: (-x[1], x[0]))
            hits[i] = hit[:RANK]
        exact[route] = hits
    con.close()
    route_seconds['exact'] = time.perf_counter() - t

    t = time.perf_counter()
    candidate_path = out / f's{src}_candidates.jsonl.gz'
    with gzip.open(candidate_path, 'wt', encoding='utf-8', compresslevel=1) as f:
        for i, row in enumerate(records):
            found = {}
            for route, (ids, scores) in best.items():
                for rank, (rid, score) in enumerate(zip(ids[i], scores[i]), 1):
                    if rid:
                        found.setdefault(int(rid), {})[route] = [rank, float(score)]
            for route, all_hits in exact.items():
                for rank, (rid, score) in enumerate(all_hits[i], 1):
                    found.setdefault(int(rid), {})[route] = [rank, float(score)]
            f.write(json.dumps({'q': i, 'entity_id': row['entity_id'], 'nf': int(freq[i, 0]),
                                'sf': int(freq[i, 1]), 'candidates': found}) + '\n')
    route_seconds['union'] = time.perf_counter() - t
    save_json(out / f's{src}_complete.json', {'profile': PROFILE, 'routes': ROUTES, 'rank': RANK,
                                               'seconds': time.perf_counter() - start,
                                               'timings': route_seconds})
    print('retrieved', src, n, round(time.perf_counter() - start, 1), route_seconds, flush=True)


def verify_completed(path, pre, start_row, records):
    mf = path / 'complete.json'
    if not mf.is_file():
        return False
    m = json.loads(mf.read_text(encoding='utf-8'))
    if m.get('status') != 'VERIFIED_COMPLETE' or m.get('start_row') != start_row or m.get('source1_count') != len(records):
        raise ValueError(f'Existing checkpoint is incomplete or mismatched: {path}')
    if m.get('profile') != PROFILE or m.get('preflight') != pre:
        raise ValueError(f'Existing checkpoint config mismatch: {path}')
    h = hashlib.sha256()
    for r in records:
        h.update((r['entity_id'] + '\n').encode())
    if m['source1_fingerprint'] != h.hexdigest():
        raise ValueError(f'Existing checkpoint S1 mismatch: {path}')
    for filename in ('matching_part.tsv', 'candidate_part.tsv'):
        if file_hash(path / filename) != m[filename + '_sha256']:
            raise ValueError(f'Existing checkpoint hash mismatch: {path}')
    return True


def score_source_fast(src, records, work, model, threads):
    """One LightGBM call per source batch avoids hundreds of model setups."""
    import numpy as np
    started = time.perf_counter()
    paths = sorted(work.glob(f's{src}_*.npz'))
    if not paths:
        raise ValueError(f'No feature files for S{src}')
    features = []
    query_index = []
    entity_ids = []
    for path in paths:
        with np.load(path) as z:
            features.append(z['X'])
            query_index.append(z['q'])
            entity_ids.append(z['eid'])
    X = np.concatenate(features)
    q = np.concatenate(query_index)
    eid = np.concatenate(entity_ids)
    del features, query_index, entity_ids
    if len(X) and (X.shape[1] != 80 or not np.isfinite(X).all()):
        raise ValueError('Invalid frozen feature matrix')
    scores = model.predict(X, num_threads=threads) if len(X) else np.empty(0)
    del X
    counts = np.bincount(q, minlength=len(records))
    boundaries = np.r_[0, np.cumsum(counts)]
    candidates = work / f's{src}_candidate.tsv'
    matching = work / f's{src}_matching.tsv'
    links = 0
    with open(candidates, 'w', encoding='utf-8', newline='') as cf, open(matching, 'w', encoding='utf-8', newline='') as mf:
        for i, record in enumerate(records):
            lo, hi = boundaries[i:i+2]
            ids = eid[lo:hi].tolist()
            chosen = [target for target, score in zip(ids, scores[lo:hi]) if score >= .642]
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate scored candidate ID')
            cf.write(record['entity_id'] + '\t' + ','.join(ids) + '\n')
            mf.write(record['entity_id'] + '\t' + ','.join(chosen) + '\n')
            links += len(chosen)
    for path in paths:
        path.unlink()
        path.with_suffix('.features.json').unlink()
    return {'seconds': time.perf_counter() - started, 'pairs': len(eid), 'links': links}


def run_features_source(src, records, work, workers):
    if workers == 1:
        _, pair = configure()
        pair.run(records, src, work)
        return
    record_path = work / f'records_s{src}.json'
    record_path.write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    children = []
    logs = []
    for part in range(workers):
        log = open(work / f's{src}_feature_{part}.log', 'w', encoding='utf-8')
        process = subprocess.Popen([sys.executable, str(ROOT / 'distributed/feature_child.py'),
                                    '--records', str(record_path), '--output', str(work),
                                    '--source', str(src), '--part', str(part), '--parts', str(workers)],
                                   stdout=log, stderr=subprocess.STDOUT)
        children.append(process)
        logs.append(log)
    codes = [child.wait() for child in children]
    for log in logs:
        log.close()
    record_path.unlink()
    if any(codes):
        raise RuntimeError(f'S{src} feature child failed: {codes}')


def process_batch(records, start_row, batch_id, pre, model, threads, feature_workers):
    import psutil
    from merge import validate_pair
    from prepare import memory
    work_root = RUN / '.working'
    work_root.mkdir(parents=True, exist_ok=True)
    work = work_root / f'batch_{batch_id:06d}'
    if work.exists():
        if not work.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Unsafe working directory')
        shutil.rmtree(work)
    work.mkdir()
    begun = time.perf_counter()
    totals = {'retrieval': 0., 'features': 0., 'scoring': 0., 'serialization': 0.}
    pairs = links = 0
    record_path = work / 'records.json'
    record_path.write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    t = time.perf_counter()
    children = [subprocess.Popen([sys.executable, str(Path(__file__)), 'retrieve-child',
                                  '--source', str(src), '--records-file', str(record_path),
                                  '--work', str(work), '--threads', str(threads)]) for src in (2, 3)]
    codes = [child.wait() for child in children]
    if any(codes):
        raise RuntimeError(f'Retrieval child failed: {codes}')
    totals['retrieval'] = time.perf_counter() - t
    record_path.unlink()
    t = time.perf_counter()
    if feature_workers > 1:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_features_source, src, records, work, feature_workers) for src in (2, 3)]
            for future in futures:
                future.result()
    else:
        for src in (2, 3):
            run_features_source(src, records, work, 1)
    totals['features'] = time.perf_counter() - t
    for src in (2, 3):
        t = time.perf_counter()
        scored = score_source_fast(src, records, work, model, threads)
        totals['scoring'] += time.perf_counter() - t
        pairs += scored['pairs']
        links += scored['links']
        for path in work.glob(f's{src}_*.json'):
            path.unlink()
        (work / f's{src}_candidates.jsonl.gz').unlink()
        gc.collect()
    t = time.perf_counter()
    combine(records, work, 'matching', work / 'matching_part.tsv')
    combine(records, work, 'candidate', work / 'candidate_part.tsv')
    checked = validate_pair(work / 'matching_part.tsv', work / 'candidate_part.tsv',
                            {r['entity_id'] for r in records})
    if checked != (pairs, links):
        raise ValueError('Checkpoint output count mismatch')
    for src in (2, 3):
        for kind in ('matching', 'candidate'):
            (work / f's{src}_{kind}.tsv').unlink()
    totals['serialization'] = time.perf_counter() - t
    h = hashlib.sha256()
    for r in records:
        h.update((r['entity_id'] + '\n').encode())
    m = {'status': 'VERIFIED_COMPLETE', 'profile': PROFILE, 'routes': ROUTES,
         'start_row': start_row, 'source1_count': len(records), 'source1_fingerprint': h.hexdigest(),
         'candidate_pair_count': pairs, 'predicted_link_count': links, 'timings': totals,
         'elapsed_seconds': time.perf_counter() - begun,
         'peak_ram_bytes': memory().get('peak_rss_bytes', psutil.Process().memory_info().rss),
         'free_disk_bytes': psutil.disk_usage(str(ROOT)).free, 'preflight': pre,
         'matching_part.tsv_sha256': file_hash(work / 'matching_part.tsv'),
         'candidate_part.tsv_sha256': file_hash(work / 'candidate_part.tsv')}
    save_json(work / 'complete.json', m)
    destination = CHECKPOINTS / f'batch_{batch_id:06d}'
    if destination.exists():
        raise ValueError(f'Checkpoint already exists: {destination}')
    work.replace(destination)
    return m


def iterate_batches(batch_size):
    from core import rows
    batch = []
    start = 0
    for row in rows(ROOT / 'dataset/test/test_source1.tsv'):
        batch.append(row)
        if len(batch) == batch_size:
            yield start, batch
            start += len(batch)
            batch = []
    if batch:
        yield start, batch


def merge_checkpoints(batch_size, pre):
    from core import rows
    source = rows(ROOT / 'dataset/test/test_source1.tsv')
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    paths = {'matching': output / 'matching_results.tsv.tmp',
             'candidate': output / 'candidate_pairs.tsv.tmp'}
    count = pairs = links = 0
    with open(paths['matching'], 'w', encoding='utf-8', newline='') as mf, open(paths['candidate'], 'w', encoding='utf-8', newline='') as cf:
        mf.write('source1_entity_id\tmatched_entity_ids\n')
        cf.write('source1_entity_id\tcandidate_entity_ids\n')
        for batch_id, (start, records) in enumerate(iterate_batches(batch_size)):
            folder = CHECKPOINTS / f'batch_{batch_id:06d}'
            if not verify_completed(folder, pre, start, records):
                raise ValueError(f'Missing completed checkpoint {folder}')
            manifest = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
            pairs += manifest['candidate_pair_count']
            links += manifest['predicted_link_count']
            with open(folder / 'matching_part.tsv', encoding='utf-8') as a, open(folder / 'candidate_part.tsv', encoding='utf-8') as b:
                if a.readline().rstrip('\n') != 'source1_entity_id\tmatched_entity_ids' or b.readline().rstrip('\n') != 'source1_entity_id\tcandidate_entity_ids':
                    raise ValueError('Checkpoint header mismatch')
                for record in records:
                    ma = a.readline(); ca = b.readline()
                    if ma.split('\t', 1)[0] != record['entity_id'] or ca.split('\t', 1)[0] != record['entity_id']:
                        raise ValueError('Checkpoint output order mismatch')
                    mf.write(ma); cf.write(ca)
                    count += 1
                if a.readline() or b.readline():
                    raise ValueError('Extra checkpoint output row')
    if count != EXPECTED_ROWS:
        raise ValueError(f'Official test S1 coverage mismatch: {count}')
    paths['matching'].replace(output / 'matching_results.tsv')
    paths['candidate'].replace(output / 'candidate_pairs.tsv')
    return {'source1_rows': count, 'candidate_pairs': pairs, 'predicted_links': links}


def main():
    import lightgbm as lgb
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('run', 'merge', 'status', 'retrieve-child'))
    p.add_argument('--batch-size', type=int, default=10000)
    p.add_argument('--max-rows', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--feature-workers', type=int, default=4)
    p.add_argument('--source', type=int, choices=(2, 3))
    p.add_argument('--records-file', type=Path)
    p.add_argument('--work', type=Path)
    a = p.parse_args()
    if a.action == 'retrieve-child':
        if not a.source or not a.records_file or not a.work:
            p.error('retrieve-child requires --source, --records-file, and --work')
        configure()
        fast_retrieve(json.loads(a.records_file.read_text(encoding='utf-8')), a.source, a.work, a.threads)
        return
    if a.batch_size <= 0 or a.batch_size % 25:
        p.error('--batch-size must be a positive multiple of 25')
    if a.action == 'status':
        complete = sorted(CHECKPOINTS.glob('batch_*/complete.json'))
        count = sum(json.loads(path.read_text())['source1_count'] for path in complete)
        print(json.dumps({'complete_batches': len(complete), 'source1_complete': count,
                          'source1_remaining': EXPECTED_ROWS - count}, indent=2))
        return
    pre = preflight(verify_files=True)
    if a.action == 'merge':
        print(json.dumps(merge_checkpoints(a.batch_size, pre), indent=2))
        return
    hw = hardware()
    print('hardware', json.dumps(hw), flush=True)
    if hw['available_ram'] < 4 * 2**30:
        raise RuntimeError('Need at least 4 GiB available RAM before deadline run')
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    model = lgb.Booster(model_file=str(ROOT / 'artifacts/v2/run/lightgbm.txt'))
    started = time.perf_counter()
    done = 0
    prior_seconds = 0.
    for batch_id, (start_row, records) in enumerate(iterate_batches(a.batch_size)):
        folder = CHECKPOINTS / f'batch_{batch_id:06d}'
        if verify_completed(folder, pre, start_row, records):
            done += len(records)
            prior_seconds += json.loads((folder / 'complete.json').read_text(encoding='utf-8'))['elapsed_seconds']
            continue
        if a.max_rows and done >= a.max_rows:
            break
        avail = hardware()['available_ram']
        if avail < 3 * 2**30:
            raise RuntimeError('Unsafe RAM pressure before next checkpoint')
        workers = min(a.feature_workers, 4 if avail >= 5 * 2**30 else 2)
        m = process_batch(records, start_row, batch_id, pre, model, a.threads, workers)
        done += len(records)
        elapsed = time.perf_counter() - started
        rate = done / (prior_seconds + elapsed)
        print('CHECKPOINT', json.dumps({'batch': batch_id, 'done': done, 'remaining': EXPECTED_ROWS-done,
                                        'pairs': m['candidate_pair_count'], 'links': m['predicted_link_count'],
                                        'seconds': m['elapsed_seconds'], 'rate_s1_per_second': rate,
                                        'eta_hours': (EXPECTED_ROWS-done)/rate/3600,
                                        'free_ram': hardware()['available_ram'], 'free_disk': m['free_disk_bytes']}), flush=True)
        gc.collect()
    if done == EXPECTED_ROWS:
        print('INFERENCE_COMPLETE', json.dumps(merge_checkpoints(a.batch_size, pre)), flush=True)


if __name__ == '__main__':
    main()
