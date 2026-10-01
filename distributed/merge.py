"""Deterministic streaming merge with strict shard and ID checks."""
import argparse
import csv
import gc
import json
import sqlite3
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'distributed'
sys.path.insert(0, str(DIST))
from shards import file_hash


def parse(path, header):
    with open(path, encoding='utf-8', newline='') as f:
        if f.readline().rstrip('\r\n') != 'source1_entity_id\t' + header:
            raise ValueError(f'Incorrect TSV header: {path}')
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) != 2:
                raise ValueError(f'Malformed TSV row: {path}')
            ids = [] if not parts[1] else parts[1].split(',')
            if len(ids) != len(set(ids)) or any(not x.startswith(('S2-', 'S3-')) for x in ids):
                raise ValueError(f'Invalid ID list: {path}')
            yield parts[0], ids


def validate_pair(matching, candidate, expected, index=None, target_ids=None):
    seen = set()
    n_pairs = n_links = 0
    for a, b in zip_longest(parse(matching, 'matched_entity_ids'), parse(candidate, 'candidate_entity_ids')):
        if a is None or b is None:
            raise ValueError('Matching/candidate TSV row counts differ')
        m_id, matches = a
        c_id, candidates = b
        if m_id != c_id or m_id in seen or m_id not in expected or not set(matches) <= set(candidates):
            raise ValueError('Shard rows, coverage or prediction subset mismatch')
        if target_ids is not None and any(x not in target_ids for x in candidates):
            raise ValueError('Candidate ID absent from official test S2/S3')
        seen.add(m_id)
        n_pairs += len(candidates)
        n_links += len(matches)
    if seen != expected:
        raise ValueError('Shard S1 coverage incomplete')
    return n_pairs, n_links


def rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        if reader.fieldnames != ['entity_id', 'business_name', 'business_address', 'country']:
            raise ValueError(f'Unexpected S1 schema: {path}')
        for row in reader:
            yield row['entity_id']


def merge(results_dir, output):
    from worker import manifest_rows, preflight
    pre = preflight()
    meta = manifest_rows()
    folders = sorted(results_dir.glob('shard_*'))
    observed = {}
    for folder in folders:
        mpath = folder / 'shard_manifest.json'
        if not mpath.exists():
            raise ValueError(f'Incomplete shard directory: {folder}')
        m = json.loads(mpath.read_text(encoding='utf-8'))
        sid = m['shard_id']
        if sid in observed or sid not in meta or m['status'] != 'VERIFIED_COMPLETE':
            raise ValueError(f'Duplicate, unknown or incomplete shard: {sid}')
        for key in ('model_checksum', 'threshold', 'feature_config_checksum', 'retrieval_config_checksum',
                    'normalization_config_checksum', 'dataset_fingerprints', 'test_index_fingerprint'):
            if m.get(key) != pre[key]:
                raise ValueError(f'Shard {sid} {key} mismatch')
        for name in ('matching_part.tsv', 'candidate_part.tsv'):
            if file_hash(folder / name) != m[name + '_sha256']:
                raise ValueError(f'Shard {sid} output checksum mismatch')
        if m['source1_fingerprint'] != meta[sid]['source1_fingerprint']:
            raise ValueError(f'Shard {sid} S1 fingerprint mismatch')
        observed[sid] = folder
    if set(observed) != set(meta):
        raise ValueError(f'Missing shards: {sorted(set(meta)-set(observed))[:20]}')
    all_s1 = set(rows(ROOT / 'dataset/test/test_source1.tsv'))
    if len(all_s1) != 1732544:
        raise ValueError('Official S1 count mismatch')
    target_ids = set()
    for src in (2, 3):
        with open(ROOT / f'dataset/test/test_source{src}.tsv', encoding='utf-8') as f:
            if f.readline().rstrip('\r\n').split('\t') != ['entity_id', 'business_name', 'business_address', 'country']:
                raise ValueError('Official target schema mismatch')
            target_ids.update(line.split('\t', 1)[0] for line in f)
    output.mkdir(parents=True, exist_ok=True)
    matching = output / 'matching_results.tsv.tmp'
    candidate = output / 'candidate_pairs.tsv.tmp'
    seen = set()
    with open(matching, 'w', encoding='utf-8', newline='') as mout, open(candidate, 'w', encoding='utf-8', newline='') as cout:
        mout.write('source1_entity_id\tmatched_entity_ids\n')
        cout.write('source1_entity_id\tcandidate_entity_ids\n')
        for sid in sorted(observed):
            folder = observed[sid]
            expected = set(rows(DIST / f's1_shards/shard_{sid}.tsv'))
            pairs, links = validate_pair(folder / 'matching_part.tsv', folder / 'candidate_part.tsv', expected, target_ids=target_ids)
            manifest = json.loads((folder / 'shard_manifest.json').read_text(encoding='utf-8'))
            if pairs != manifest['candidate_pair_count'] or links != manifest['predicted_link_count']:
                raise ValueError(f'Shard {sid} count mismatch')
            if seen & expected:
                raise ValueError(f'Duplicate S1 across shards: {sid}')
            seen.update(expected)
            with open(folder / 'matching_part.tsv', encoding='utf-8') as source:
                next(source)
                for line in source:
                    mout.write(line)
            with open(folder / 'candidate_part.tsv', encoding='utf-8') as source:
                next(source)
                for line in source:
                    cout.write(line)
    if seen != all_s1:
        raise ValueError('Merged S1 coverage mismatch')
    del target_ids, all_s1, seen
    gc.collect()
    # The organizer helper materializes all candidate lists and can exceed 16 GiB.
    # Candidate TSV receives stricter independent streaming checks above.
    cmd = [sys.executable, str(ROOT / 'utils/validate_submission.py'), '--matching', str(matching),
           '--candidate', str(output / '__independently_validated_candidate_file__'),
           '--test-dir', str(ROOT / 'dataset/test'), '--check-ids']
    result = subprocess.run(cmd, text=True)
    if result.returncode:
        raise ValueError('Organizer validator did not PASS')
    matching.replace(output / 'matching_results.tsv')
    candidate.replace(output / 'candidate_pairs.tsv')
    print('PASS: merged all official S1 rows; candidate TSV independently validated; organizer matching --check-ids PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path, default=DIST / 'results')
    p.add_argument('--output', type=Path, default=ROOT / 'output')
    a = p.parse_args()
    merge(a.results_dir, a.output)
