"""Stable, streaming partition of official test Source 1."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'distributed'
COUNT = 160
HEADER = ['entity_id', 'business_name', 'business_address', 'country']


def shard_id(entity_id):
    return int.from_bytes(hashlib.sha256(('amazon-ml-2026-s1-v1|' + entity_id).encode()).digest()[:8], 'big') % COUNT


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build():
    source = ROOT / 'dataset/test/test_source1.tsv'
    out = DIST / 's1_shards'
    out.mkdir(parents=True, exist_ok=True)
    handles = []
    counts = [0] * COUNT
    first = [None] * COUNT
    last = [None] * COUNT
    hashes = [hashlib.sha256() for _ in range(COUNT)]
    try:
        for i in range(COUNT):
            f = open(out / f'shard_{i:04d}.tsv.tmp', 'w', encoding='utf-8', newline='')
            handles.append(f)
            f.write('\t'.join(HEADER) + '\n')
        seen = set()
        with open(source, encoding='utf-8', newline='') as f:
            if f.readline().rstrip('\r\n').split('\t') != HEADER:
                raise ValueError('Unexpected official S1 schema')
            for line in f:
                eid = line.split('\t', 1)[0]
                if eid in seen:
                    raise ValueError(f'Duplicate official S1 ID: {eid}')
                seen.add(eid)
                i = shard_id(eid)
                handles[i].write(line)
                counts[i] += 1
                first[i] = eid if first[i] is None else first[i]
                last[i] = eid
                hashes[i].update(eid.encode() + b'\n')
    finally:
        for f in handles:
            f.close()
    manifest = DIST / 'shard_manifest.tsv'
    with open(manifest.with_suffix('.tsv.tmp'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['shard_id', 'source1_count', 'source1_fingerprint', 'first_source1_id', 'last_source1_id', 'status'])
        for i in range(COUNT):
            (out / f'shard_{i:04d}.tsv.tmp').replace(out / f'shard_{i:04d}.tsv')
            w.writerow([f'{i:04d}', counts[i], hashes[i].hexdigest(), first[i], last[i], 'UNSTARTED'])
    manifest.with_suffix('.tsv.tmp').replace(manifest)
    # Independent second scan verifies coverage, exclusivity, content and assignment.
    check = set()
    for i in range(COUNT):
        local = hashlib.sha256()
        with open(out / f'shard_{i:04d}.tsv', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            if reader.fieldnames != HEADER:
                raise ValueError('Shard schema mismatch')
            n = 0
            for row in reader:
                eid = row['entity_id']
                if shard_id(eid) != i or eid in check:
                    raise ValueError('Shard overlap or wrong assignment')
                check.add(eid)
                local.update(eid.encode() + b'\n')
                n += 1
            if n != counts[i] or local.hexdigest() != hashes[i].hexdigest():
                raise ValueError('Shard content mismatch')
    if check != seen:
        raise ValueError('Shard union mismatch')
    report = {'algorithm': 'SHA256 first 64 bits of amazon-ml-2026-s1-v1|entity_id modulo 160',
              'source_sha256': file_hash(source), 'source1_count': len(seen), 'shard_count': COUNT,
              'union_equals_official': True, 'pairwise_disjoint': True, 'exactly_once': True,
              'total_shard_rows': sum(counts)}
    (DIST / 'shard_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    build()
