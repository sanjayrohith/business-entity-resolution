"""Prepare a new training-only grouped validation sample and zero-copy index links."""
import hashlib
import heapq
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/business_entity_resolution/v2'))
from common import rows, view

OUT = ROOT / 'distributed/recovery'
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE = OUT / 'fresh_split.json'


def keys(record):
    v = view(record)
    return [('name', v['s']), ('address', ' '.join(sorted(v['at'])))]


def main():
    if SAMPLE.exists():
        print('REUSE', SAMPLE, flush=True)
        return
    old = json.loads((ROOT / 'artifacts/v1/split_6000.json').read_text(encoding='utf-8'))['records']
    old += json.loads((ROOT / 'artifacts/v2/split.json').read_text(encoding='utf-8'))['records']
    old_ids = {r['entity_id'] for r in old}
    old_keys = {key for r in old for key in keys(r) if key[1]}
    heap = []
    for record in rows(ROOT / 'dataset/train/train_source1.tsv'):
        sid = record['entity_id']
        if sid in old_ids:
            continue
        h = int.from_bytes(hashlib.blake2b(('recovery-20260927|'+sid).encode(), digest_size=8).digest(), 'big')
        if len(heap) < 20000:
            heapq.heappush(heap, (-h, sid, record))
        elif h < -heap[0][0]:
            heapq.heapreplace(heap, (-h, sid, record))
    chosen = {}
    excluded_key = 0
    for _, sid, record in sorted(heap, reverse=True):
        if any(key in old_keys for key in keys(record) if key[1]):
            excluded_key += 1
            continue
        chosen[sid] = record
        if len(chosen) == 10000:
            break
    if len(chosen) != 10000:
        raise ValueError('Insufficient fresh source1 sample')
    for row in rows(ROOT / 'dataset/train/train_ground_truth.tsv'):
        sid = row['source1_entity_id']
        if sid in chosen:
            chosen[sid]['truth'] = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
    parent = {sid:sid for sid in chosen}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    seen = {}
    for sid, record in chosen.items():
        for key in keys(record):
            if not key[1]:
                continue
            if key in seen:
                parent[find(sid)] = find(seen[key])
            else:
                seen[key] = sid
    groups = defaultdict(list)
    for sid in chosen:
        groups[find(sid)].append(sid)
    counts = Counter()
    strata = defaultdict(Counter)
    for group, sids in sorted(groups.items(), key=lambda item:(-len(item[1]), item[0])):
        selected = min(('development','confirmation'), key=lambda fold:counts[fold]/(.6 if fold=='development' else .4))
        for sid in sids:
            r = chosen[sid]
            r['fold'] = selected
            r['group'] = group
            strata[selected][(r['country'],min(5,len(r['truth'])))] += 1
        counts[selected] += len(sids)
    records = [chosen[sid] for sid in sorted(chosen)]
    manifest = {'seed':'recovery-20260927', 'records':records,'groups':len(groups),
                'largest_group':max(map(len,groups.values())), 'excluded_existing_keys':excluded_key,
                'fold_counts':dict(counts),
                'country_cardinality':{fold:{str(k):v for k,v in values.items()} for fold,values in strata.items()}}
    SAMPLE.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    print('FRESH_SPLIT',json.dumps({k:v for k,v in manifest.items() if k!='records'}),flush=True)

    # NTFS hard links use the existing verified bytes without copying large matrices.
    index_root = OUT / 'train_index'
    index_root.mkdir(exist_ok=True)
    for source in (2,3):
        for file in (ROOT / f'artifacts/v2/index2_s{source}').iterdir():
            if not file.is_file():
                continue
            dest_dir = index_root / f'index2_s{source}'
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / file.name
            if not dest.exists():
                os.link(file, dest)
            if dest.stat().st_size != file.stat().st_size:
                raise ValueError(f'Index link mismatch {file}')
        file = ROOT / f'artifacts/v1/s{source}.sqlite'
        dest = index_root / f's{source}.sqlite'
        if not dest.exists():
            os.link(file,dest)
        if dest.stat().st_size != file.stat().st_size:
            raise ValueError(f'SQLite link mismatch {source}')
    print('INDEX_LINKS_READY',index_root,flush=True)


if __name__ == '__main__':
    main()
