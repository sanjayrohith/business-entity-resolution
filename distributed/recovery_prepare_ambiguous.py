"""Train-only stress sample of repeated S1 names for ownership decoding."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'distributed/recovery/ambiguous_split.json'


def rows(path):
    with open(path,encoding='utf-8') as f:
        keys=f.readline().rstrip('\n').split('\t')
        for line in f:
            yield dict(zip(keys,line.rstrip('\r\n').split('\t')))


def main():
    if OUT.exists():print('REUSE',OUT);return
    old=set()
    for path in (ROOT/'artifacts/v1/split_6000.json',ROOT/'artifacts/v2/split.json',ROOT/'distributed/recovery/fresh_split.json'):
        old.update(r['entity_id'] for r in json.loads(path.read_text(encoding='utf-8'))['records'])
    path=ROOT/'dataset/train/train_source1.tsv'
    counts=Counter(r['business_name'].casefold().strip() for r in rows(path) if r['entity_id'] not in old)
    eligible=[(key,count) for key,count in counts.items() if count>=3 and key]
    eligible.sort(key=lambda item:int.from_bytes(hashlib.blake2b(('ambiguous|'+item[0]).encode(),digest_size=8).digest(),'big'))
    chosen=set();total=0
    for key,count in eligible:
        chosen.add(key);total+=count
        if total>=10000:break
    selected={}
    for r in rows(path):
        if r['entity_id'] not in old and r['business_name'].casefold().strip() in chosen:
            selected[r['entity_id']]=r
    for r in rows(ROOT/'dataset/train/train_ground_truth.tsv'):
        sid=r['source1_entity_id']
        if sid in selected:
            selected[sid]['truth']=r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []
    groups=defaultdict(list)
    for sid,r in selected.items():groups[r['business_name'].casefold().strip()].append(sid)
    fold_counts=Counter()
    for key,sids in sorted(groups.items()):
        fold=min(('development','confirmation'),key=lambda f:fold_counts[f]/(.6 if f=='development' else .4))
        for sid in sids:
            selected[sid]['fold']=fold
            selected[sid]['group']=key
        fold_counts[fold]+=len(sids)
    records=[selected[sid] for sid in sorted(selected)]
    manifest={'selection':'repeated raw S1 name groups with >=3 records; lowest seeded group hashes',
              'records':records,'groups':len(groups),'largest_group':max(map(len,groups.values())),
              'fold_counts':dict(fold_counts)}
    OUT.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
    print('AMBIGUOUS_SPLIT',json.dumps({k:v for k,v in manifest.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
