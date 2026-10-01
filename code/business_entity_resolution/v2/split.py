"""Fresh grouped V2 pool; excludes every V1 entity and its sampled grouping keys."""
from common import *
import heapq
from collections import defaultdict,Counter
from experiment import slices

def run(n=10000):
    path=V2/'split.json'
    if path.exists():return
    start=time.perf_counter();old=read(V1/'split_6000.json')['records']
    oldids={r['entity_id'] for r in old};oldkeys=set()
    def keys(r):
        v=view(r);return [('n',v['s']),('a',' '.join(sorted(v['at'])))]
    for r in old:oldkeys.update(keys(r))
    heap=[]
    for r in rows(ROOT/'dataset/train/train_source1.tsv'):
        if r['entity_id'] in oldids:continue
        h=int.from_bytes(hashlib.blake2b(('20260926|'+r['entity_id']).encode(),digest_size=8).digest(),'big')
        if len(heap)<n*2:heapq.heappush(heap,(-h,r['entity_id'],r))
        elif h < -heap[0][0]:heapq.heapreplace(heap,(-h,r['entity_id'],r))
    chosen={};excluded=0
    for _,eid,r in sorted(heap,reverse=True):
        if any(k in oldkeys for k in keys(r)):excluded+=1;continue
        chosen[eid]=r
        if len(chosen)==n:break
    assert len(chosen)==n
    for r in rows(ROOT/'dataset/train/train_ground_truth.tsv'):
        if r['source1_entity_id'] in chosen:chosen[r['source1_entity_id']]['truth']=sorted(parse_ids(r['matched_entity_ids']))
    target={};freq={}
    for src in [2,3]:
        con=connection(src);ids=sorted({e for r in chosen.values() for e in r['truth'] if e.startswith(f'S{src}-')})
        for i in range(0,len(ids),500):
            chunk=ids[i:i+500]
            for row in con.execute('SELECT eid,business_name,business_address,country FROM r WHERE eid IN ('+','.join('?'*len(chunk))+')',chunk):target[row[0]]=dict(zip(['entity_id','business_name','business_address','country'],row))
        # Counts are unlabeled corpus statistics.
        for eid,r in chosen.items():
            nf=fold(norm(r['business_name']))
            freq[eid,src]=con.execute('SELECT count(*) FROM r WHERE n=?',(nf,)).fetchone()[0]
        con.close()
    parent={e:e for e in chosen}
    def find(e):
        while e!=parent[e]:parent[e]=parent[parent[e]];e=parent[e]
        return e
    seen={};metadata={}
    for e,r in chosen.items():
        for k in keys(r)+[('target',t) for t in r['truth']]:
            if not k[1]:continue
            if k in seen:parent[find(e)]=find(seen[k])
            else:seen[k]=e
        q=view(r);md={t:{'slices':slices(q,view(target[t]),freq[e,int(t[1])]),'candidate':target[t]} for t in r['truth']}
        metadata[e]=md
        all_slices={s for m in md.values() for s in m['slices']}
        r['difficulty']={s:s in all_slices for s in ['cross_script','missing_address','low_name_similarity','common_name']}
    groups=defaultdict(list)
    for e in chosen:groups[find(e)].append(e)
    counts={f:Counter() for f in ['fit','tune','validation']};ratios={'fit':.5,'tune':.25,'validation':.25}
    for g,es in sorted(groups.items(),key=lambda x:(-len(x[1]),stable('v2'+x[0]))):
        strata=Counter()
        for e in es:
            r=chosen[e];strata[('country_card',r['country'],min(5,len(r['truth'])))]+=1
            for s,b in r['difficulty'].items():strata[(s,r['country'],b)]+=1
        f=min(ratios,key=lambda f:sum((counts[f][s]+v/2)*v/ratios[f] for s,v in strata.items()))
        counts[f].update(strata)
        for e in es:chosen[e].update(fold=f,group=g)
    records=[chosen[e] for e in sorted(chosen)]
    save(path,{'seed':20260926,'records':records,'groups':len(groups),'largest_group':max(map(len,groups.values())),
      'v1_keys_excluded':excluded,'strata':{f:{str(k):v for k,v in c.items()} for f,c in counts.items()},'seconds':time.perf_counter()-start})
    save(V2/'truth_metadata.json',metadata)
    with open(V2/'split_ids.tsv','w',encoding='utf8') as f:
        f.write('source1_entity_id\tfold\tgroup\n')
        for r in records:f.write(f"{r['entity_id']}\t{r['fold']}\t{r['group']}\n")
    print('split',Counter(r['fold'] for r in records),'seconds',time.perf_counter()-start,flush=True)
if __name__=='__main__':run()
