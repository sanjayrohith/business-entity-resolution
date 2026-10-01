from common import *
from collections import Counter
from retrieval import FEATURE_GLOB
old=read(V1/'split_6000.json')['records'];chosen=[(i,r) for i,r in enumerate(old) if r['fold']=='fit'][:500];indices={i for i,r in chosen}
meta=read(V1/'run_6000/truth_diagnostics_development.json');hit=Counter();total=Counter();counts=Counter()
for i,r in chosen:
    for md in meta[str(i)].values():total.update(md['slices'])
for p in (V1/'run_6000').glob(FEATURE_GLOB):
    z=np.load(p);mask=np.isin(z['q'],list(indices))
    for i,e in zip(z['q'][mask],z['eid'][mask]):
        counts[int(i)]+=1
        if str(e) in meta[str(i)]:hit.update(meta[str(i)][str(e)]['slices'])
save(V2/'v1_paired_fit500.json',{'recall':{s:{'total':v,'hit':hit[s],'recall':hit[s]/v} for s,v in total.items()},
    'counts':{'mean':sum(counts.values())/500,'p99':float(np.quantile(list(counts.values()),.99))},'scope':'First 500 V1 fitting entities; neither holdout used'})
print('V1 paired recall',hit['all']/total['all'],flush=True)
