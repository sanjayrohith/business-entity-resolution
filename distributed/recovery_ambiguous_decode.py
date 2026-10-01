"""Train-only stress test of target ownership using a held-out repeated-name sample."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/v2/packages'))
import lightgbm as lgb
import numpy as np
import recovery_meta_experiment as meta

OUT=ROOT/'distributed/recovery'
meta.FRESH=OUT/'ambiguous_r10'
meta.RECORDS=json.loads((OUT/'ambiguous_split.json').read_text(encoding='utf-8'))['records']

def metrics(pred,fold):
    n=0;fs=ps=rs=0.;tp=links=0
    for qi,r in enumerate(meta.RECORDS):
        if r['fold']!=fold:continue
        t=set(r['truth']);p=pred.get(qi,set());hit=len(t&p)
        n+=1;tp+=hit;links+=len(p)
        fs+=1 if not t and not p else (1.25*hit/(.25*len(t)+len(p)) if p else 0.)
        ps+=1 if not t and not p else (hit/len(p) if p else 0.)
        rs+=1 if not t and not p else (hit/len(t) if t else 0.)
    return {'source1':n,'macro_f0_5':fs/n,'macro_precision':ps/n,
            'macro_recall':rs/n,'true_positive_links':tp,'predicted_links':links}

def paired_interval(first,second,fold):
    diff=[]
    for qi,r in enumerate(meta.RECORDS):
        if r['fold']!=fold:continue
        truth=set(r['truth'])
        def fscore(pred):
            p=pred.get(qi,set());hit=len(truth&p)
            return 1. if not truth and not p else (1.25*hit/(.25*len(truth)+len(p)) if p else 0.)
        diff.append(fscore(second)-fscore(first))
    values=np.asarray(diff)
    rng=np.random.default_rng(20260927)
    draws=np.asarray([values[rng.integers(0,len(values),len(values))].mean() for _ in range(1000)])
    return {'mean':float(values.mean()),'bootstrap_95_percent':np.quantile(draws,[.025,.975]).tolist()}

def main():
    x,q,eid,src,base=meta.load_pairs()
    model=lgb.Booster(model_file=str(OUT/'meta_r10_lightgbm.txt'))
    score=model.predict(np.concatenate([x,meta.context(q,src,base)],axis=1),num_threads=4)
    report={}
    for fold in ('development','confirmation'):
        mask=np.array([r['fold']==fold for r in meta.RECORDS])[q]
        sel=np.flatnonzero(mask&(score>=.7))
        target=defaultdict(list)
        baseline=defaultdict(set);current=defaultdict(set)
        for i in np.flatnonzero(mask&(base>=.642)):
            baseline[int(q[i])].add(eid[i])
        for i in sel:
            current[int(q[i])].add(eid[i]);target[eid[i]].append((int(q[i]),float(score[i])))
        conflicted={e:v for e,v in target.items() if len(v)>1}
        result={'baseline':metrics(baseline,fold),'meta':metrics(current,fold),
                'conflicted_targets':len(conflicted),
                'excess_ownership_links':sum(len(v)-1 for v in conflicted.values())}
        for margin in (0.,.05,.1,.2):
            pred=defaultdict(set)
            for e,values in target.items():
                values.sort(key=lambda z:(-z[1],z[0]))
                if len(values)==1 or values[0][1]-values[1][1]>=margin:
                    pred[values[0][0]].add(e)
            result[f'exclusive_margin_{margin}']=metrics(pred,fold)
            if margin==.05:result['exclusive_0.05_minus_meta_paired']=paired_interval(current,pred,fold)
        report[fold]=result
    (OUT/'ambiguous_decode.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
