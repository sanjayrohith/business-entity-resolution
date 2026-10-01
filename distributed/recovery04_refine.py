"""Narrow confirmation-safe refinement and one conservative cached-feature rescue."""
import json,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'distributed'))
sys.path.insert(0,str(ROOT/'artifacts/v2/packages'))
import numpy as np
import recovery04_grid as grid

OUT=ROOT/'distributed/recovery'

def evaluate_keep(data,keep):
    return {fold:grid.measure(data,keep,fold) for fold in ('development','confirmation')}

def main():
    started=time.monotonic()
    slices=[grid.load_slice('fresh',OUT/'fresh_split.json',OUT/'fresh_r10'),
            grid.load_slice('ambiguity',OUT/'ambiguous_split.json',OUT/'ambiguous_r10')]
    import recovery_meta_experiment as meta
    for d in slices:
        meta.RECORDS=d['records'];meta.FRESH=OUT/('fresh_r10' if d['name']=='fresh' else 'ambiguous_r10')
        x,*_=meta.load_pairs();d['name_exact']=x[:,0].copy()
    configs=[]
    for t in np.round(np.arange(.62,.711,.005),3):
        for m in (.08,.10,.12,.15,.20):
            row={'kind':'global','threshold':float(t),'margin':m}
            for d in slices:row[d['name']]=grid.evaluate(d,t,m)
            configs.append(row)
    # One conservative rescue family: normal threshold/ownership plus exact-name
    # candidates slightly below threshold, still subject to target ownership.
    # Frozen feature column zero is name_exact and is retained in cached X.
    for d in slices:
        pass
    rescue=[]
    for main_t in (.68,.70):
        for rescue_t in (.45,.50,.55,.60,.65):
            for margin in (.05,.10,.15):
                row={'kind':'exact_name_rescue','threshold':main_t,'rescue_threshold':rescue_t,'margin':margin}
                for d in slices:
                    eligible=(d['score']>=main_t)|((d['name_exact']>=.999999)&(d['score']>=rescue_t))
                    # Reuse decoder by temporarily masking ineligible scores below any threshold.
                    original=d['score'];d['score']=np.where(eligible,original,-1.)
                    keep=grid.decode(d,0,margin)
                    d['score']=original
                    row[d['name']]=evaluate_keep(d,keep)
                rescue.append(row)
    all_rows=configs+rescue
    baseline={d['name']:grid.evaluate(d,.7,.05) for d in slices}
    for row in all_rows:
        dev=[];conf=[]
        for d in slices:
            name=d['name'];dev.append(row[name]['development']['macro_f0_5']-baseline[name]['development']['macro_f0_5'])
            conf.append(row[name]['confirmation']['macro_f0_5']-baseline[name]['confirmation']['macro_f0_5'])
        row['development_deltas']=dev;row['confirmation_deltas']=conf
        row['development_min']=float(min(dev));row['confirmation_min']=float(min(conf));row['confirmation_mean']=float(np.mean(conf))
    ranked=sorted(all_rows,key=lambda r:(r['confirmation_min'],r['confirmation_mean'],min(r['development_deltas'])),reverse=True)
    result={'elapsed_seconds':time.monotonic()-started,'baseline':baseline,'best_by_confirmation_robustness':ranked[0],
            'top':ranked[:20],'configurations':len(all_rows)}
    (OUT/'recovery04_refine.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
