"""Fast ownership-aware decoder search using retained validation scores only."""
import json,sys,time
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/v2/packages'))
sys.path.insert(0,str(ROOT/'distributed'))
import lightgbm as lgb
import numpy as np
import recovery_meta_experiment as meta

OUT=ROOT/'distributed/recovery'
MODEL=lgb.Booster(model_file=str(OUT/'meta_r10_lightgbm.txt'))
THRESHOLDS=np.asarray([.40,.45,.48,.50,.52,.54,.56,.58,.60,.62,.64,.66,.68,.70,.72,.74,.76])
MARGINS=np.asarray([0,.01,.02,.03,.05,.08,.10])

def load_slice(name,split_path,data_path):
    meta.RECORDS=json.loads(split_path.read_text(encoding='utf-8'))['records']
    meta.FRESH=data_path
    x,q,eid,src,base=meta.load_pairs()
    score=MODEL.predict(np.concatenate([x,meta.context(q,src,base)],axis=1),num_threads=6)
    records=meta.RECORDS
    truth=[set(r['truth']) for r in records]
    label=np.fromiter((target in truth[int(qi)] for qi,target in zip(q,eid)),dtype=np.uint8,count=len(q))
    _,target_code=np.unique(np.asarray(eid),return_inverse=True)
    # ranks among all candidates for one S1 and among candidates from the same source
    rank=np.empty(len(q),np.int32);rank_src=np.empty(len(q),np.int32)
    order=np.lexsort((-score,q)); qs=q[order]; bounds=np.r_[0,np.flatnonzero(np.diff(qs))+1,len(q)]
    for lo,hi in zip(bounds[:-1],bounds[1:]):rank[order[lo:hi]]=np.arange(1,hi-lo+1)
    for source in (2,3):
        ix=np.flatnonzero(src==source);o=ix[np.lexsort((-score[ix],q[ix]))];z=q[o];b=np.r_[0,np.flatnonzero(np.diff(z))+1,len(o)]
        for lo,hi in zip(b[:-1],b[1:]):rank_src[o[lo:hi]]=np.arange(1,hi-lo+1)
    folds={fold:np.asarray([r['fold']==fold for r in records]) for fold in ('development','confirmation')}
    return {'name':name,'records':records,'truth':truth,'q':q,'eid':np.asarray(eid),'src':src,'score':score,
            'label':label,'target':target_code,'rank':rank,'rank_src':rank_src,'folds':folds}

def decode(data,threshold,margin,mutual=0,t2=None,t3=None):
    score=data['score'];src=data['src']
    selected=score>=(np.where(src==2,t2,t3) if t2 is not None else threshold)
    if mutual:selected &= data['rank']<=mutual
    ix=np.flatnonzero(selected)
    if not len(ix):return np.zeros(len(score),bool)
    # target ID, descending score, deterministic q order
    order=ix[np.lexsort((data['q'][ix],-score[ix],data['target'][ix]))]
    codes=data['target'][order]; bounds=np.r_[0,np.flatnonzero(np.diff(codes))+1,len(order)]
    keep=np.zeros(len(score),bool)
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        if hi-lo==1 or score[order[lo]]-score[order[lo+1]]>=margin:keep[order[lo]]=True
    return keep

def measure(data,keep,fold):
    mask=data['folds'][fold];n=len(mask);chosen=np.flatnonzero(keep)
    pred=np.bincount(data['q'][chosen],minlength=n);tp=np.bincount(data['q'][chosen],weights=data['label'][chosen],minlength=n)
    true=np.asarray([len(t) for t in data['truth']]);empty=(true==0)&(pred==0)
    p=np.divide(tp,pred,out=np.zeros(n),where=pred>0);r=np.divide(tp,true,out=np.zeros(n),where=true>0)
    f=np.divide(1.25*tp,.25*true+pred,out=np.zeros(n),where=(.25*true+pred)>0);p[empty]=r[empty]=f[empty]=1
    return {'source1':int(mask.sum()),'macro_f0_5':float(f[mask].mean()),'macro_precision':float(p[mask].mean()),
            'macro_recall':float(r[mask].mean()),'singleton_accuracy':float(f[mask&(true==0)].mean()),
            'predicted_links':int(pred[mask].sum()),'predicted_links_per_entity':float(pred[mask].mean()),
            'true_positive_links':int(tp[mask].sum()),'false_positive_links':int((pred-tp)[mask].sum()),
            'false_negative_links':int((true-tp)[mask].sum())}

def evaluate(data,t,m,mutual=0,t2=None,t3=None):
    keep=decode(data,t,m,mutual,t2,t3)
    return {fold:measure(data,keep,fold) for fold in ('development','confirmation')}

def main():
    started=time.monotonic()
    slices=[load_slice('fresh',OUT/'fresh_split.json',OUT/'fresh_r10'),
            load_slice('ambiguity',OUT/'ambiguous_split.json',OUT/'ambiguous_r10')]
    grid=[]
    for t in THRESHOLDS:
        for m in MARGINS:
            row={'threshold':float(t),'margin':float(m),'mutual':0}
            for data in slices:row[data['name']]=evaluate(data,t,m)
            grid.append(row)
    baseline=next(r for r in grid if r['threshold']==.7 and r['margin']==.05)
    # Select on both development folds: mean delta, with the weaker delta breaking ties.
    for row in grid:
        ds=[row[s['name']]['development']['macro_f0_5']-baseline[s['name']]['development']['macro_f0_5'] for s in slices]
        row['development_delta_mean']=float(np.mean(ds));row['development_delta_min']=float(np.min(ds))
    ranked=sorted(grid,key=lambda r:(r['development_delta_min'],r['development_delta_mean']),reverse=True)
    best=ranked[0]
    # Cheap mutual rank checks around selected threshold/margin.
    mutual=[]
    for rank in (1,3):
        row={'threshold':best['threshold'],'margin':best['margin'],'mutual':rank}
        for data in slices:row[data['name']]=evaluate(data,best['threshold'],best['margin'],rank)
        mutual.append(row)
    # One small source-specific grid near best.
    source_grid=[];around=sorted(set(float(np.clip(best['threshold']+d,.4,.8)) for d in (-.04,-.02,0,.02,.04)))
    for t2 in around:
        for t3 in around:
            row={'s2_threshold':t2,'s3_threshold':t3,'margin':best['margin']}
            for data in slices:row[data['name']]=evaluate(data,0,best['margin'],0,t2,t3)
            ds=[row[s['name']]['development']['macro_f0_5']-baseline[s['name']]['development']['macro_f0_5'] for s in slices]
            row['development_delta_mean']=float(np.mean(ds));row['development_delta_min']=float(np.min(ds));source_grid.append(row)
    source_best=max(source_grid,key=lambda r:(r['development_delta_min'],r['development_delta_mean']))
    result={'elapsed_seconds':time.monotonic()-started,'baseline_recovery03':baseline,'best_global_development':best,
            'top_global':ranked[:15],'mutual_tests':mutual,'best_source_thresholds_development':source_best,
            'selection_rule':'maximize minimum development improvement across fresh and ambiguity slices, then mean',
            'grid_size':len(grid)}
    (OUT/'recovery04_grid.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
