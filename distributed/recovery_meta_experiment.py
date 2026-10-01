"""Train a second-stage pair model on fresh out-of-sample R10 scores."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'artifacts/v2/packages'))
import lightgbm as lgb
import numpy as np

OUT = ROOT / 'distributed/recovery'
FRESH = OUT / 'fresh_r10'
RECORDS = json.loads((OUT / 'fresh_split.json').read_text(encoding='utf-8'))['records']
BASE_COLS = [0,1,2,3,4,5,6,7,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,
             60,61,62,63,64,65,66,67,68,69,70,77,78,79]
BASE_NAMES = json.loads((ROOT / 'artifacts/v2/run/frozen_selection.json').read_text(encoding='utf-8'))['features']
CONTEXT_NAMES = ['base_score','s1_rank_all','s1_rank_source','s1_top_score','s1_second_score',
                 's1_score_gap_top','s1_score_gap_second','s1_count_score_ge_05',
                 's1_count_score_ge_0642','s1_count_score_ge_08','s1_count_score_ge_095',
                 'other_source_top_score','same_source_top_score','source3_indicator']


def load_pairs():
    arrays=[]; queries=[]; ids=[]; sources=[]; scores=[]
    for src in (2,3):
        score_path=FRESH/f's{src}_scores.tsv'
        source_scores=[]
        with open(score_path,encoding='utf-8') as f:
            next(f)
            for line in f:
                source_scores.append(float(line.split('\t',3)[2]))
        base_idx=0
        for q0 in range(0,len(RECORDS),25):
            with np.load(FRESH/f's{src}_{q0:05d}.npz') as z:
                x,q,eid=z['X'],z['q'],z['eid']
            arrays.append(x[:,BASE_COLS])
            queries.append(q)
            ids.extend(map(str,eid))
            sources.append(np.full(len(q),src,dtype=np.uint8))
            scores.extend(source_scores[base_idx:base_idx+len(q)])
            base_idx+=len(q)
        assert base_idx==len(source_scores)
    return np.concatenate(arrays),np.concatenate(queries),ids,np.concatenate(sources),np.asarray(scores,np.float32)


def context(q,source,score):
    n=len(score)
    ctx=np.zeros((n,len(CONTEXT_NAMES)),dtype=np.float32)
    ctx[:,0]=score
    ctx[:,-1]=(source==3)
    order=np.argsort(q,kind='stable')
    q_sorted=q[order]
    bounds=np.r_[0,np.flatnonzero(np.diff(q_sorted))+1,n]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        ix=order[lo:hi]
        v=score[ix]
        ranks=np.argsort(np.argsort(-v,kind='stable'),kind='stable')+1
        ctx[ix,1]=ranks
        sorted_scores=np.sort(v)[::-1]
        ctx[ix,3]=sorted_scores[0]
        ctx[ix,4]=sorted_scores[1] if len(sorted_scores)>1 else 0
        ctx[ix,5]=sorted_scores[0]-v
        ctx[ix,6]=v-(sorted_scores[1] if len(sorted_scores)>1 else 0)
        for col,t in ((7,.5),(8,.642),(9,.8),(10,.95)):
            ctx[ix,col]=np.count_nonzero(v>=t)
        for src in (2,3):
            mask=source[ix]==src
            if not mask.any():continue
            j=ix[mask]
            vv=score[j]
            ctx[j,2]=np.argsort(np.argsort(-vv,kind='stable'),kind='stable')+1
            ctx[j,12]=vv.max()
            other=score[ix[~mask]]
            ctx[j,11]=other.max() if len(other) else 0
    return ctx


def entity_metrics(q,ids,truth,scores,selected,mask):
    pred={}
    for qi,e,keep in zip(q,ids,selected):
        if keep:pred.setdefault(int(qi),set()).add(e)
    f=p=r=0.; n=tp=pl=tl=0
    for qi,record in enumerate(RECORDS):
        if not mask[qi]:continue
        t=truth[qi]; x=pred.get(qi,set()); z=len(t&x)
        n+=1; tp+=z; pl+=len(x); tl+=len(t)
        f+=1 if not t and not x else (1.25*z/(.25*len(t)+len(x)) if x else 0)
        p+=1 if not t and not x else (z/len(x) if x else 0)
        r+=1 if not t and not x else (z/len(t) if t else 0)
    return {'source1':n,'macro_f0_5':f/n,'macro_precision':p/n,'macro_recall':r/n,
            'true_positive_links':tp,'predicted_links':pl,'true_links':tl}


def main():
    started=time.monotonic()
    x,q,ids,src,base=load_pairs()
    print('LOADED',len(q),round(time.monotonic()-started,1),flush=True)
    ctx=context(q,src,base)
    X=np.concatenate([x,ctx],axis=1)
    names=[BASE_NAMES[i] for i in BASE_COLS]+CONTEXT_NAMES
    truth=[set(record['truth']) for record in RECORDS]
    label=np.fromiter((eid in truth[int(qi)] for qi,eid in zip(q,ids)),dtype=np.uint8,count=len(q))
    is_development=np.array([record['fold']=='development' for record in RECORDS])
    group_hash=np.array([int.from_bytes(hashlib.blake2b(('meta|'+record['group']).encode(),digest_size=8).digest(),'big')%3
                         for record in RECORDS])
    fit_s1=is_development&(group_hash!=0)
    tune_s1=is_development&(group_hash==0)
    confirm_s1=~is_development
    fit=fit_s1[q]; tune=tune_s1[q]
    model=lgb.LGBMClassifier(n_estimators=350,num_leaves=15,learning_rate=.05,
                             min_child_samples=90,reg_lambda=10,colsample_bytree=.9,
                             verbosity=-1,n_jobs=6,random_state=20260927)
    model.fit(X[fit],label[fit],eval_set=[(X[tune],label[tune])],eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(35,verbose=False)])
    score=model.predict_proba(X)[:,1]
    tune_results={}
    for threshold in np.round(np.arange(.2,.801,.025),3):
        tune_results[str(threshold)]=entity_metrics(q,ids,truth,score,score>=threshold,tune_s1)
    best=max(tune_results,key=lambda k:tune_results[k]['macro_f0_5'])
    result={'rows':len(q),'feature_names':names,'fit_source1':int(fit_s1.sum()),
            'tune_source1':int(tune_s1.sum()),'confirmation_source1':int(confirm_s1.sum()),
            'best_iteration':model.best_iteration_,'selected_threshold':float(best),
            'tune_baseline':entity_metrics(q,ids,truth,base,base>=.642,tune_s1),
            'tune_meta':tune_results[best],
            'confirmation_baseline':entity_metrics(q,ids,truth,base,base>=.642,confirm_s1),
            'confirmation_meta':entity_metrics(q,ids,truth,score,score>=float(best),confirm_s1),
            'tune_threshold_curve':tune_results,
            'seconds':time.monotonic()-started}
    model.booster_.save_model(str(OUT/'meta_r10_lightgbm.txt'))
    (OUT/'meta_r10_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('feature_names','tune_threshold_curve')}),flush=True)


if __name__=='__main__':main()
