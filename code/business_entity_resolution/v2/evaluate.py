"""Development-only selection and once-frozen confirmation evaluation."""
from common import *
import argparse,gc,pickle
from collections import Counter,defaultdict
from retrieval import FEATURE_GLOB
from pair_features import FEATURES
from experiment import eval_arrays,slices

def diagnostic(records,out,confirmation=False):
    meta=read(V2/'truth_metadata.json');tot=defaultdict(Counter);hit=defaultdict(Counter);counts=np.zeros(len(records),int);found=defaultdict(set)
    for p in out.glob(FEATURE_GLOB):
        z=np.load(p);counts+=np.bincount(z['q'],minlength=len(records))
        for i,e in zip(z['q'],z['eid']):
            if records[i]['fold']=='validation' and not confirmation:continue
            if e in meta[records[i]['entity_id']]:found[int(i)].add(str(e))
    result={};misses=[]
    for f in ['fit','tune']+(['validation'] if confirmation else []):
        ids=[i for i,r in enumerate(records) if r['fold']==f]
        for i in ids:
            for e,m in meta[records[i]['entity_id']].items():
                tot[f].update(m['slices'])
                if e in found[i]:hit[f].update(m['slices'])
                else:misses.append({'s1':records[i]['entity_id'],'fold':f,'target':e,**m})
        c=counts[ids];result[f]={'recall':{s:{'true':v,'found':hit[f][s],'recall':hit[f][s]/v} for s,v in tot[f].items()},
          'counts':{'mean':float(c.mean()),'p50':float(np.quantile(c,.5)),'p90':float(np.quantile(c,.9)),'p99':float(np.quantile(c,.99)),'max':int(c.max()),'pairs':int(c.sum()),'reduction_ratio':1-c.sum()/(len(ids)*10320219)},
          'oracle':metrics({records[i]['entity_id']:set(records[i]['truth']) for i in ids},{records[i]['entity_id']:found[i] for i in ids})}
    save(out/('candidate_confirmation.json' if confirmation else 'candidate_development.json'),result)
    save(out/('misses_confirmation.json' if confirmation else 'misses_development.json'),misses)
    print('candidate recall',{f:x['recall']['all'] for f,x in result.items()},flush=True)
    return result

def arrays(records,out,fold,sample=False):
    xs=[];ys=[];ws=[];qs=[];es=[];rng=np.random.default_rng(20260926);stat=Counter()
    for p in sorted(out.glob(FEATURE_GLOB)):
        z=np.load(p);X=z['X'];q=z['q'];eid=z['eid']
        for i in np.unique(q):
            if records[i]['fold']!=fold:continue
            ix=np.flatnonzero(q==i);x=X[ix];e=eid[ix];y=np.isin(e,records[i]['truth']);w=np.ones(len(x),np.float32)
            if sample:
                hard=(x[:,0]>0)|(x[:,1]>0)|(x[:,2]>.45)|(x[:,16]>.45)|((x[:,25]>0)&(x[:,2]>.2))
                rank=x[:,[j for j,s in enumerate(FEATURES) if s.endswith('inverse_rank')]].max(axis=1)
                top=np.zeros(len(x),bool);top[np.argsort(-rank,kind='stable')[:25]]=True
                keep=y|hard|top;tail=np.flatnonzero(~keep);take=min(15,len(tail))
                if take:
                    selected=rng.choice(tail,take,replace=False);keep[selected]=True;w[selected]=len(tail)/take
                stat.update(available=len(x),positive=int(y.sum()),hard_negative=int((hard&~y).sum()),selected=int(keep.sum()),tail=take)
            else:keep=np.ones(len(x),bool)
            xs.append(x[keep]);ys.append(y[keep]);ws.append(w[keep]);qs.append(np.full(keep.sum(),i,np.int32));es.append(e[keep])
    if sample:save(out/'training_sampling.json',dict(stat,seed=20260926,scheme='All retrieved positives, all hard collisions/strong-address/missing-address negatives, top 25 reciprocal route ranks, 15 weighted tail negatives per S1/source'))
    return np.concatenate(xs),np.concatenate(ys).astype(np.float32),np.concatenate(ws),np.concatenate(qs),np.concatenate(es)

def threshold(records,q,e,y,score):
    grid=[eval_arrays(records,'tune',q,e,y,score,h) for h in np.r_[.001,.005,np.linspace(.01,.99,50),.995,.999,1.]]
    best=max(grid,key=lambda x:(x['macro_f0_5'],x['threshold']))
    grid += [eval_arrays(records,'tune',q,e,y,score,h) for h in np.linspace(max(.0001,best['threshold']-.025),min(1,best['threshold']+.025),51)]
    best=max(grid,key=lambda x:(x['macro_f0_5'],x['threshold']))
    # Compare two thresholds on two deterministic halves of tune. Require gain in both.
    src3=np.char.startswith(e,'S3-');global_t=best['threshold'];separate=[]
    for s2 in np.linspace(max(.01,global_t-.1),min(.99,global_t+.1),9):
        for s3 in np.linspace(max(.01,global_t-.1),min(.99,global_t+.1),9):
            adjusted=score-np.where(src3,s3,s2)
            m=eval_arrays(records,'tune',q,e,y,adjusted,0);m.update(s2=float(s2),s3=float(s3));separate.append(m)
    sb=max(separate,key=lambda x:x['macro_f0_5']);support=[]
    for half in [0,1]:
        d=[dict(r,fold=r['fold'] if stable(r['group'])%2==half else 'ignored') for r in records]
        a=eval_arrays(d,'tune',q,e,y,score,global_t)['macro_f0_5'];b=eval_arrays(d,'tune',q,e,y,score-np.where(src3,sb['s3'],sb['s2']),0)['macro_f0_5'];support.append(b-a)
    use=sb['macro_f0_5']-best['macro_f0_5']>.002 and min(support)>.001
    calibration={}
    for src in ['S2','S3']:
        mask=np.char.startswith(e,src);bins=[]
        for lo,hi in zip([0,.1,.3,.5,.7,.9],[.1,.3,.5,.7,.9,1.00001]):
            ix=mask&(score>=lo)&(score<hi);bins.append({'lo':lo,'hi':hi,'n':int(ix.sum()),'predicted':float(score[ix].mean()) if ix.any() else None,'observed':float(y[ix].mean()) if ix.any() else None})
        calibration[src]=bins
    return {'global':best,'selected':sb if use else best,'thresholds':[sb['s2'],sb['s3']] if use else [global_t,global_t],
      'source_threshold_gain_halves':support,'source_thresholds_adopted':use,'global_grid':grid,'source_grid':separate,'calibration_bins':calibration,'calibration_fit':'none; threshold tuned directly'}

def train(records,out):
    import lightgbm as lgb,xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    diagnostic(records,out)
    start=time.perf_counter();X,y,w,_,_=arrays(records,out,'fit',True);T,ty,tw,tq,te=arrays(records,out,'tune')
    print('fit',X.shape,'tune',T.shape,'memory',memory(),flush=True);results={}
    for label in ['logistic','lightgbm','xgboost']:
        t=time.perf_counter()
        if label=='logistic':
            scaler=StandardScaler().fit(X);model=LogisticRegression(C=1,max_iter=250,solver='lbfgs');model.fit(scaler.transform(X),y,sample_weight=w)
            score=model.predict_proba(scaler.transform(T))[:,1].astype(np.float32)
            with open(out/'logistic.pkl','wb') as f:pickle.dump((scaler,model),f)
            params={'C':1,'max_iter':250};count=int(model.coef_.size+model.intercept_.size+2*X.shape[1])
        elif label=='lightgbm':
            params={'objective':'binary','metric':'binary_logloss','learning_rate':.04,'num_leaves':31,'min_data_in_leaf':30,'lambda_l2':5,'feature_fraction':.9,'verbosity':-1,'num_threads':4,'seed':20260926,'deterministic':True,'force_col_wise':True}
            model=lgb.train(params,lgb.Dataset(X,label=y,weight=w,feature_name=FEATURES),num_boost_round=600,
               valid_sets=[lgb.Dataset(T,label=ty,feature_name=FEATURES)],callbacks=[lgb.early_stopping(60,verbose=False)])
            score=model.predict(T,num_threads=4).astype(np.float32);model.save_model(str(out/'lightgbm.txt'))
            count=sum(4*(t['num_leaves']-1)+t['num_leaves'] for t in model.dump_model()['tree_info'])
        else:
            params={'objective':'binary:logistic','eval_metric':'logloss','learning_rate':.05,'max_depth':6,'min_child_weight':5,'reg_lambda':5,'subsample':.9,'colsample_bytree':.9,'tree_method':'hist','n_jobs':4,'random_state':20260926,'n_estimators':600,'early_stopping_rounds':60}
            model=xgb.XGBClassifier(**params);model.fit(X,y,sample_weight=w,eval_set=[(T,ty)],verbose=False)
            score=model.predict_proba(T)[:,1].astype(np.float32);model.save_model(out/'xgboost.ubj');count=sum(len(t.splitlines())*5 for t in model.get_booster().get_dump())
        if label=='logistic':
            with open(out/'logistic.pkl','rb') as f:rs,rm=pickle.load(f)
            replay=rm.predict_proba(rs.transform(T[:1000]))[:,1]
        elif label=='lightgbm':replay=lgb.Booster(model_file=str(out/'lightgbm.txt')).predict(T[:1000],num_threads=4)
        else:
            rm=xgb.XGBClassifier();rm.load_model(out/'xgboost.ubj');replay=rm.predict_proba(T[:1000])[:,1]
        np.testing.assert_allclose(replay,score[:1000],atol=1e-6)
        selection=threshold(records,tq,te,ty,score);save(out/f'{label}_thresholds.json',selection)
        np.savez_compressed(out/f'{label}_tune_scores.npz',q=tq,eid=te,score=score)
        results[label]={'tune':selection['selected'],'thresholds':selection['thresholds'],'parameters':count,'config':params,'seconds':time.perf_counter()-t,'memory':memory()}
        print(label,results[label],flush=True)
        del model;gc.collect()
    chosen=max(['lightgbm','xgboost'],key=lambda m:results[m]['tune']['macro_f0_5'])
    save(out/'frozen_selection.json',{'selected':chosen,'models':results,'features':FEATURES,'seconds':time.perf_counter()-start,'memory':memory(),'split_sha256':digest(V2/'split.json')})

def confirm(records,out):
    import lightgbm as lgb,xgboost as xgb
    frozen=read(out/'frozen_selection.json');meta=read(V2/'truth_metadata.json');cr=diagnostic(records,out,True)
    start=time.perf_counter();X,y,w,q,e=arrays(records,out,'validation');ids=[i for i,r in enumerate(records) if r['fold']=='validation']
    truth={records[i]['entity_id']:set(records[i]['truth']) for i in ids};candidates={s:set() for s in truth}
    for qi,eid in zip(q,e):candidates[records[qi]['entity_id']].add(str(eid))
    results={}
    for label,info in frozen['models'].items():
        if label=='logistic':
            with open(out/'logistic.pkl','rb') as f:scaler,model=pickle.load(f)
            scores=model.predict_proba(scaler.transform(X))[:,1]
        elif label=='lightgbm':model=lgb.Booster(model_file=str(out/'lightgbm.txt'));scores=model.predict(X,num_threads=4)
        else:model=xgb.XGBClassifier();model.load_model(out/'xgboost.ubj');scores=model.predict_proba(X)[:,1]
        scores=scores.astype(np.float32);thresholds=np.where(np.char.startswith(e,'S3-'),info['thresholds'][1],info['thresholds'][0]);pred={s:set() for s in truth}
        errors=[];slicecounts=defaultdict(Counter);entity_slices={}
        for qi in ids:
            for md in meta[records[qi]['entity_id']].values():
                for s in md['slices']:slicecounts[s]['truth']+=1
        conns={s:connection(s) for s in [2,3]};raw={}
        selected=np.flatnonzero(scores>=thresholds)
        for src in [2,3]:
            targetids=sorted(set(str(e[j]) for j in selected if e[j].startswith(f'S{src}-')))
            for offset in range(0,len(targetids),500):
                chunk=targetids[offset:offset+500]
                for row in conns[src].execute('SELECT eid,business_name,business_address,country FROM r WHERE eid IN ('+','.join('?'*len(chunk))+')',chunk):raw[row[0]]=dict(zip(['entity_id','business_name','business_address','country'],row))
            conns[src].close()
        for j in selected:
            qi=int(q[j]);eid=str(e[j]);sid=records[qi]['entity_id'];pred[sid].add(eid);tp=eid in truth[sid]
            for s in slices(view(records[qi]),view(raw[eid]),np.expm1(X[j,13])+1e-5):slicecounts[s]['predicted']+=1;slicecounts[s]['tp']+=int(tp)
            if not tp:
                cats=[name for condition,name in [(X[j,16]>.5,'strong_address'),(X[j,0]>0,'exact_name'),(X[j,1]>0,'suffix_name'),(X[j,13]>=math.log1p(5),'common_name'),(X[j,25]>0,'missing_address'),(X[j,21]>0,'numeric_conflict'),(X[j,2]<.2,'low_name_similarity')] if condition]
                errors.append({'kind':'false_positive','s1':sid,'target':eid,'score':float(scores[j]),'categories':cats or ['other'],'query':records[qi],'candidate':raw[eid]})
        for sid,t in truth.items():
            assert pred[sid]<=candidates[sid]
            for eid in t-pred[sid]:errors.append({'kind':'false_negative','s1':sid,'target':eid,'categories':['classifier_miss' if eid in candidates[sid] else 'retrieval_miss']+meta[sid][eid]['slices'][3:],'candidate':meta[sid][eid]['candidate']})
        for country in sorted({records[i]['country'] for i in ids}):
            keys=[records[i]['entity_id'] for i in ids if records[i]['country']==country];entity_slices[country]=metrics({s:truth[s] for s in keys},{s:pred[s] for s in keys})
        for cardinality in range(6):
            for mode,mapping in [('true',truth),('predicted',pred)]:
                keys=[s for s in truth if min(5,len(mapping[s]))==cardinality]
                if keys:entity_slices[f'{mode}_cardinality_{cardinality}']=metrics({s:truth[s] for s in keys},{s:pred[s] for s in keys})
        for source in ['S2','S3']:entity_slices[source]=metrics({s:{e for e in t if e.startswith(source)} for s,t in truth.items()},{s:{e for e in p if e.startswith(source)} for s,p in pred.items()})
        lc={s:dict(c,precision=c['tp']/c['predicted'] if c['predicted'] else None,recall=c['tp']/c['truth'] if c['truth'] else None) for s,c in slicecounts.items()}
        result={'overall':metrics(truth,pred),'entity_slices':entity_slices,'link_slices':lc,'errors':{kind:dict(Counter(cat for r in errors if r['kind']==kind for cat in r['categories'])) for kind in ['false_positive','false_negative']}}
        save(out/f'{label}_errors.json',errors);write_sets(out/f'confirmation_{label}.tsv',pred,'matched_entity_ids')
        np.savez_compressed(out/f'{label}_confirmation_scores.npz',q=q,eid=e,score=scores)
        results[label]=result;print('CONFIRM',label,result['overall'],flush=True)
    write_sets(out/'confirmation_candidates.tsv',candidates,'candidate_entity_ids')
    save(out/'confirmation_results.json',{'selected':frozen['selected'],'models':results,'seconds':time.perf_counter()-start,'memory':memory()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnose','train','confirm']);a=p.parse_args()
    records=read(V2/'split.json')['records'];out=V2/'run'
    if a.stage=='diagnose':diagnostic(records,out)
    elif a.stage=='train':train(records,out)
    else:confirm(records,out)
