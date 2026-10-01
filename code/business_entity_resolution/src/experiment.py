"""Candidate diagnostics, fit-only training, tuning-only selection and untouched validation."""
import argparse
import gzip
import json
import time
from collections import Counter,defaultdict
import numpy as np
from core import *
from prepare import ROOT,CACHE,save,memory
from retrieval import FEATURES,ROUTES,Retriever,FEATURE_GLOB
from models import Logistic,HistogramBoost,parameters

def load(n):
    data=json.loads((CACHE/f'split_{n}.json').read_text(encoding='utf-8'))['records']
    return data,CACHE/f'run_{n}'

def slices(q,c,nf=0):
    ss=['all',q['raw']['country'],c['raw']['entity_id'][:2]]
    if c['missing']:ss.append('missing_address')
    if jaccard(q['ng'],c['ng'])<.2:ss.append('low_name_similarity')
    if c['script']-{'LATIN'}:
        ss+=['non_latin_name']+['script_'+s for s in sorted(c['script']-{'LATIN'})]
        if q['script']!=c['script']:
            ss.append('cross_script')
            if q['raw']['country']=='India':ss.append('india_cross_script')
    if nf>=5:ss.append('common_name')
    if q['n']==c['n']:ss.append('exact_name')
    return ss

def truth_metadata(data,out,include_validation=False):
    p=out/('truth_diagnostics_final.json' if include_validation else 'truth_diagnostics_development.json')
    if p.exists():return json.loads(p.read_text(encoding='utf-8'))
    ret={};rts={s:Retriever(s) for s in [2,3]}
    for i,r in enumerate(data):
        if r['fold']=='validation' and not include_validation:continue
        q=view(r);ret[str(i)]={}
        for e in r['truth']:
            rt=rts[int(e[1])];_,c=rt.lookup(e)
            nf=rt.con.execute('SELECT count(*) FROM r WHERE n=?',(q['nf'],)).fetchone()[0]
            ret[str(i)][e]={'slices':slices(q,c,nf),'candidate':c['raw'],'name_similarity':jaccard(q['ng'],c['ng']),
                           'address_similarity':jaccard(q['ag'],c['ag'])}
    for rt in rts.values():rt.con.close()
    Retriever.get.cache_clear();save(p,ret);return ret

def candidate_diagnostics(n,include_validation=False):
    started=time.perf_counter();data,out=load(n);meta=truth_metadata(data,out,include_validation)
    folds=['fit','tune']+(['validation'] if include_validation else [])
    totals={f:Counter() for f in folds}; hits={(f,k):Counter() for f in folds for k in [10,20,50,100]}
    counts={k:np.zeros(n,dtype=np.int64) for k in [10,20,50,100]}; hitsets=defaultdict(set)
    source_counts={s:np.zeros(n,dtype=np.int64) for s in [2,3]}
    for i,r in enumerate(data):
        if r['fold'] in folds:
            for m in meta[str(i)].values():totals[r['fold']].update(m['slices'])
    rank_cols=[FEATURES.index(route+'_inverse_rank') for route in ROUTES]
    for p in sorted(out.glob(FEATURE_GLOB)):
        z=np.load(p);q=z['q'];X=z['X'];eid=z['eid'];src=int(p.name[1])
        for i,e in zip(q,eid):
            if str(e) in meta.get(str(i),{}):hitsets[int(i)].add(str(e))
        source_counts[src]+=np.bincount(q,minlength=n)
        for k in [10,20,50,100]:
            keep=np.any(X[:,rank_cols]>=1/k-1e-7,axis=1)
            counts[k]+=np.bincount(q[keep],minlength=n)
            for i,e in zip(q[keep],eid[keep]):
                f=data[i]['fold']
                if f in folds and str(e) in meta[str(i)]:hits[f,k].update(meta[str(i)][str(e)]['slices'])
    report={};misses=[]
    for f in folds:
        sel=np.array([i for i,r in enumerate(data) if r['fold']==f])
        report[f]={}
        for k in [10,20,50,100]:
            c=counts[k][sel]
            report[f][str(k)]={'recall':{s:{'true_links':v,'retrieved':hits[f,k][s],'recall':hits[f,k][s]/v} for s,v in totals[f].items()},
                              'counts':{'mean':float(c.mean()),'p50':float(np.quantile(c,.5)),'p90':float(np.quantile(c,.9)),
                                'p99':float(np.quantile(c,.99)),'max':int(c.max()),'pairs':int(c.sum()),
                                'reduction_ratio':1-float(c.sum())/(len(sel)*10320219)},
                              'source_pairs':{str(s):int(source_counts[s][sel].sum()) for s in [2,3]} if k==100 else {}}
        for i in sel:
            for e,m in meta[str(i)].items():
                if e not in hitsets[i]:misses.append({'s1':data[i]['entity_id'],'fold':f,'target':e,**m,'query':{k:data[i][k] for k in ['business_name','business_address','country']}})
        report[f]['100']['oracle_entity_metrics']=metrics(
            {data[i]['entity_id']:set(data[i]['truth']) for i in sel},
            {data[i]['entity_id']:hitsets[i] for i in sel})
    save(out/('candidate_report_final.json' if include_validation else 'candidate_report_development.json'),report)
    save(out/('retrieval_misses_final.json' if include_validation else 'retrieval_misses_development.json'),misses)
    print('Candidate recall', {f:report[f]['100']['recall']['all'] for f in folds},'seconds',time.perf_counter()-started,flush=True)
    return report

def training_pairs(data,out):
    rng=np.random.default_rng(SEED);xs=[];ys=[];ws=[];counter=Counter()
    for p in sorted(out.glob(FEATURE_GLOB)):
        z=np.load(p);X=z['X'];qi=z['q'];eid=z['eid']
        for i in np.unique(qi):
            if data[i]['fold']!='fit':continue
            ix=np.flatnonzero(qi==i);x=X[ix];truth=set(data[i]['truth']);y=np.array([e in truth for e in eid[ix]])
            hard=(x[:,0]>0)|(x[:,1]>0)|(x[:,2]>.5)|(x[:,16]>.5)|((x[:,25]>0)&(x[:,2]>.3))
            # All positives and hard negatives; top retrieval candidates; sampled tail with unbiased inverse-probability weights.
            rr=np.max(x[:,[FEATURES.index(r+'_inverse_rank') for r in ROUTES]],axis=1)
            top=np.zeros(len(x),bool);top[np.argsort(-rr,kind='stable')[:30]]=True
            keep=y|hard|top;tail=np.flatnonzero(~keep);take=min(20,len(tail))
            w=np.ones(len(x),dtype=np.float32)
            if take:
                selected=rng.choice(tail,take,replace=False);keep[selected]=True;w[selected]=len(tail)/take
            xs.append(x[keep]);ys.append(y[keep].astype(np.float32));ws.append(w[keep])
            counter.update({'available':len(x),'selected':int(keep.sum()),'positive':int(y.sum()),'hard_negative':int((hard&~y).sum()),'tail_selected':take})
    X=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws)
    save(out/'training_sampling.json',dict(counter,scheme='all positives; all exact/suffix/high-similarity/missing-address hard negatives; top 30 route-rank negatives; 20 random tail per entity/source with inverse inclusion weights',seed=SEED))
    return X,y,w

def eval_arrays(data,fold,q,eid,y,score,threshold):
    sel=np.array([i for i,r in enumerate(data) if r['fold']==fold]);n=len(data)
    keep=score>=threshold
    pc=np.bincount(q[keep],minlength=n)[sel];tp=np.bincount(q[keep],weights=y[keep],minlength=n)[sel]
    tc=np.array([len(data[i]['truth']) for i in sel]);empty=tc==0
    prec=np.divide(tp,pc,out=np.zeros(len(sel)),where=pc>0);rec=np.divide(tp,tc,out=np.zeros(len(sel)),where=tc>0)
    f=np.divide(1.25*tp,.25*tc+pc,out=np.zeros(len(sel)),where=(tc+pc)>0)
    both=empty&(pc==0);prec[both]=1;rec[both]=1;f[both]=1
    return dict(threshold=float(threshold),n_entities=len(sel),macro_f0_5=float(f.mean()),macro_precision=float(prec.mean()),macro_recall=float(rec.mean()),
                singleton_accuracy=float((pc[empty]==0).mean()) if empty.any() else None,singleton_count=int(empty.sum()),zero_match_rate=float((pc==0).mean()),
                average_links=float(pc.mean()),prediction_counts=dict(Counter(str(min(5,int(v))) for v in pc)),
                false_positive_links=int((pc-tp).sum()),false_negative_links=int((tc-tp).sum()))

def fit(n):
    started=time.perf_counter();data,out=load(n)
    if not (out/'candidate_report_development.json').exists():raise ValueError('Run retrieval diagnostics before fitting')
    X,y,w=training_pairs(data,out);print('training shape',X.shape,memory(),flush=True)
    results={}
    for label,model in [('logistic',Logistic()),('boosted',HistogramBoost())]:
        t=time.perf_counter();model.fit(X,y,w);state=model.state()
        state['parameter_count']=parameters(state);state['license']='MIT';state['features']=FEATURES
        state['implementation_version']='numpy-original-v1.0'
        save(out/f'{label}_model.json',state)
        reloaded=type(model).load(state);np.testing.assert_allclose(model.predict(X[:1000]),reloaded.predict(X[:1000]),atol=1e-6)
        scores=[];qs=[];es=[];labels=[]
        for p in sorted(out.glob(FEATURE_GLOB)):
            z=np.load(p);v=model.predict(z['X']).astype(np.float32);q=z['q'];e=z['eid']
            score_path=p.with_suffix('.'+label+'.scores.npz');np.savez_compressed(score_path,score=v.astype(np.float32))
            use=np.array([data[i]['fold']=='tune' for i in q])
            scores.append(v[use]);qs.append(q[use]);es.append(e[use]);labels.extend(int(ej in data[i]['truth']) for i,ej in zip(q[use],e[use]))
        score=np.concatenate(scores);q=np.concatenate(qs);eid=np.concatenate(es);yy=np.array(labels)
        broad=sorted(set(np.linspace(.01,.99,50).tolist()+[.001,.005,.995,.999,1.0]))
        grid=[eval_arrays(data,'tune',q,eid,yy,score,h) for h in broad]
        best=max(grid,key=lambda x:(x['macro_f0_5'],x['threshold']))
        fine=np.linspace(max(.0001,best['threshold']-.025),min(1,best['threshold']+.025),51)
        grid += [eval_arrays(data,'tune',q,eid,yy,score,h) for h in fine]
        best=max(grid,key=lambda x:(x['macro_f0_5'],x['threshold']))
        # Inspect source calibration on the tuning fold only. Keep a shared threshold for V1.
        calibration={}
        for src in ['S2','S3']:
            use=np.array([e.startswith(src) for e in eid]);bins=[]
            for lo,hi in zip([0,.1,.3,.5,.7,.9],[.1,.3,.5,.7,.9,1.00001]):
                ix=use&(score>=lo)&(score<hi)
                bins.append({'lower':lo,'upper':min(1,hi),'n':int(ix.sum()),'positive_rate':float(yy[ix].mean()) if ix.any() else None,'mean_score':float(score[ix].mean()) if ix.any() else None})
            calibration[src]=bins
        save(out/f'{label}_thresholds.json',{'fold':'tune','results':grid,'selected':best,'source_calibration':calibration,'calibration':'none; scores are uncalibrated','shared_threshold':True})
        results[label]={'selected_threshold':best['threshold'],'tune':best,'parameters':state['parameter_count'],'training_and_scoring_seconds':time.perf_counter()-t,'memory':memory()}
        print(label,results[label],flush=True)
    save(out/'models_complete.json',{'models':results,'seconds':time.perf_counter()-started,'memory':memory(),'negative_sampling':'training_sampling.json',
         'hyperparameters':{'logistic':{'steps':240,'learning_rate':.04,'L2':.0001},'boosted':{'trees':70,'depth':3,'bins':32,'learning_rate':.12,'L2_leaf':5}},
         'selection':'model and threshold chosen on tune only; no post-validation changes allowed'})

def final(n):
    data,out=load(n);model_result=json.loads((out/'models_complete.json').read_text(encoding='utf-8'))
    dev=json.loads((out/'candidate_report_development.json').read_text(encoding='utf-8'))
    selected=max(model_result['models'],key=lambda m:model_result['models'][m]['tune']['macro_f0_5'])
    # Persist model/threshold decision before final holdout labels are scored.
    save(out/'frozen_selection.json',{'selected':selected,'thresholds':{m:v['selected_threshold'] for m,v in model_result['models'].items()},
        'development_candidate_recall':dev['tune']['100']['recall']['all']['recall']})
    candidate_diagnostics(n,True);meta=truth_metadata(data,out,True)
    val=[i for i,r in enumerate(data) if r['fold']=='validation'];truth={data[i]['entity_id']:set(data[i]['truth']) for i in val}
    candidates={e:set() for e in truth};outputs={m:{e:set() for e in truth} for m in model_result['models']}
    errors={m:[] for m in outputs};link_counts={m:defaultdict(Counter) for m in outputs}
    for i in val:
        for e,md in meta[str(i)].items():
            for m in outputs:
                for s in md['slices']:link_counts[m][s]['truth']+=1
    for p in sorted(out.glob(FEATURE_GLOB)):
        z=np.load(p);q=z['q'];eid=z['eid'];X=z['X']
        for i,e in zip(q,eid):
            if data[i]['fold']=='validation':
                if str(e) in candidates[data[i]['entity_id']]:raise ValueError('Duplicate candidate')
                candidates[data[i]['entity_id']].add(str(e))
        for m in outputs:
            score=np.load(p.with_suffix('.'+m+'.scores.npz'))['score'];threshold=model_result['models'][m]['selected_threshold']
            for j in np.flatnonzero(score>=threshold):
                i=int(q[j]);e=str(eid[j]);r=data[i]
                if r['fold']!='validation':continue
                outputs[m][r['entity_id']].add(e);tp=e in r['truth']
                ss=['all',r['country'],e[:2]]
                if X[j,25]:ss.append('missing_address')
                if X[j,2]<.2:ss.append('low_name_similarity')
                if X[j,13]>=math.log1p(5):ss.append('common_name')
                if X[j,0]:ss.append('exact_name')
                # Script slices on predictions use raw candidate text, not label membership.
                rt=None
                # Predicted positives are few: lookups are performed in a separate pass below.
                for s in ss:link_counts[m][s]['predicted']+=1;link_counts[m][s]['tp']+=int(tp)
                if not tp:
                    cats=[]
                    if X[j,25]:cats.append('missing_address')
                    if X[j,0]:cats.append('exact_name')
                    if X[j,1]:cats.append('suffix_name')
                    if X[j,21]:cats.append('numeric_conflict')
                    if X[j,16]>.5:cats.append('strong_address')
                    errors[m].append({'kind':'false_positive','source1':r['entity_id'],'target':e,'score':float(score[j]),'categories':cats or ['other']})
    rts={s:Retriever(s) for s in [2,3]}
    for m,pr in outputs.items():
        for sid,ids in pr.items():
            for eid in ids:
                _,c=rts[int(eid[1])].lookup(eid)
                qview=view(next(r for r in data if r['entity_id']==sid))
                ss=slices(qview,c)
                for s in ss:
                    if s in ['cross_script','india_cross_script','non_latin_name'] or s.startswith('script_'):
                        link_counts[m][s]['predicted']+=1;link_counts[m][s]['tp']+=int(eid in truth[sid])
    results={}
    for m,pr in outputs.items():
        overall=metrics(truth,pr);sliced={}
        for country in sorted({data[i]['country'] for i in val}):
            ids={data[i]['entity_id'] for i in val if data[i]['country']==country}
            sliced['country_'+country]=metrics({e:truth[e] for e in ids},{e:pr[e] for e in ids})
        for card in range(6):
            ids={e for e,t in truth.items() if min(5,len(t))==card}
            if ids:sliced['true_cardinality_'+str(card)]=metrics({e:truth[e] for e in ids},{e:pr[e] for e in ids})
            ids={e for e,t in pr.items() if min(5,len(t))==card}
            if ids:sliced['predicted_cardinality_'+str(card)]=metrics({e:truth[e] for e in ids},{e:pr[e] for e in ids})
        for src in ['S2','S3']:
            sliced['source_'+src]=metrics({e:{x for x in t if x.startswith(src)} for e,t in truth.items()},
                                        {e:{x for x in p if x.startswith(src)} for e,p in pr.items()})
        for i in val:
            sid=data[i]['entity_id']
            for e in sorted(truth[sid]-pr[sid]):
                errors[m].append({'kind':'false_negative','source1':sid,'target':e,'categories':['retrieval_miss' if e not in candidates[sid] else 'classifier_miss']+meta[str(i)][e]['slices'][3:]})
        lc={s:dict(c,precision=c['tp']/c['predicted'] if c['predicted'] else None,recall=c['tp']/c['truth'] if c['truth'] else None) for s,c in link_counts[m].items()}
        results[m]={'overall':overall,'entity_slices':sliced,'link_slices':lc,'error_categories':{kind:dict(Counter(c for e in errors[m] if e['kind']==kind for c in e['categories'])) for kind in ['false_positive','false_negative']}}
        focus=sorted([e for e in errors[m] if e['kind']=='false_positive'],key=lambda e:(-e['score'],e['source1'],e['target']))[:10]
        focus += [e for e in errors[m] if e['kind']=='false_negative'][:10]
        query_by_id={r['entity_id']:r for r in data}
        examples=[]
        for error in focus:
            _,c=rts[int(error['target'][1])].lookup(error['target']);r=query_by_id[error['source1']]
            examples.append(dict(error,query={k:r[k] for k in ['entity_id','business_name','business_address','country']},candidate=c['raw']))
        save(out/f'{m}_error_examples.json',examples)
        save(out/f'{m}_errors.json',errors[m]);write_sets(out/f'validation_{m}_predictions.tsv',pr,'matched_entity_ids')
        assert all(pr[e]<=candidates[e] for e in pr)
    write_sets(out/'validation_candidates.tsv',candidates,'candidate_entity_ids')
    # Strict TSV roundtrip and valid-corpus target checks; no test files are involved.
    valid=set().union(*candidates.values());read_sets(out/'validation_candidates.tsv','candidate_entity_ids',truth,valid)
    for m in outputs:assert read_sets(out/f'validation_{m}_predictions.tsv','matched_entity_ids',truth,valid)==outputs[m]
    for rt in rts.values():rt.con.close()
    save(out/'final_results.json',{'selected_model':selected,'models':results,'memory':memory()})
    print('FINAL',selected,{m:r['overall'] for m,r in results.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnose','fit','final']);p.add_argument('--n',type=int,default=6000);a=p.parse_args()
    if a.stage=='diagnose':candidate_diagnostics(a.n)
    elif a.stage=='fit':fit(a.n)
    else:final(a.n)
