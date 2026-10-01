"""Read-only corpus integrity, candidate lineage and local output verification."""
import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter
import numpy as np
from core import *
from prepare import ROOT,CACHE,save,memory
from experiment import load
from retrieval import FEATURES,Retriever,features,FEATURE_GLOB
from models import Logistic,HistogramBoost

def verify(n):
    started=time.perf_counter();data,out=load(n);checks={}
    target_folds={}
    for r in data:
        for e in r['truth']:target_folds.setdefault(e,set()).add(r['fold'])
    checks['shared_positive_targets_across_folds']=sum(len(x)>1 for x in target_folds.values())
    assert checks['shared_positive_targets_across_folds']==0
    expected=json.loads((CACHE/'data_fingerprints.json').read_text(encoding='utf-8'))
    actual={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'dataset').rglob('*.tsv')}
    assert all(actual[k]==v['sha256'] for k,v in expected.items());checks['original_sha256_unchanged']=True
    files=list(out.glob(FEATURE_GLOB))
    assert len(files)==2*((n+49)//50)
    corpus={s:sqlite3.connect(f'file:{(CACHE/f"s{s}.sqlite").as_posix()}?mode=ro',uri=True) for s in [2,3]}
    lineage=0;finite=0;val_ids={i for i,r in enumerate(data) if r['fold']=='validation'};rids={2:set(),3:set()}
    for p in sorted(files):
        z=np.load(p);src=int(p.name[1]);X=z['X'];assert X.shape[1]==len(FEATURES);assert np.isfinite(X).all();finite+=len(X)
        mapping=dict(zip(map(int,z['rid']),map(str,z['eid'])))
        ids=sorted(mapping)
        for start in range(0,len(ids),500):
            chunk=ids[start:start+500]
            found=dict(corpus[src].execute('SELECT rowid,eid FROM r WHERE rowid IN ('+','.join('?' for _ in chunk)+')',chunk))
            assert found=={i:mapping[i] for i in chunk}
        lineage+=len(X)
        for q,rid in zip(z['q'],z['rid']):
            if int(q) in val_ids:rids[src].add(int(rid))
    checks['pairs_verified_against_full_training_index']=lineage;checks['finite_feature_rows']=finite
    fixture=out/'validator_training_fixture';fixture.mkdir(exist_ok=True)
    cols=['entity_id','business_name','business_address','country']
    with open(fixture/'test_source1.tsv','w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE,quotechar=None,lineterminator='\n');w.writerow(cols)
        for i in sorted(val_ids):w.writerow([data[i][c] for c in cols])
    for src in [2,3]:
        with open(fixture/f'test_source{src}.tsv','w',encoding='utf-8',newline='') as f:
            w=csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE,quotechar=None,lineterminator='\n');w.writerow(cols)
            ids=sorted(rids[src])
            for j in range(0,len(ids),500):
                chunk=ids[j:j+500]
                w.writerows(corpus[src].execute('SELECT eid,business_name,business_address,country FROM r WHERE rowid IN ('+','.join('?' for _ in chunk)+') ORDER BY rowid',chunk))
        corpus[src].close()
    for model in ['logistic','boosted']:
        cmd=[sys.executable,str(ROOT/'utils/validate_submission.py'),'--matching',str(out/f'validation_{model}_predictions.tsv'),
             '--candidate',str(out/'validation_candidates.tsv'),'--test-dir',str(fixture),'--check-ids']
        child_env=dict(os.environ,PYTHONIOENCODING='utf-8')
        p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',env=child_env)
        (out/f'{model}_validator.log').write_text(p.stdout+p.stderr,encoding='utf-8')
        assert p.returncode==0 and 'WARNING:' not in p.stdout
        checks[model+'_official_validator']='PASS with --check-ids against training-only fixture'
    # Only read a bounded unlabeled test sample for schema/Unicode sanity; no retrieval or scores on test entities.
    sanity={}
    for src in [1,2,3]:
        counts=Counter();accent=0;nonlatin=0
        for j,r in enumerate(rows(ROOT/f'dataset/test/test_source{src}.tsv')):
            if j>=1000:break
            v=view(r);assert v['raw']==r;assert r['entity_id'].startswith(f'S{src}-')
            counts[r['country']]+=1;accent+=int(v['n']!=v['nf']);nonlatin+=int(bool(v['script']-{'LATIN'}))
            # Synthetic cross-source pairing only checks finite features; never makes test predictions.
            b=dict(r,entity_id='S3-synthetic',country='Previously unseen country')
            assert np.isfinite(features(v,view(b),{},0,0,0,3,{})).all()
        sanity[str(src)]={'rows':sum(counts.values()),'countries':dict(counts),'accent_fold_changed_names':accent,'non_latin_names':nonlatin}
    save(out/'unlabeled_schema_sanity.json',sanity);checks['unseen_country_unicode_sanity']=True
    result=json.loads((out/'final_results.json').read_text(encoding='utf-8'));truth={data[i]['entity_id']:set(data[i]['truth']) for i in val_ids}
    rng=np.random.default_rng(SEED);cis={}
    for model in ['logistic','boosted']:
        pred=read_sets(out/f'validation_{model}_predictions.tsv','matched_entity_ids',truth)
        recomputed=metrics(truth,pred)
        assert abs(recomputed['macro_f0_5']-result['models'][model]['overall']['macro_f0_5'])<1e-12
        grouped={}
        for i in val_ids:grouped.setdefault(data[i]['group'],[]).append(entity_metric(truth[data[i]['entity_id']],pred[data[i]['entity_id']])[2])
        sums=np.array([sum(v) for v in grouped.values()]);sizes=np.array([len(v) for v in grouped.values()]);samples=rng.integers(0,len(sums),(1000,len(sums)))
        boot=sums[samples].sum(axis=1)/sizes[samples].sum(axis=1)
        cis[model]={'cluster_bootstrap_95_percent':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'replicates':1000,'seed':SEED}
        # Check a real saved shard after model deserialization.
        state=json.loads((out/f'{model}_model.json').read_text(encoding='utf-8'));m=(Logistic if model=='logistic' else HistogramBoost).load(state)
        p=sorted(files)[0];z=np.load(p);s=np.load(p.with_suffix('.'+model+'.scores.npz'))['score']
        np.testing.assert_allclose(m.predict(z['X']),s,atol=1e-6);checks[model+'_reload_verified']=True
    save(out/'confidence_intervals.json',cis)
    checks['metric_recomputed_from_tsv']=True
    linear=json.loads((out/'logistic_model.json').read_text(encoding='utf-8'))
    boosted=json.loads((out/'boosted_model.json').read_text(encoding='utf-8'))
    gains=Counter()
    def collect_gain(tree):
        if 'leaf' in tree:return
        gains[FEATURES[tree['feature']]]+=tree['gain'];collect_gain(tree['left']);collect_gain(tree['right'])
    for tree in boosted['trees']:collect_gain(tree)
    save(out/'model_explanation.json',{'logistic_standardized_coefficients':dict(zip(FEATURES,linear['coef'][1:])),
        'logistic_intercept':linear['coef'][0],'boosted_training_split_gain':dict(gains.most_common()),
        'caution':'Associational model diagnostics, not causal feature effects. Correlated routes/features can share importance.'})
    test_run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'code/business_entity_resolution/src'),'-v'],capture_output=True,text=True,encoding='utf-8')
    (out/'automated_tests.log').write_text(test_run.stdout+test_run.stderr,encoding='utf-8')
    assert test_run.returncode==0;checks['automated_tests']='PASS (see automated_tests.log)'
    save(out/'license_verification.json',{'official_rule_source':'README.md Constraints item 5',
        'official_source_sha256':digest(ROOT/'README.md'),'model_license':'MIT','license_file':'code/business_entity_resolution/LICENSE',
        'license_sha256':digest(ROOT/'code/business_entity_resolution/LICENSE'),'pretrained_models':[],
        'models':{m:{k:json.loads((out/f'{m}_model.json').read_text(encoding='utf-8'))[k] for k in ['type','implementation_version','parameter_count','license']} for m in ['logistic','boosted']},
        'dependency_distinction':'NumPy is numerical software; neither model imports pretrained weights or an external identity resource.'})
    code={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'code/business_entity_resolution').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    save(out/'code_fingerprints.json',code)
    critical=['config.json','logistic_model.json','boosted_model.json','logistic_thresholds.json','boosted_thresholds.json',
              'frozen_selection.json','validation_candidates.tsv','validation_logistic_predictions.tsv','validation_boosted_predictions.tsv','final_results.json']
    save(out/'artifact_manifest.json',{name:{'bytes':(out/name).stat().st_size,'sha256':digest(out/name)} for name in critical})
    checks['seconds']=time.perf_counter()-started;checks['memory']=memory();save(out/'verification.json',checks)
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=6000);a=p.parse_args();verify(a.n)
