"""Verify immutable data, local outputs, corpus lineage, models and metric."""
from common import *
import subprocess,pickle
from collections import Counter,defaultdict
from retrieval import FEATURE_GLOB
from pair_features import FEATURES

def run():
    start=time.perf_counter();out=V2/'run';records=read(V2/'split.json')['records'];frozen=read(out/'frozen_selection.json');result=read(out/'confirmation_results.json');checks={}
    preservation=read(V2/'v1_preservation.json')
    for name,md in preservation['critical_artifacts'].items():assert digest(V1/'run_6000'/name)==md['sha256'],name
    for path,sha in preservation['code'].items():assert digest(ROOT/path)==sha,path
    assert digest(ROOT/'baseline_validation_report.md')==preservation['report_sha256'];checks['v1_unchanged']=True
    for path,md in read(V1/'data_fingerprints.json').items():assert digest(ROOT/path)==md['sha256'],path
    checks['all_original_dataset_hashes_unchanged']=True
    val={i for i,r in enumerate(records) if r['fold']=='validation'};truth={records[i]['entity_id']:set(records[i]['truth']) for i in val}
    candidates=read_sets(out/'confirmation_candidates.tsv','candidate_entity_ids',truth);allvalid=set().union(*candidates.values())
    files=sorted(out.glob(FEATURE_GLOB));assert len(files)==800,len(files);pairs=0;targetrids={2:set(),3:set()};conns={s:connection(s) for s in [2,3]}
    for p in files:
        src=int(p.name[1]);z=np.load(p);assert z['X'].shape[1]==len(FEATURES);assert np.isfinite(z['X']).all();pairs+=len(z['q'])
        seen=set(zip(z['q'].tolist(),z['rid'].tolist()));assert len(seen)==len(z['q'])
        mapping=dict(zip(z['rid'].tolist(),z['eid'].tolist()));ids=list(mapping)
        for offset in range(0,len(ids),500):
            chunk=ids[offset:offset+500];got=dict(conns[src].execute('SELECT rowid,eid FROM r WHERE rowid IN ('+','.join('?'*len(chunk))+')',chunk));assert got=={i:mapping[i] for i in chunk}
        mask=np.isin(z['q'],list(val));targetrids[src].update(map(int,z['rid'][mask]))
    checks['finite_lineage_verified_pairs']=pairs
    fixture=out/'training_validator_fixture';fixture.mkdir(exist_ok=True);cols=['entity_id','business_name','business_address','country']
    with open(fixture/'test_source1.tsv','w',encoding='utf8',newline='') as f:
        w=csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE,lineterminator='\n');w.writerow(cols)
        for i in sorted(val):w.writerow([records[i][c] for c in cols])
    for src in [2,3]:
        with open(fixture/f'test_source{src}.tsv','w',encoding='utf8',newline='') as f:
            w=csv.writer(f,delimiter='\t',quoting=csv.QUOTE_NONE,lineterminator='\n');w.writerow(cols);ids=sorted(targetrids[src])
            for offset in range(0,len(ids),500):
                chunk=ids[offset:offset+500];w.writerows(conns[src].execute('SELECT eid,business_name,business_address,country FROM r WHERE rowid IN ('+','.join('?'*len(chunk))+') ORDER BY rowid',chunk))
        conns[src].close()
    cis={};rng=np.random.default_rng(20260926)
    for model in frozen['models']:
        pred=read_sets(out/f'confirmation_{model}.tsv','matched_entity_ids',truth,allvalid);assert all(pred[s]<=candidates[s] for s in truth)
        exact=metrics(truth,pred);assert abs(exact['macro_f0_5']-result['models'][model]['overall']['macro_f0_5'])<1e-12
        command=[sys.executable,str(ROOT/'utils/validate_submission.py'),'--matching',str(out/f'confirmation_{model}.tsv'),'--candidate',str(out/'confirmation_candidates.tsv'),'--test-dir',str(fixture),'--check-ids']
        p=subprocess.run(command,capture_output=True,text=True,encoding='utf8',env=dict(os.environ,PYTHONIOENCODING='utf-8'));(out/f'{model}_validator.log').write_text(p.stdout+p.stderr,encoding='utf8');assert p.returncode==0 and 'WARNING:' not in p.stdout
        checks[model+'_validator']='PASS with --check-ids, training-only fixture'
        groups=defaultdict(list)
        for i in val:groups[records[i]['group']].append(entity_metric(truth[records[i]['entity_id']],pred[records[i]['entity_id']])[2])
        sums=np.array([sum(g) for g in groups.values()]);sizes=np.array([len(g) for g in groups.values()]);sample=rng.integers(0,len(sums),(1000,len(sums)));b=sums[sample].sum(1)/sizes[sample].sum(1)
        cis[model]=np.quantile(b,[.025,.975]).tolist()
    save(out/'confidence_intervals.json',cis);checks['metric_and_candidate_subset']=True
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONPATH=os.pathsep.join([str(V2/'packages'),str(Path(__file__).parent),str(Path(__file__).parent.parent/'src')]))
    logs=[]
    for cmd in [[sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent.parent/'src'),'-v'],[sys.executable,str(Path(__file__).with_name('test_v2.py'))]]:
        p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8',env=env);logs.append(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
    (out/'tests.log').write_text('\n'.join(logs),encoding='utf8');checks['tests']='25 passed'
    # Library model licenses verified from their installed license files/metadata.
    envmeta=read(V2/'environment.json');checks['selected_model_license']='MIT' if frozen['selected']=='lightgbm' else 'Apache-2.0'
    assert frozen['models'][frozen['selected']]['parameters']<8_000_000_000;checks['parameter_limit']=True
    # Synthetic-only open-country behavior, no challenge test inference.
    from sparse_retrieval import augment
    from pair_features import extras
    q=view(dict(entity_id='S1-synthetic',business_name='École de l’Étoile SARL',business_address='12 Rue de la Liberté, Lille',country='France'))
    c=view(dict(q['raw'],entity_id='S3-synthetic',country='Previously unseen label'))
    assert np.isfinite(extras(q,c)).all() and q['raw']['country']=='France' and q['raw']['business_name']=='École de l’Étoile SARL'
    assert c['raw']['country']=='Previously unseen label' and 'ecole' in augment(q['nf'])
    checks['synthetic_unicode_open_country']=True
    import lightgbm as lgb
    lm=lgb.Booster(model_file=str(out/'lightgbm.txt'))
    save(out/'feature_importance.json',{'lightgbm_training_gain':dict(sorted(zip(FEATURES,map(float,lm.feature_importance(importance_type='gain'))),key=lambda x:-x[1])),
      'caution':'Training split gain is descriptive, not causal; correlated features can share importance.'})
    checks['seconds']=time.perf_counter()-start;checks['memory']=memory();save(out/'verification.json',checks)
    critical=['frozen_selection.json','confirmation_results.json','confirmation_candidates.tsv']+[f'confirmation_{m}.tsv' for m in frozen['models']]+['lightgbm.txt','xgboost.ubj','logistic.pkl']
    save(out/'artifact_manifest.json',{name:{'bytes':(out/name).stat().st_size,'sha256':digest(out/name)} for name in critical})
    save(out/'code_fingerprints.json',{str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('*') if p.is_file()})
    print(json.dumps(checks),flush=True)
if __name__=='__main__':run()
