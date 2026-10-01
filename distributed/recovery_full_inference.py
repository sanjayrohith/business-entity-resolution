"""Replay exact R10 test retrieval and apply validated score-context meta model."""
import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/v2/packages'))
sys.path.insert(0,str(ROOT/'distributed'))
import lightgbm as lgb
import numpy as np
import psutil
import deadline_inference as d
from recovery_meta_experiment import BASE_COLS,context
from worker import preflight,file_hash
from merge import validate_pair

PROFILE='V2_RECOVERY_META_R10'
PENDING='V2_RECOVERY_META_R10_PENDING'
RESULT=ROOT/'submissions/recovery_02'
RUN=ROOT/'distributed/recovery/full_r10'
CHECKPOINTS=RUN/'checkpoints'
META=ROOT/'distributed/recovery/meta_r10_lightgbm.txt'
META_THRESHOLD=.7
OLD=ROOT/'distributed/deadline/checkpoints'

d.RUN=RUN
d.CHECKPOINTS=CHECKPOINTS


def retain_score_source(src,records,work,model,threads):
    started=time.perf_counter()
    paths=sorted(work.glob(f's{src}_*.npz'))
    if not paths:raise ValueError('No source feature files')
    xs=[];qs=[];es=[]
    for path in paths:
        with np.load(path) as z:
            xs.append(z['X']);qs.append(z['q']);es.append(z['eid'])
    X=np.concatenate(xs);q=np.concatenate(qs);eid=np.concatenate(es)
    if X.shape[1]!=80 or not np.isfinite(X).all():raise ValueError('Invalid V2 feature matrix')
    score=model.predict(X,num_threads=threads) if len(X) else np.empty(0)
    np.savez_compressed(work/f's{src}_meta.npz',X=X[:,BASE_COLS],q=q,eid=eid,score=np.asarray(score,np.float32))
    counts=np.bincount(q,minlength=len(records));bounds=np.r_[0,np.cumsum(counts)]
    links=0
    with open(work/f's{src}_candidate.tsv','w',encoding='utf-8',newline='') as cf,open(work/f's{src}_matching.tsv','w',encoding='utf-8',newline='') as mf:
        for i,record in enumerate(records):
            lo,hi=bounds[i:i+2]
            ids=eid[lo:hi].tolist()
            chosen=[target for target,s in zip(ids,score[lo:hi]) if s>=.642]
            if len(ids)!=len(set(ids)):raise ValueError('Duplicate scored candidate ID')
            cf.write(record['entity_id']+'\t'+','.join(ids)+'\n')
            mf.write(record['entity_id']+'\t'+','.join(chosen)+'\n')
            links+=len(chosen)
    for path in paths:
        path.unlink();path.with_suffix('.features.json').unlink()
    return {'seconds':time.perf_counter()-started,'pairs':len(eid),'links':links}


d.score_source_fast=retain_score_source


def finalize_batch(folder,records,meta_model):
    loaded=[]
    for src in (2,3):
        with np.load(folder/f's{src}_meta.npz') as z:
            loaded.append((z['X'],z['q'],z['eid'],z['score'],
                           np.full(len(z['q']),src,dtype=np.uint8)))
    X=np.concatenate([z[0] for z in loaded])
    q=np.concatenate([z[1] for z in loaded])
    eid=np.concatenate([z[2] for z in loaded])
    score=np.concatenate([z[3] for z in loaded])
    source=np.concatenate([z[4] for z in loaded])
    context_features=context(q,source,score)
    features=np.concatenate([X,context_features],axis=1)
    if features.shape[1]!=meta_model.num_feature():raise ValueError('Meta feature count mismatch')
    predicted=meta_model.predict(features,num_threads=6)>=META_THRESHOLD
    accepted=[set() for _ in records]
    for qi,target,keep in zip(q,eid,predicted):
        if keep:accepted[int(qi)].add(str(target))
    temporary=folder/'matching_part.tsv.meta.tmp'
    links=0
    with open(folder/'candidate_part.tsv',encoding='utf-8') as candidate,open(temporary,'w',encoding='utf-8',newline='') as out:
        if candidate.readline().rstrip('\n')!='source1_entity_id\tcandidate_entity_ids':raise ValueError('Candidate header mismatch')
        out.write('source1_entity_id\tmatched_entity_ids\n')
        for qi,record in enumerate(records):
            sid,values=candidate.readline().rstrip('\r\n').split('\t')
            if sid!=record['entity_id']:raise ValueError('Candidate row order mismatch')
            ids=values.split(',') if values else []
            chosen=[target for target in ids if target in accepted[qi]]
            if len(chosen)!=len(accepted[qi]):raise ValueError('Meta prediction absent from candidate set')
            out.write(sid+'\t'+','.join(chosen)+'\n')
            links+=len(chosen)
        if candidate.readline():raise ValueError('Extra candidate row')
    temporary.replace(folder/'matching_part.tsv')
    manifest=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    counted=validate_pair(folder/'matching_part.tsv',folder/'candidate_part.tsv',{r['entity_id'] for r in records})
    if counted!=(manifest['candidate_pair_count'],links):raise ValueError('Meta output verification failed')
    old=json.loads((OLD/folder.name/'complete.json').read_text(encoding='utf-8'))
    if manifest['candidate_part.tsv_sha256']!=old['candidate_part.tsv_sha256']:
        raise ValueError('R10 candidate set differs from frozen first submission')
    manifest.update(profile=PROFILE,predicted_link_count=links,
                    meta_model_sha256=file_hash(META),meta_threshold=META_THRESHOLD,
                    matching_part__meta='score-context LightGBM',
                    **{'matching_part.tsv_sha256':file_hash(folder/'matching_part.tsv')})
    d.save_json(folder/'complete.json',manifest)
    return {'candidate_pairs':manifest['candidate_pair_count'],'predicted_links':links}


def merge(pre):
    RESULT.mkdir(parents=True,exist_ok=True)
    matching_tmp=RESULT/'matching_results.tsv.tmp'
    candidate_tmp=RESULT/'candidate_pairs.tsv.tmp'
    rows=pairs=links=0
    with open(matching_tmp,'w',encoding='utf-8',newline='') as mf,open(candidate_tmp,'w',encoding='utf-8',newline='') as cf:
        mf.write('source1_entity_id\tmatched_entity_ids\n')
        cf.write('source1_entity_id\tcandidate_entity_ids\n')
        for batch_id,(start,records) in enumerate(d.iterate_batches(10000)):
            folder=CHECKPOINTS/f'batch_{batch_id:06d}'
            if not d.verify_completed(folder,pre,start,records):raise ValueError(f'Missing recovery checkpoint {batch_id}')
            manifest=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            if manifest.get('meta_model_sha256')!=file_hash(META) or manifest.get('meta_threshold')!=META_THRESHOLD:
                raise ValueError('Mismatched recovery meta model')
            pairs+=manifest['candidate_pair_count'];links+=manifest['predicted_link_count']
            with open(folder/'matching_part.tsv',encoding='utf-8') as a,open(folder/'candidate_part.tsv',encoding='utf-8') as b:
                if a.readline().rstrip('\n')!='source1_entity_id\tmatched_entity_ids' or b.readline().rstrip('\n')!='source1_entity_id\tcandidate_entity_ids':
                    raise ValueError('Recovery checkpoint header mismatch')
                for record in records:
                    ma=a.readline();ca=b.readline()
                    if ma.split('\t',1)[0]!=record['entity_id'] or ca.split('\t',1)[0]!=record['entity_id']:
                        raise ValueError('Recovery output row mismatch')
                    mf.write(ma);cf.write(ca);rows+=1
                if a.readline() or b.readline():raise ValueError('Extra recovery checkpoint row')
    if rows!=d.EXPECTED_ROWS:raise ValueError(f'Recovery S1 coverage mismatch {rows}')
    matching_tmp.replace(RESULT/'matching_results.tsv')
    candidate_tmp.replace(RESULT/'candidate_pairs.tsv')
    result={'source1_rows':rows,'candidate_pairs':pairs,'predicted_links':links,
            'matching_sha256':file_hash(RESULT/'matching_results.tsv'),
            'candidate_sha256':file_hash(RESULT/'candidate_pairs.tsv')}
    d.save_json(RESULT/'merge_complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('run','merge'))
    p.add_argument('--max-batches',type=int,default=0)
    a=p.parse_args()
    started=time.perf_counter()
    pre=preflight(verify_files=True)
    meta_model=lgb.Booster(model_file=str(META))
    if meta_model.num_feature()!=len(BASE_COLS)+14:raise ValueError('Meta model feature mismatch')
    CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    d.PROFILE=PROFILE
    if a.action=='merge':
        print('RECOVERY_MERGED',json.dumps(merge(pre)),flush=True)
        return
    base_model=lgb.Booster(model_file=str(ROOT/'artifacts/v2/run/lightgbm.txt'))
    count=0
    for batch_id,(start_row,records) in enumerate(d.iterate_batches(10000)):
        if a.max_batches and count>=a.max_batches:break
        folder=CHECKPOINTS/f'batch_{batch_id:06d}'
        if folder.exists():
            manifest=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            if manifest.get('profile')==PENDING:
                if not folder.resolve().is_relative_to(RUN.resolve()):raise ValueError('Unsafe partial checkpoint path')
                shutil.rmtree(folder)
            elif d.verify_completed(folder,pre,start_row,records):
                continue
        available=psutil.virtual_memory().available
        if available<3*2**30:raise RuntimeError('Unsafe RAM pressure')
        workers=4 if available>=5*2**30 else 2
        d.PROFILE=PENDING
        t=time.perf_counter()
        d.process_batch(records,start_row,batch_id,pre,base_model,6,workers)
        d.PROFILE=PROFILE
        counts=finalize_batch(folder,records,meta_model)
        count+=1
        print('RECOVERY_CHECKPOINT',json.dumps({'batch':batch_id,'source1_done':start_row+len(records),
              'seconds':round(time.perf_counter()-t,2),**counts,
              'available_ram':psutil.virtual_memory().available}),flush=True)
        gc.collect()
    if not a.max_batches:
        print('RECOVERY_INFERENCE_DONE','seconds',round(time.perf_counter()-started,2),flush=True)
        d.PROFILE=PROFILE
        print('RECOVERY_MERGED',json.dumps(merge(pre)),flush=True)


if __name__=='__main__':main()
