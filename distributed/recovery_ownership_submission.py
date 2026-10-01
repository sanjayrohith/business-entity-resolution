"""Validated target-ownership alternative derived from recovery #2 scores.

Only selected duplicate target IDs are reconsidered. All other decisions and
every scored candidate remain byte-for-byte those of recovery #2.
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/v2/packages'))
sys.path.insert(0,str(ROOT/'distributed'))
import lightgbm as lgb
import numpy as np
from recovery_meta_experiment import context
from worker import file_hash
import finalize_deadline as final

SOURCE=ROOT/'submissions/recovery_02'
DEST=ROOT/'submissions/recovery_03'
CHECKPOINTS=ROOT/'distributed/recovery/full_r10/checkpoints'
MODEL=ROOT/'distributed/recovery/meta_r10_lightgbm.txt'
MARGIN=.05


def conflicts(path=None):
    first={};multiple={}
    with open(path or SOURCE/'matching_results.tsv',encoding='utf-8') as f:
        if f.readline().rstrip('\r\n')!='source1_entity_id\tmatched_entity_ids':
            raise ValueError('Recovery #2 matching header mismatch')
        for line in f:
            sid,values=line.rstrip('\r\n').split('\t')
            for eid in values.split(',') if values else ():
                if eid in multiple:multiple[eid].append(sid)
                elif eid in first:multiple[eid]=[first.pop(eid),sid]
                else:first[eid]=sid
    del first
    return multiple


def score_conflicts(multiple,max_folders=None):
    model=lgb.Booster(model_file=str(MODEL))
    results=defaultdict(dict)
    folders=sorted(CHECKPOINTS.glob('batch_*'))
    if max_folders is not None:folders=folders[:max_folders]
    for folder in folders:
        manifest=folder/'complete.json'
        if not manifest.exists():raise ValueError(f'Missing checkpoint manifest: {folder}')
        if json.loads(manifest.read_text(encoding='utf-8')).get('status')!='VERIFIED_COMPLETE':
            raise ValueError(f'Unverified checkpoint: {folder}')
        selected=set();needed=set()
        with open(folder/'matching_part.tsv',encoding='utf-8') as f:
            if f.readline().rstrip('\r\n')!='source1_entity_id\tmatched_entity_ids':
                raise ValueError('Checkpoint matching header mismatch')
            for qi,line in enumerate(f):
                sid,values=line.rstrip('\r\n').split('\t')
                for eid in values.split(',') if values else ():
                    if eid in multiple:
                        if sid not in multiple[eid]:raise ValueError('Owner mismatch')
                        selected.add((qi,eid,sid));needed.add(qi)
        if not needed:continue
        loaded=[]
        for src in (2,3):
            with np.load(folder/f's{src}_meta.npz') as z:
                mask=np.isin(z['q'],list(needed))
                loaded.append((z['X'][mask],z['q'][mask],z['eid'][mask],z['score'][mask],
                               np.full(mask.sum(),src,dtype=np.uint8)))
        X=np.concatenate([part[0] for part in loaded])
        q=np.concatenate([part[1] for part in loaded])
        eid=np.concatenate([part[2] for part in loaded])
        base=np.concatenate([part[3] for part in loaded])
        source=np.concatenate([part[4] for part in loaded])
        if len(q)==0:raise ValueError('Selected conflict without stored candidates')
        scores=model.predict(np.concatenate([X,context(q,source,base)],axis=1),num_threads=4)
        local_ids={qi:sid for qi,target,sid in selected}
        selected_pairs={(qi,target) for qi,target,sid in selected}
        found=set()
        for qi,target,s in zip(q,eid,scores):
            key=(int(qi),str(target))
            if key not in selected_pairs:continue
            sid=local_ids[int(qi)]
            if s<.7-1e-12:raise ValueError('Stored selected pair below frozen meta threshold')
            if sid in results[str(target)]:raise ValueError('Duplicate ownership score')
            results[str(target)][sid]=float(s)
            found.add(key)
        if found!=selected_pairs:raise ValueError(f'Missing scored conflict owner in {folder}')
        print('OWNERSHIP_CHECKPOINT',folder.name,len(selected),flush=True)
    for target,owners in multiple.items():
        if set(owners)!=set(results[target]):raise ValueError(f'Incomplete conflict: {target}')
    return results


def write_alternative(multiple,scores):
    winner={};dropped=0
    for target,owners in scores.items():
        ranked=sorted(owners.items(),key=lambda item:(-item[1],item[0]))
        if ranked[0][1]-ranked[1][1]>=MARGIN:winner[target]=ranked[0][0]
        else:winner[target]=None
        dropped+=len(ranked)-(winner[target] is not None)
    DEST.mkdir(parents=True,exist_ok=True)
    tmp=DEST/'matching_results.tsv.tmp'
    removed=0;rows=0
    with open(SOURCE/'matching_results.tsv',encoding='utf-8') as original,open(tmp,'w',encoding='utf-8',newline='') as out:
        out.write(original.readline())
        for line in original:
            sid,values=line.rstrip('\r\n').split('\t')
            ids=values.split(',') if values else []
            kept=[e for e in ids if e not in multiple or winner[e]==sid]
            removed+=len(ids)-len(kept);rows+=1
            out.write(sid+'\t'+','.join(kept)+'\n')
    if removed!=dropped:raise ValueError('Unexpected removed link count')
    os.replace(tmp,DEST/'matching_results.tsv')
    candidate=DEST/'candidate_pairs.tsv'
    if not candidate.exists():os.link(SOURCE/'candidate_pairs.tsv',candidate)
    if file_hash(candidate)!=file_hash(SOURCE/'candidate_pairs.tsv'):
        raise ValueError('Candidate hash mismatch')
    return {'conflicted_targets':len(multiple),'removed_links':removed,'source1_rows':rows,
            'winner_targets':sum(v is not None for v in winner.values()),
            'no_winner_targets':sum(v is None for v in winner.values())}


def main():
    started=time.monotonic()
    original=json.loads((SOURCE/'submission_manifest.json').read_text(encoding='utf-8'))
    if original.get('status')!='PASS':raise ValueError('Recovery #2 is not validated')
    if file_hash(MODEL)!=original.get('meta_model_sha256'):
        raise ValueError('Meta model differs from validated recovery #2')
    if len(list(CHECKPOINTS.glob('batch_*/complete.json')))!=174:
        raise ValueError('Expected 174 verified recovery checkpoints')
    if file_hash(SOURCE/'matching_results.tsv')!=original['sha256']['matching_results.tsv']:
        raise ValueError('Recovery #2 matching hash differs from its manifest')
    multiple=conflicts()
    print('TARGET_CONFLICTS',len(multiple),flush=True)
    if not multiple:raise ValueError('No target conflicts; no distinct submission to prepare')
    scored=score_conflicts(multiple)
    decision=write_alternative(multiple,scored)
    if conflicts(DEST/'matching_results.tsv'):
        raise ValueError('Ownership alternative still has duplicated target IDs')
    final.MATCHING=DEST/'matching_results.tsv'
    final.CANDIDATE=DEST/'candidate_pairs.tsv'
    final.WORK=DEST/'validation_work'
    summary=final.independent_scan()
    print('INDEPENDENT_PASS',json.dumps(summary),flush=True)
    validator=final.organizer_chunks()
    stress=json.loads((ROOT/'distributed/recovery/ambiguous_decode.json').read_text(encoding='utf-8'))
    manifest={'status':'PASS','profile':'V2_RECOVERY_META_R10_TARGET_EXCLUSIVE',
              'source_submission':'recovery_02','target_ownership_margin':MARGIN,
              'ownership_decision':decision,'summary':summary,
              'fresh_confirmation':original['fresh_confirmation'],
              'fresh_candidate_recall':original['candidate_recall'],
              'repeated_name_stress_development':stress['development']['exclusive_margin_0.05'],
              'repeated_name_stress_confirmation':stress['confirmation']['exclusive_margin_0.05'],
              'repeated_name_meta_only_confirmation':stress['confirmation']['meta'],
              'organizer_validator':{'mode':'18 exhaustive chunks using full official test S2/S3 and --check-ids',
                                     'chunks':validator},
              'runtime_seconds':time.monotonic()-started,
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'sha256':{'matching_results.tsv':file_hash(final.MATCHING),
                        'candidate_pairs.tsv':file_hash(final.CANDIDATE)},
              'base_manifest_sha256':file_hash(SOURCE/'submission_manifest.json'),
              'meta_model_sha256':file_hash(MODEL)}
    (DEST/'submission_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('SUBMISSION_3_READY',json.dumps({'sha256':manifest['sha256'],
          'decision':decision,'summary':summary}),flush=True)

if __name__=='__main__':main()
