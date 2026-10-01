"""Fresh grouped train-only R10 retrieval and frozen V2 scoring."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/v2/packages'))
sys.path.insert(0, str(ROOT / 'code/business_entity_resolution/v2'))
sys.path.insert(0, str(ROOT / 'distributed'))
import lightgbm as lgb
import numpy as np

import deadline_inference as deadline
from pair_features import run as feature_run

OUT = Path(os.environ.get('RECOVERY_OUT', str(ROOT / 'distributed/recovery/fresh_r10')))
SPLIT = Path(os.environ.get('RECOVERY_SPLIT', str(ROOT / 'distributed/recovery/fresh_split.json')))
RECORDS = json.loads(SPLIT.read_text(encoding='utf-8'))['records']


def source_score(src):
    model = lgb.Booster(model_file=str(ROOT / 'artifacts/v2/run/lightgbm.txt'))
    path = OUT / f's{src}_scores.tsv'
    if path.exists():
        return
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('q\teid\tscore\tname_gram_jaccard\taddress_gram_jaccard\tname_token_jaccard\tnumeric_conflict\tmissing_c_address\tname_exact\taddress_exact\n')
        for q0 in range(0,len(RECORDS),25):
            with np.load(OUT / f's{src}_{q0:05d}.npz') as z:
                X,q,eid = z['X'],z['q'],z['eid']
            score = model.predict(X,num_threads=4) if len(X) else []
            for qi,target,s,x in zip(q,eid,score,X):
                f.write(f'{int(qi)}\t{target}\t{s:.12g}\t{x[2]:.7g}\t{x[16]:.7g}\t{x[4]:.7g}\t{x[21]:.7g}\t{x[25]:.7g}\t{x[0]:.7g}\t{x[15]:.7g}\n')
    print('SCORED',src,path,flush=True)


def assess():
    predictions = defaultdict(set)
    candidates = defaultdict(set)
    for src in (2,3):
        with open(OUT / f's{src}_scores.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                q,eid,score,*_ = line.rstrip('\n').split('\t')
                qi = int(q)
                candidates[qi].add(eid)
                if float(score) >= .642:
                    predictions[qi].add(eid)
    result = {}
    for fold in ('development','confirmation'):
        n = cand = links = found = fsum = psum = rsum = 0
        for qi,record in enumerate(RECORDS):
            if record['fold'] != fold:
                continue
            true = set(record['truth']); pred = predictions[qi]; candidate = candidates[qi]
            tp = len(true & pred); oracle_tp = len(true & candidate)
            n += 1; cand += len(candidate); links += len(true); found += oracle_tp
            fsum += 1 if not true and not pred else (1.25*tp/(.25*len(true)+len(pred)) if pred else 0)
            psum += 1 if not true and not pred else (tp/len(pred) if pred else 0)
            rsum += 1 if not true and not pred else (tp/len(true) if true else 0)
        result[fold] = {'n_source1':n,'candidate_pairs':cand,'true_links':links,
                        'true_links_retrieved':found,'candidate_recall':found/links,
                        'macro_f0_5':fsum/n,'macro_precision':psum/n,'macro_recall':rsum/n}
    (OUT / 'assessment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('ASSESSMENT',json.dumps(result),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=('run','retrieve','feature','score','assess'))
    p.add_argument('--source',type=int,choices=(2,3))
    p.add_argument('--part',type=int,default=0)
    a = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.stage == 'retrieve':
        deadline.INDEX = ROOT / 'distributed/recovery/train_index'
        deadline.fast_retrieve(RECORDS,a.source,OUT,6)
    elif a.stage == 'feature':
        feature_run(RECORDS,a.source,OUT,a.part,2)
    elif a.stage == 'score':
        source_score(a.source)
    elif a.stage == 'assess':
        assess()
    else:
        started = time.perf_counter()
        jobs=[]
        for src in (2,3):
            log = open(OUT / f's{src}_retrieval.log','w',encoding='utf-8')
            proc = subprocess.Popen([sys.executable,__file__,'retrieve','--source',str(src)],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log))
        codes=[p.wait() for p,_ in jobs]
        for _,log in jobs:log.close()
        if any(codes):raise RuntimeError(f'Retrieval failed {codes}')
        print('RETRIEVAL_DONE',round(time.perf_counter()-started,1),flush=True)
        jobs=[]
        for src in (2,3):
            for part in (0,1):
                log=open(OUT / f's{src}_feature_{part}.log','w',encoding='utf-8')
                proc=subprocess.Popen([sys.executable,__file__,'feature','--source',str(src),'--part',str(part)],stdout=log,stderr=subprocess.STDOUT)
                jobs.append((proc,log))
        codes=[p.wait() for p,_ in jobs]
        for _,log in jobs:log.close()
        if any(codes):raise RuntimeError(f'Feature generation failed {codes}')
        print('FEATURES_DONE',round(time.perf_counter()-started,1),flush=True)
        for src in (2,3):source_score(src)
        assess()
        print('FRESH_R10_DONE',round(time.perf_counter()-started,1),flush=True)


if __name__ == '__main__':
    main()
