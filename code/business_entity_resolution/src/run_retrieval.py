"""Run bounded local worker processes; no agents, network, or test inference."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from prepare import CACHE,save
from retrieval import Retriever

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=6000);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('At most eight workers for RAM safety')
    # Build shared, deterministic IDF caches once before starting workers.
    for s in [2,3]:r=Retriever(s);r.con.close()
    start=time.perf_counter();processes=[];logs=[]
    for i in range(a.workers):
        log=open(CACHE/f'retrieval_worker_{i}.log','a',encoding='utf-8');logs.append(log)
        processes.append(subprocess.Popen([sys.executable,str(Path(__file__).with_name('retrieval.py')),'--n',str(a.n),'--part',str(i),'--parts',str(a.workers)],stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for f in logs:f.close()
    out=CACHE/f'run_{a.n}';save(out/'retrieval_walltime.json',{'seconds':time.perf_counter()-start,'workers':a.workers,'exit_codes':codes})
    if any(codes):raise SystemExit('A retrieval worker failed; inspect artifact logs')
    expected=2*((a.n+49)//50)
    if len(list(out.glob('s[23]_*.done.json')))!=expected:raise ValueError('Incomplete retrieval shards')
    print('Retrieval complete',time.perf_counter()-start,flush=True)
