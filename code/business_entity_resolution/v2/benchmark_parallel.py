"""Label-free throughput benchmark of identical frozen training-query work.

Run only after model stages finish, so concurrent fitting does not distort timings.
"""
from common import *
import subprocess,argparse

def child(src):
    import sparse_retrieval,targeted
    from sparse_dot_topn import sp_matmul_topn
    # Same arithmetic/candidate policy, more independent native worker threads.
    def six(*args,**kwargs):kwargs['n_threads']=6;return sp_matmul_topn(*args,**kwargs)
    sparse_retrieval.sp_matmul_topn=six;targeted.sp_matmul_topn=six
    sparse_retrieval.retrieve(read(V2/'split.json')['records'],src,V2/'benchmark_parallel',k=100,policy='tiered')

def run():
    out=V2/'benchmark_parallel';out.mkdir(exist_ok=True)
    if (out/'benchmark.json').exists():return
    started=time.perf_counter();jobs=[]
    for src in [2,3]:
        log=open(out/f's{src}.log','w',encoding='utf8');p=subprocess.Popen([sys.executable,__file__,'--source',str(src)],stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
    codes=[p.wait() for p,l in jobs]
    for p,l in jobs:l.close()
    assert not any(codes),codes
    seconds=time.perf_counter()-started
    # Logical content equality, independent of gzip headers/timestamps.
    import gzip
    equality={}
    for src in [2,3]:
        with gzip.open(out/f's{src}_candidates.jsonl.gz','rb') as a,gzip.open(V2/f'run/s{src}_candidates.jsonl.gz','rb') as b:
            equality[str(src)]=hashlib.sha256(a.read()).hexdigest()==hashlib.sha256(b.read()).hexdigest()
        assert equality[str(src)]
    save(out/'benchmark.json',{'seconds':seconds,'entities':10000,'source_processes':2,'native_threads_each':6,'candidate_content_identical':equality,
      'peak_ram_bound':sum(read(out/f's{s}_complete.json')['memory']['peak_rss_bytes'] for s in [2,3]),'estimated_test_hours':seconds*1732544/10000/3600})
    print('parallel retrieval',seconds,'seconds',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=int,default=0);a=p.parse_args()
    if a.source:child(a.source)
    else:run()
