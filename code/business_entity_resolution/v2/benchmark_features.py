"""Eight-worker benchmark, checking every feature against the validated run."""
from common import *
from retrieval import FEATURE_GLOB
import subprocess
out=V2/'benchmark_parallel'
if (out/'feature_benchmark.json').exists():raise SystemExit(0)
assert (out/'benchmark.json').exists()
start=time.perf_counter();jobs=[]
for src in [2,3]:
    for part in range(4):
        log=open(out/f'features_{src}_{part}.log','w',encoding='utf8');p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('pair_features.py')),'--source',str(src),'--part',str(part),'--parts','4','--output','benchmark_parallel'],stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
codes=[p.wait() for p,l in jobs]
for p,l in jobs:l.close()
assert not any(codes),codes
seconds=time.perf_counter()-start;pairs=0
for p in sorted((V2/'run').glob(FEATURE_GLOB)):
    a=np.load(p);b=np.load(out/p.name)
    for key in ['q','rid','eid','X']:np.testing.assert_array_equal(a[key],b[key])
    pairs+=len(a['q'])
peak=sum(read(p)['memory']['peak_rss_bytes'] for p in out.glob('s*_features_complete_*.json'))
save(out/'feature_benchmark.json',{'seconds':seconds,'workers':8,'identical_feature_pairs':pairs,'peak_ram_bound':peak,'estimated_test_hours':seconds*1732544/10000/3600})
print('parallel features',seconds,'seconds',flush=True)
