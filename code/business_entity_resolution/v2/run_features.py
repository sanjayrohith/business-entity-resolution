"""Four bounded local CPU processes; no external tasks or test inference."""
from common import *
import subprocess
start=time.perf_counter();jobs=[]
for src in [2,3]:
    for part in [0,1]:
        log=open(V2/f'features_s{src}_{part}.log','w',encoding='utf8')
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('pair_features.py')),'--source',str(src),'--part',str(part),'--parts','2'],stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log))
codes=[p.wait() for p,l in jobs]
for p,l in jobs:l.close()
save(V2/'run/features_walltime.json',{'seconds':time.perf_counter()-start,'workers':4,'exit_codes':codes})
assert not any(codes),codes
