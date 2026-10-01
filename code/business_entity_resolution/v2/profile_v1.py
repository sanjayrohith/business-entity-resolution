"""Deterministic fit-only end-to-end profiling; never changes V1 artifacts."""
import sys, json, time, cProfile, pstats, io, gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from retrieval import Retriever,features,serialize_view
from core import view
from prepare import ROOT,save,memory
import numpy as np

out=ROOT/'artifacts/v2';out.mkdir(exist_ok=True)
data=json.loads((ROOT/'artifacts/v1/split_6000.json').read_text())['records']
data=[r for r in data if r['fold']=='fit'][:30]
pr=cProfile.Profile();start=time.perf_counter();pairs=0
pr.enable()
for src in [2,3]:
    rt=Retriever(src);X=[];views={}
    for raw in data:
        q=view(raw);found,nf,sf=rt.query(q)
        for rid,hits in found.items():
            c=rt.get(rid);X.append(features(q,c,hits,len(found),nf,sf,src,rt.idf))
            views[c['raw']['entity_id']]=serialize_view(c)
    np.savez_compressed(out/f'profile_s{src}.npz',X=np.asarray(X,dtype=np.float32))
    with gzip.open(out/f'profile_s{src}.jsonl.gz','wt',encoding='utf8',compresslevel=1) as f:
        for v in views.values():f.write(json.dumps(v,ensure_ascii=False)+'\n')
    pairs+=len(X);rt.con.close();rt.get.cache_clear()
pr.disable();pr.dump_stats(str(out/'v1_profile.prof'))
s=io.StringIO();stats=pstats.Stats(pr,stream=s).sort_stats('cumulative');stats.print_stats(60)
(out/'v1_profile.txt').write_text(s.getvalue(),encoding='utf8')
save(out/'v1_profile.json',{'queries':len(data),'source_queries':len(data)*2,'pairs':pairs,'seconds':time.perf_counter()-start,'memory':memory(),
 'functions':[{'file':k[0],'line':k[1],'function':k[2],'calls':v[1],'self_seconds':v[2],'cumulative_seconds':v[3]} for k,v in stats.stats.items()]})
print(s.getvalue(),flush=True)
