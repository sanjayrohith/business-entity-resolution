"""Unprofiled route timing complements the overlapping cProfile call tree."""
from common import *
from retrieval import Retriever,features
from collections import Counter
class Timed(Retriever):
    def __init__(self,src):super().__init__(src);self.timing=Counter()
    def fts(self,table,expr,limit):
        kind='name_character_fts' if table=='tri' else ('numeric_address_fts' if ' AND ' in expr else ('address_token_fts' if expr.startswith('a') else 'name_token_fts'))
        t=time.perf_counter();result=super().fts(table,expr,limit);self.timing[kind]+=time.perf_counter()-t;return result
data=[r for r in read(V1/'split_6000.json')['records'] if r['fold']=='fit'][:10];timing=Counter();pairs=0
for src in [2,3]:
    rt=Timed(src)
    for r in data:
        t=time.perf_counter();q=view(r);timing['query_normalization']+=time.perf_counter()-t
        t=time.perf_counter();found,nf,sf=rt.query(q);timing['retrieval_total']+=time.perf_counter()-t
        t=time.perf_counter()
        for rid,h in found.items():features(q,rt.get(rid),h,len(found),nf,sf,src,rt.idf)
        timing['feature_generation']+=time.perf_counter()-t;pairs+=len(found)
    timing.update(rt.timing);rt.con.close();rt.get.cache_clear()
save(V2/'v1_route_timing.json',{'entities':10,'pairs':pairs,'seconds':dict(timing),'note':'Single process; FTS measurements are subsets of retrieval_total; excludes construction and disk serialization.'})
print(dict(timing),flush=True)
