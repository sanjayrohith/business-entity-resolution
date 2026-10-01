"""Batch sparse top-k retrieval, bounded target partitions, no dense products."""
from common import *
import argparse,gc,pickle
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from anyascii import anyascii
from rapidfuzz import fuzz

ROUTES=['name_word','name_char','address_word','address_char']
COLS=['rid','eid','business_name','business_address','country','n','a','s','ak']
SHARD=250000
def augment(s):
    if s.isascii():return s
    t=anyascii(s).lower()
    return s+' '+t if t!=s else s
def vectorizer(route):
    return TfidfVectorizer(analyzer='char' if 'char' in route else 'word',ngram_range=(3,5) if route=='name_char' else ((3,3) if route=='address_char' else (1,1)),
      token_pattern=r'(?u)\b\w+\b',lowercase=False,max_df=.003 if 'char' in route else .02,min_df=1,max_features=220000 if route=='name_char' else 160000,dtype=np.float32,binary=True)

def build(src,limit=0):
    out=V2/f'index_s{src}{"_smoke" if limit else ""}';out.mkdir(exist_ok=True)
    config={'source':src,'limit':limit,'shard':SHARD,'version':'sparse-v2.0','source_fingerprint':read(V1/f's{src}.json')['fingerprint']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')['config']==config;return
    start=time.perf_counter();con=connection(src);n=min(limit,read(V1/f's{src}.json')['rows']) if limit else read(V1/f's{src}.json')['rows']
    stride=max(1,n//100000)
    sample=con.execute('SELECT n,a FROM r WHERE rowid<=? AND rowid % ?=0',(n,stride)).fetchall()
    vecs={}
    for route in ROUTES:
        v=vectorizer(route);v.fit([augment(r[0 if route.startswith('name') else 1]) for r in sample]);vecs[route]=v
        with open(out/f'{route}.pkl','wb') as f:pickle.dump(v,f)
    del sample
    for offset in range(0,n,SHARD):
        end=min(n,offset+SHARD);p=out/f'{offset:08d}'
        if p.with_suffix('.done.json').exists():continue
        records=con.execute('SELECT rowid,eid,business_name,business_address,country,n,a,s,ak FROM r WHERE rowid>? AND rowid<=?',(offset,end)).fetchall()
        df=pl.DataFrame(records,schema=COLS,orient='row');df.write_parquet(p.with_suffix('.parquet'));del records
        for field in ['n','a']:
            texts=[augment(s) for s in df[field].to_list()]
            for route in [r for r in ROUTES if r.startswith('name' if field=='n' else 'address')]:
                mat=vecs[route].transform(texts).T.tocsr()
                sparse.save_npz(out/f'{offset:08d}_{route}.npz',mat,compressed=False)
                del mat;gc.collect()
            del texts
        save(p.with_suffix('.done.json'),{'rows':end-offset,'seconds':time.perf_counter()-start,'memory':memory()})
        print('indexed',src,end,'seconds',round(time.perf_counter()-start,1),'RAM',memory(),flush=True)
        del df;gc.collect()
    con.close();save(out/'manifest.json',{'config':config,'rows':n,'seconds':time.perf_counter()-start,'memory':memory(),'bytes':sum(p.stat().st_size for p in out.iterdir() if p.is_file())})

def top_merge(ids,scores,newids,newscores,k):
    ids=np.concatenate([ids,newids],axis=1);scores=np.concatenate([scores,newscores],axis=1)
    # Deterministic lexicographic ties by corpus row ID.
    ix=np.lexsort((ids,-scores),axis=1)[:,:k]
    return np.take_along_axis(ids,ix,axis=1),np.take_along_axis(scores,ix,axis=1)

def retrieve(records,src,out,limit=0,k=100,policy='all'):
    out.mkdir(exist_ok=True,parents=True);index=V2/f'index_s{src}{"_smoke" if limit else ""}'
    if not limit and (V2/f'index2_s{src}/manifest.json').exists():index=V2/f'index2_s{src}'
    meta=read(index/'manifest.json')
    rescues=not limit and (V2/f'targeted_s{src}/complete.json').exists() and 'hashed' in meta['config']['version']
    config={'index':meta['config'],'query_ids':[r['entity_id'] for r in records],'k':k,'targeted_rescues':rescues,'policy':policy,'version':'sparse-v2.2-joint'}
    if (out/f's{src}_complete.json').exists():assert read(out/f's{src}_complete.json')['config']==config;return
    start=time.perf_counter();n=len(records);views=[view(r) for r in records];times={};routeoutputs={}
    for route in ROUTES:
        t=time.perf_counter()
        with open(index/f'{route}.pkl','rb') as f:v=pickle.load(f)
        q=v.transform([x['nf' if route.startswith('name') else 'af'] if hasattr(v,'idf') else augment(x['nf' if route.startswith('name') else 'af']) for x in views])
        bestid=np.empty((n,0),np.int32);bestscore=np.empty((n,0),np.float32)
        for p in sorted(index.glob(f'*_{route}.npz')):
            offset=int(p.name.split('_')[0]);mat=sparse.load_npz(p)
            result=sp_matmul_topn(q,mat,top_n=k,threshold=.015,sort=True,n_threads=4)
            ids=np.zeros((n,k),np.int32);scores=np.zeros((n,k),np.float32)
            for i in range(n):
                lo,hi=result.indptr[i:i+2];ln=hi-lo;ids[i,:ln]=result.indices[lo:hi]+offset+1;scores[i,:ln]=result.data[lo:hi]
            bestid,bestscore=top_merge(bestid,bestscore,ids,scores,k)
            del mat,result;gc.collect()
        np.savez_compressed(out/f's{src}_{route}.npz',rid=bestid,score=bestscore)
        routeoutputs[route]=(bestid,bestscore);times[route]=time.perf_counter()-t
        print('route',src,route,round(times[route],2),'seconds',flush=True)
    if rescues:
        from targeted import retrieve as rescue
        extra,seconds=rescue(records,src,index);routeoutputs.update(extra);times['targeted_rescues']=seconds
        for route,(rid,score) in extra.items():np.savez_compressed(out/f's{src}_{route}.npz',rid=rid,score=score)
    # Batched exact-key joins preserve exact/suffix/reordered-address routes.
    con=connection(src);con.execute('CREATE TEMP TABLE queries(q INTEGER,n TEXT,s TEXT,ak TEXT)')
    con.executemany('INSERT INTO queries VALUES(?,?,?,?)',[(i,v['nf'],v['s'],' '.join(sorted(v['at']))) for i,v in enumerate(views)])
    exact={};freq=np.zeros((n,2),np.int32);t=time.perf_counter()
    for route,col in [('exact_name','n'),('suffix_name','s'),('address_key','ak')]:
        hits=[[] for _ in records]
        for qi,rid,addr in con.execute(f'SELECT q.q,r.rowid,r.a FROM queries q JOIN r ON r.{col}=q.{col} WHERE q.{col}<>\'\' AND r.rowid<=?',(meta['rows'],)):
            hits[qi].append((rid,fuzz.token_set_ratio(views[qi]['af'],addr)/100))
        for i,h in enumerate(hits):
            if route!='address_key':freq[i,0 if route=='exact_name' else 1]=len(h)
            h.sort(key=lambda x:(-x[1],x[0]));hits[i]=h[:k]
        exact[route]=hits
    con.close();times['exact_keys']=time.perf_counter()-t
    import gzip
    t=time.perf_counter()
    hardflags=np.zeros(meta['rows']+1,bool)
    if policy=='tiered':
        for name in ['translit_ids.npy','missing_ids.npy']:hardflags[np.load(V2/f'targeted_s{src}'/name)]=True
    with gzip.open(out/f's{src}_candidates.jsonl.gz','wt',encoding='utf8',compresslevel=1) as f:
        for i,r in enumerate(records):
            found={}
            for route,(ids,sc) in routeoutputs.items():
                for rank,(rid,s) in enumerate(zip(ids[i],sc[i]),1):
                    if rid:found.setdefault(int(rid),{})[route]=[rank,float(s)]
            for route,hits in exact.items():
                for rank,(rid,s) in enumerate(hits[i],1):found.setdefault(int(rid),{})[route]=[rank,float(s)]
            if policy=='tiered':found={rid:rs for rid,rs in found.items() if hardflags[rid] or any(v[0]<=50 for v in rs.values())}
            f.write(json.dumps({'q':i,'entity_id':r['entity_id'],'nf':int(freq[i,0]),'sf':int(freq[i,1]),'candidates':found})+'\n')
    times['union_dedup_write']=time.perf_counter()-t
    save(out/f's{src}_complete.json',{'config':config,'seconds':time.perf_counter()-start,'timings':times,'memory':memory()})
    print('retrieved',src,n,'seconds',time.perf_counter()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','retrieve']);p.add_argument('--source',type=int,choices=[2,3],required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',type=int,default=0);p.add_argument('--k',type=int,default=100);p.add_argument('--policy',choices=['all','tiered'],default='all');a=p.parse_args()
    if a.stage=='build':build(a.source,a.limit)
    else:
        records=read(V2/'split.json')['records']
        if a.smoke:records=[r for r in records if r['fold']=='fit'][:a.smoke]
        retrieve(records,a.source,V2/(f'smoke_{a.smoke}' if a.smoke else 'run'),a.limit,a.k,a.policy)
