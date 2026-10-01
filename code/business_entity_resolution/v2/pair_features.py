"""Batch target fetches, cached representations and focused V2 pair features."""
from common import *
import argparse,gzip,functools,gc
from rapidfuzz import fuzz
from anyascii import anyascii
import retrieval as v1
from sparse_retrieval import augment

EXTRA=['name_token_sort','name_partial','name_content_ratio','name_content_subset','translit_ratio','translit_token_set',
 'address_token_set','address_partial','address_alpha_ratio','numeric_fuzzy','name_best_margin','address_best_margin']
RESCUES=['translit_char','missing_name_char','joint_name_address']
FEATURES=v1.FEATURES+EXTRA+[r+'_'+suffix for r in RESCUES for suffix in ['hit','inverse_rank','score']]
v1.edit=lambda a,b:fuzz.ratio(a,b)/100 if a and b else 0.

@functools.lru_cache(maxsize=4000)
def cached_view(rawtuple):
    return view(dict(zip(['entity_id','business_name','business_address','country'],rawtuple)))
def extras(q,c):
    a=' '.join(t for t in q['nf'].split() if t not in LEGAL);b=' '.join(t for t in c['nf'].split() if t not in LEGAL)
    qa=anyascii(q['nf']).lower();ca=anyascii(c['nf']).lower()
    return [fuzz.token_sort_ratio(q['nf'],c['nf'])/100,fuzz.partial_ratio(q['nf'],c['nf'])/100,
       fuzz.ratio(a,b)/100 if a and b else 0,containment(set(a.split()),set(b.split())),
       fuzz.ratio(qa,ca)/100,fuzz.token_set_ratio(qa,ca)/100,
       fuzz.token_set_ratio(q['af'],c['af'])/100 if c['af'] else 0,fuzz.partial_ratio(q['af'],c['af'])/100 if c['af'] else 0,
       fuzz.token_set_ratio(' '.join(t for t in q['at'] if not t.isnumeric()),' '.join(t for t in c['at'] if not t.isnumeric()))/100 if c['af'] else 0,
       max((fuzz.ratio(x,y)/100 for x in q['nums'] for y in c['nums']),default=0),0,0]

def run(records,src,out,part=0,parts=1):
    start=time.perf_counter();con=connection(src);idf=read(V1/f's{src}_idf.json')['char_idf'];nrows=0
    route_map={'name_word':'rare_name_token','name_char':'name_char_tfidf','address_word':'address_token','address_char':'address_char',
       'exact_name':'exact_name','suffix_name':'suffix_name','address_key':'address_key'}
    with gzip.open(out/f's{src}_candidates.jsonl.gz','rt',encoding='utf8') as f:
        batch=[]
        def process(batch):
            if (batch[0]['q']//25)%parts!=part:return
            stem=out/f's{src}_{batch[0]["q"]:05d}'
            if stem.with_suffix('.npz').exists():return
            ids=sorted({int(rid) for b in batch for rid in b['candidates']});targets={}
            for i in range(0,len(ids),500):
                chunk=ids[i:i+500]
                for row in con.execute('SELECT rowid,eid,business_name,business_address,country FROM r WHERE rowid IN ('+','.join('?'*len(chunk))+')',chunk):targets[row[0]]=row[1:]
            X=[];qs=[];rr=[];ee=[]
            for b in batch:
                qi=b['q'];q=view(records[qi]);begin=len(X)
                for rid,hits in sorted(b['candidates'].items(),key=lambda x:int(x[0])):
                    rid=int(rid);c=cached_view(targets[rid]);routes={route_map[r]:v for r,v in hits.items() if r in route_map}
                    if 'address_word' in hits and q['nums']&c['nums']:routes['numeric_address']=hits['address_word']
                    rescuefeatures=[]
                    for route in RESCUES:
                        rank,score=hits.get(route,(0,0));rescuefeatures.extend([bool(rank),1/rank if rank else 0,score])
                    X.append(v1.features(q,c,routes,len(b['candidates']),b['nf'],b['sf'],src,idf)+extras(q,c)+rescuefeatures)
                    qs.append(qi);rr.append(rid);ee.append(c['raw']['entity_id'])
                # Relative evidence is not top-1 enforcement: every candidate can still pass.
                if len(X)>begin:
                    bestn=max(x[2] for x in X[begin:]);besta=max(x[16] for x in X[begin:])
                    for x in X[begin:]:x[69]=x[2]-bestn;x[70]=x[16]-besta
            arr=np.asarray(X,dtype=np.float32).reshape(-1,len(FEATURES));assert np.isfinite(arr).all()
            np.savez_compressed(stem.with_suffix('.npz'),X=arr,q=np.asarray(qs,np.int32),rid=np.asarray(rr,np.int32),eid=np.asarray(ee))
            save(stem.with_suffix('.features.json'),{'pairs':len(X),'memory':memory()})
            print('features',src,batch[-1]['q']+1,'pairs',len(X),'seconds',round(time.perf_counter()-start,1),flush=True)
        for line in f:
            batch.append(json.loads(line))
            if len(batch)==25:process(batch);batch=[]
        if batch:process(batch)
    con.close();save(out/f's{src}_features_complete_{part}.json',{'seconds':time.perf_counter()-start,'memory':memory(),'features':FEATURES,'part':part,'parts':parts,
      'differences_from_v1':'RapidFuzz normalized Indel replaces SequenceMatcher name_edit; sparse TFIDF route scores/ranks replace FTS seeded scores. Numeric route hit denotes shared numeric component within address-word retrieval.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=int,required=True);p.add_argument('--smoke',type=int,default=0);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);p.add_argument('--output',choices=['run','benchmark_parallel'],default='run');a=p.parse_args()
    records=read(V2/'split.json')['records']
    if a.smoke:records=[r for r in records if r['fold']=='fit'][:a.smoke]
    run(records,a.source,V2/(f'smoke_{a.smoke}' if a.smoke else a.output),a.part,a.parts)
