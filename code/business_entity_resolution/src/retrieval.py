"""Auditable bounded retrieval over full S2/S3. No labels used by Retriever."""
import argparse
import functools
import gzip
import json
import math
import sqlite3
import time
from collections import Counter,defaultdict
from difflib import SequenceMatcher
import numpy as np
from core import *
from prepare import CACHE,ROOT,memory,save

ROUTES=['exact_name','suffix_name','rare_name_token','name_char_tfidf','address_token','address_key','numeric_address','address_char']
FEATURE_GLOB='s[23]_[0-9][0-9][0-9][0-9][0-9].npz'
FEATURES=['name_exact','suffix_exact','name_gram_jaccard','name_tfidf','name_token_jaccard','name_containment','name_edit',
 'name_length_ratio','name_token_count_delta','name_prefix','name_suffix','acronym','script_compatible','name_frequency','suffix_frequency',
 'address_exact','address_gram_jaccard','address_token_jaccard','address_containment','address_length_ratio',
 'numeric_jaccard','numeric_conflict','postal_agreement','postal_conflict','missing_q_address','missing_c_address',
 'name_address_product','exact_name_address_conflict','name_missing_address','weak_name_strong_address',
 'country_equal','source3','candidate_density','accent_name_exact','unicode_name_exact']
FEATURES += [f'{r}_{x}' for r in ROUTES for x in ['hit','inverse_rank','score']]

def cosine(a,b,idf):
    if not a or not b:return 0.0
    dot=sum(idf.get(t,1.)**2 for t in a&b)
    return dot/math.sqrt(sum(idf.get(t,1.)**2 for t in a)*sum(idf.get(t,1.)**2 for t in b))

def ratio(a,b):return min(len(a),len(b))/max(len(a),len(b),1)
def edit(a,b):return SequenceMatcher(None,a,b,autojunk=False).ratio() if a and b else 0.
def acronym(v):return ''.join(t[0] for t in v['s'].split())

def features(q,c,routes,density,nfreq,sfreq,src,idf):
    nj=jaccard(q['ng'],c['ng']); aj=jaccard(q['ag'],c['ag']); ne=float(q['n']==c['n'])
    nm=c['missing']; nc=float(bool(q['nums'] and c['nums'] and not q['nums']&c['nums']))
    pe=float(bool(q['postal']&c['postal'])); pc=float(bool(q['postal'] and c['postal'] and not pe))
    z=[ne,float(bool(q['s']) and q['s']==c['s']),nj,cosine(q['ng'],c['ng'],idf),jaccard(q['nt'],c['nt']),containment(q['nt'],c['nt']),edit(q['nf'],c['nf']),
        ratio(q['nf'],c['nf']),abs(len(q['nt'])-len(c['nt'])),q['nf'][:4]==c['nf'][:4],q['nf'][-4:]==c['nf'][-4:],
        float(bool(acronym(q)) and (acronym(q)==acronym(c) or acronym(q)==c['s'].replace(' ','') or acronym(c)==q['s'].replace(' ',''))),
        bool(q['script']&c['script']),math.log1p(nfreq),math.log1p(sfreq),bool(q['a']) and q['a']==c['a'],aj,jaccard(q['at'],c['at']),containment(q['at'],c['at']),ratio(q['af'],c['af']),
        jaccard(q['nums'],c['nums']),nc,pe,pc,q['missing'],nm,nj*aj,ne*nc,nj*nm,(1-nj)*aj,
        q['raw']['country']==c['raw']['country'],src==3,math.log1p(density),q['nf']==c['nf'],q['unicode_name']==c['unicode_name']]
    for r in ROUTES:
        rank,score=routes.get(r,(0,0))
        z += [bool(rank),1/rank if rank else 0,score]
    return z

class Retriever:
    def __init__(self,src,smoke=0):
        self.src=src;self.label=f's{src}'+(f'_smoke{smoke}' if smoke else '')
        meta=json.loads((CACHE/f'{self.label}.json').read_text(encoding='utf-8'))
        if not meta['complete']:raise ValueError('Incomplete corpus index')
        self.n=meta['rows'];self.con=sqlite3.connect(f'file:{(CACHE/(self.label+".sqlite")).as_posix()}?mode=ro',uri=True)
        self.con.execute('PRAGMA cache_size=-98304')
        self.df,self.idf,self.adf=self.statistics()

    def statistics(self):
        p=CACHE/f'{self.label}_idf.json'
        if p.exists():
            d=json.loads(p.read_text(encoding='utf-8'));return d['token_df'],d['char_idf'],d['address_df']
        df=Counter();gd=Counter();adf=Counter();n=0
        # Label-free, fixed row stride sample across the entire source.
        stride=max(1,self.n//20000)
        for ntext,atext in self.con.execute('SELECT n,a FROM r WHERE rowid % ? = 0',(stride,)):
            n+=1;df.update(set(ntext.split()));adf.update(set(atext.split()));gd.update(grams(ntext))
        idf={g:1+math.log((n+1)/(d+1)) for g,d in gd.items()}
        save(p,dict(sample_rows=n,stride=stride,token_df=df,char_idf=idf,address_df=adf,version=VERSION))
        return df,idf,adf

    @functools.lru_cache(maxsize=3000)
    def get(self,rid):
        r=self.con.execute('SELECT eid,business_name,business_address,country FROM r WHERE rowid=?',(int(rid),)).fetchone()
        if r is None: raise ValueError('Unknown row ID')
        return view(dict(zip(['entity_id','business_name','business_address','country'],r)))

    def lookup(self,eid):
        row=self.con.execute('SELECT rowid FROM r WHERE eid=?',(eid,)).fetchone()
        if row is None:raise ValueError('Unknown official target ID '+eid)
        return row[0],self.get(row[0])

    def fts(self,table,expr,limit):
        if not expr:return []
        return [r[0] for r in self.con.execute(f'SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?',(expr,limit))]

    @staticmethod
    def term(t):return '"'+t.replace('"','""')+'"'

    def query(self,q,k=100,seed_budget=400):
        found=defaultdict(dict)
        def add(route,ranking):
            for rank,(rid,score) in enumerate(ranking[:k],1):
                found[int(rid)][route]=[rank,float(score)]
        nkeys=[r[0] for r in self.con.execute('SELECT rowid FROM r WHERE n=?',(q['nf'],))] if q['nf'] else []
        skeys=[r[0] for r in self.con.execute('SELECT rowid FROM r WHERE s=?',(q['s'],))] if q['s'] else []
        # Use address agreement to truncate collision blocks; no label access.
        def addr_rank(ids):return sorted(((i,jaccard(q['ag'],self.get(i)['ag'])) for i in ids),key=lambda x:(-x[1],x[0]))
        add('exact_name',addr_rank(nkeys));add('suffix_name',addr_rank(skeys))
        nt=sorted(q['nt']-LEGAL,key=lambda t:(self.df.get(t,0),t))[:4]
        name_ids=self.fts('lex','n : ('+' OR '.join(map(self.term,nt))+')' if nt else '',seed_budget)
        add('rare_name_token',sorted(((i,containment(q['nt'],self.get(i)['nt'])) for i in name_ids),key=lambda x:(-x[1],x[0])))
        # FTS seeds are approximate: eight rare, spaced trigrams; TF-IDF reranks 3–5-grams.
        tg=[q['nf'][i:i+3] for i in range(max(0,len(q['nf'])-2)) if ' ' not in q['nf'][i:i+3]]
        tg=sorted(set(tg),key=lambda g:(-self.idf.get(g,20.),g))[:8]
        tri_ids=self.fts('tri',' OR '.join(map(self.term,tg)),seed_budget)
        add('name_char_tfidf',sorted(((i,cosine(q['ng'],self.get(i)['ng'],self.idf)) for i in set(tri_ids)|set(name_ids)),key=lambda x:(-x[1],x[0])))
        at=sorted((t for t in q['at'] if len(t)>2),key=lambda t:(self.adf.get(t,0),t))[:5]
        address_ids=self.fts('lex','a : ('+' OR '.join(map(self.term,at))+')' if at else '',seed_budget)
        add('address_token',sorted(((i,jaccard(q['at'],self.get(i)['at'])) for i in address_ids),key=lambda x:(-x[1],x[0])))
        add('address_char',addr_rank(address_ids))
        ak=' '.join(sorted(q['at']))
        ids=[r[0] for r in self.con.execute('SELECT rowid FROM r WHERE ak=?',(ak,))] if ak else []
        add('address_key',[(i,1.) for i in ids])
        numbers=sorted(q['nums']|q['postal'],key=lambda t:(self.adf.get(t,0),t))[:8]
        words=sorted(t for t in q['at'] if not t.isnumeric() and len(t)>2)
        expr='a : (('+' OR '.join(map(self.term,numbers))+') AND ('+' OR '.join(map(self.term,words))+'))' if numbers and words else ''
        ids=self.fts('lex',expr,seed_budget)
        add('numeric_address',sorted(((i,jaccard(q['nums'],self.get(i)['nums'])+jaccard(q['at'],self.get(i)['at'])) for i in ids),key=lambda x:(-x[1],x[0])))
        return dict(found),len(nkeys),len(skeys)

def serialize_view(v):
    return {k:sorted(value) if isinstance(value,(set,frozenset)) else value for k,value in v.items()}

def run(n,smoke_queries=0,smoke_index=0,part=0,parts=1):
    start=time.perf_counter();out=CACHE/(f'smoke_{smoke_queries}_v2' if smoke_queries else f'run_{n}')
    out.mkdir(exist_ok=True)
    config={'version':VERSION,'retrieval_version':2,'n':n,'k':100,'seed_budget':400,'routes':ROUTES,'smoke_index':smoke_index,'smoke_queries':smoke_queries,
            'split_sha256':digest(CACHE/f'split_{n}.json')}
    # Corpus hashes are in the index manifests; no cache silently reused after a source/config change.
    config['index_manifests']={str(s):json.loads((CACHE/f's{s}{"_smoke"+str(smoke_index) if smoke_index else ""}.json').read_text(encoding='utf-8'))['fingerprint'] for s in ([2] if smoke_index else [2,3])}
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Run config mismatch')
    else:save(out/'config.json',config)
    data=json.loads((CACHE/f'split_{n}.json').read_text(encoding='utf-8'))['records']
    # Smoke only fitting entities; final validation labels are never inspected while developing retrieval.
    if smoke_queries:data=[r for r in data if r['fold']=='fit'][:smoke_queries]
    sources=[2] if smoke_index else [2,3]
    for src in sources:
        rt=Retriever(src,smoke_index)
        for offset in range(0,len(data),50):
            if (offset//50)%parts!=part:continue
            stem=out/f's{src}_{offset:05d}'
            if stem.with_suffix('.done.json').exists():continue
            xx=[];qq=[];rr=[];eids=[];audit=[];derived={}
            for qidx,raw in enumerate(data[offset:offset+50],offset):
                q=view({k:raw[k] for k in ['entity_id','business_name','business_address','country']})
                found,nf,sf=rt.query(q); cs=[]
                for rid,hits in sorted(found.items()):
                    c=rt.get(rid);eid=c['raw']['entity_id']
                    assert eid.startswith(f'S{src}-')
                    xx.append(features(q,c,hits,len(found),nf,sf,src,rt.idf));qq.append(qidx);rr.append(rid);eids.append(eid)
                    cs.append([eid,rid,hits]);derived[eid]=serialize_view(c)
                audit.append({'source1_entity_id':raw['entity_id'],'query_index':qidx,'source':src,'name_frequency':nf,'suffix_frequency':sf,'candidates':cs})
                derived[raw['entity_id']]=serialize_view(q)
            np.savez_compressed(stem.with_suffix('.npz'),X=np.asarray(xx,dtype=np.float32).reshape(-1,len(FEATURES)),q=np.array(qq,dtype=np.int32),rid=np.array(rr,dtype=np.int32),eid=np.array(eids))
            with gzip.open(stem.with_suffix('.candidates.jsonl.gz'),'wt',encoding='utf-8',compresslevel=1) as f:
                for a in audit:f.write(json.dumps(a,ensure_ascii=False)+'\n')
            with gzip.open(stem.with_suffix('.views.jsonl.gz'),'wt',encoding='utf-8',compresslevel=1) as f:
                for eid,v in derived.items():f.write(json.dumps(v,ensure_ascii=False)+'\n')
            save(stem.with_suffix('.done.json'),{'entities':len(audit),'pairs':len(xx),'elapsed_seconds':time.perf_counter()-start,'memory':memory()})
            print('retrieved',src,offset+len(audit),'pairs',len(xx),'seconds',round(time.perf_counter()-start,1),memory(),flush=True)
        rt.con.close();rt.get.cache_clear()
    save(out/f'retrieval_complete_{part}.json',{'seconds_this_invocation':time.perf_counter()-start,'memory':memory(),'features':FEATURES})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=6000);p.add_argument('--smoke-queries',type=int,default=0);p.add_argument('--smoke-index',type=int,default=0)
    p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);a=p.parse_args()
    run(a.n,a.smoke_queries,a.smoke_index,a.part,a.parts)
