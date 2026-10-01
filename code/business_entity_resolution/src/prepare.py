"""Read-only source scans, entity splits, restartable full-corpus disk indexes."""
import argparse
import ctypes
import heapq
import json
import os
import platform
import sqlite3
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path
from core import *

ROOT=Path(__file__).resolve().parents[3]
CACHE=ROOT/'artifacts'/'v1'

def memory():
    if os.name!='nt': return {}
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in ['peak_rss','rss','pp','p','pnp','np','page','peakpage']]
    p=PMC(); p.cb=ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb)
    if not ok: return {'measurement_unavailable':True}
    return {'peak_rss_bytes':p.peak_rss,'rss_bytes':p.rss}

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def audit():
    import importlib.metadata as md
    files=list((ROOT/'dataset').rglob('*.tsv'))
    data={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files}
    save(CACHE/'data_fingerprints.json',data)
    save(CACHE/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
         'sqlite':sqlite3.sqlite_version,'packages':{d.metadata['Name']:d.version for d in md.distributions()},'memory':memory()})
    print('Fingerprint and environment audit complete',flush=True)

def split(n):
    path=CACHE/f'split_{n}.json'
    if path.exists(): print('Using saved split',path); return
    heap=[]
    for row in rows(ROOT/'dataset/train/train_source1.tsv'):
        score=stable(row['entity_id'])
        item=(-score,row['entity_id'],row)
        if len(heap)<n: heapq.heappush(heap,item)
        elif score < -heap[0][0]: heapq.heapreplace(heap,item)
    chosen={x[1]:x[2] for x in heap}
    truth={r['source1_entity_id']:sorted(parse_ids(r['matched_entity_ids'])) for r in rows(ROOT/'dataset/train/train_ground_truth.tsv') if r['source1_entity_id'] in chosen}
    assert set(truth)==set(chosen)
    parent={e:e for e in chosen}
    def find(e):
        while e!=parent[e]: parent[e]=parent[parent[e]]; e=parent[e]
        return e
    keys={}
    for e,r in sorted(chosen.items()):
        v=view(r)
        # Conservative union by suffix name or reordered address, independent of labels.
        for key in [('name',v['s']),('address',' '.join(sorted(v['at'])))]:
            if not key[1]: continue
            if key in keys: parent[find(e)]=find(keys[key])
            else: keys[key]=e
    groups=defaultdict(list)
    for e in chosen: groups[find(e)].append(e)
    counts={f:Counter() for f in ['fit','tune','validation']}
    assignment={}; targets={'fit':.5,'tune':.25,'validation':.25}
    for g,es in sorted(groups.items(),key=lambda kv:(-len(kv[1]),stable(kv[0]))):
        strata=Counter((chosen[e]['country'],min(5,len(truth[e]))) for e in es)
        f=min(targets,key=lambda f:sum((counts[f][s]+v/2)*v/targets[f] for s,v in strata.items()))
        counts[f].update(strata)
        for e in es: assignment[e]=f
    records=[dict(chosen[e],truth=truth[e],fold=assignment[e],group=find(e)) for e in sorted(chosen)]
    save(path,{'seed':SEED,'n':n,'group_rule':'union(suffix-normalized name,sorted address tokens) within sampled S1',
        'group_count':len(groups),'largest_group':max(map(len,groups.values())),
        'strata':{f:{str(k):v for k,v in c.items()} for f,c in counts.items()},'records':records})
    with open(CACHE/f'split_{n}.tsv','w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['source1_entity_id','fold','group','country','cardinality'])
        w.writerows((r['entity_id'],r['fold'],r['group'],r['country'],len(r['truth'])) for r in records)
    print('Saved split',n, {f:sum(c.values()) for f,c in counts.items()},flush=True)

def build(src,limit):
    start=time.perf_counter(); label=f's{src}'+(f'_smoke{limit}' if limit else '')
    db=CACHE/f'{label}.sqlite'; manifest=CACHE/f'{label}.json'
    input_path=ROOT/f'dataset/train/train_source{src}.tsv'
    fingerprint={'size':input_path.stat().st_size,'sha256':digest(input_path),'normalization':VERSION,'limit':limit}
    if manifest.exists():
        m=json.loads(manifest.read_text(encoding='utf-8'))
        if m['fingerprint']!=fingerprint: raise ValueError('Stale cache fingerprint')
        if m.get('complete'): print('Index already complete:',label,flush=True);return
    else: save(manifest,{'fingerprint':fingerprint,'complete':False})
    con=sqlite3.connect(db)
    con.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-131072; PRAGMA temp_store=FILE;')
    con.executescript("""
      CREATE TABLE IF NOT EXISTS r(eid TEXT, business_name TEXT,business_address TEXT,country TEXT,n TEXT,a TEXT,s TEXT,ak TEXT);
      CREATE VIRTUAL TABLE IF NOT EXISTS lex USING fts5(n,a,content='r',content_rowid='rowid',tokenize='unicode61 remove_diacritics 0');
      CREATE VIRTUAL TABLE IF NOT EXISTS tri USING fts5(n,content='r',content_rowid='rowid',tokenize='trigram');
    """)
    done=con.execute('SELECT count(*) FROM r').fetchone()[0]
    batch=[]; last=done
    for i,r in enumerate(rows(input_path),1):
        if limit and i>limit: break
        if i<=done: continue
        n,a=fold(norm(r['business_name'])),fold(norm(r['business_address']))
        batch.append((i,r['entity_id'],r['business_name'],r['business_address'],r['country'],n,a,suffix(n),' '.join(sorted(set(a.split())))))
        if len(batch)==25000:
            insert(con,batch);last=i;batch=[]
            print(label,i,round(time.perf_counter()-start,1),memory(),flush=True)
    if batch: insert(con,batch);last=batch[-1][0]
    print(label,'Building exact-key indexes',flush=True)
    for field in ['eid','n','s','ak']:
        con.execute(f'CREATE INDEX IF NOT EXISTS idx_{field} ON r({field})');con.commit()
    con.execute("INSERT INTO lex(lex) VALUES('optimize')");con.commit()
    con.execute("INSERT INTO tri(tri) VALUES('optimize')");con.commit()
    con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close()
    save(manifest,dict(fingerprint=fingerprint,complete=True,rows=last,seconds=time.perf_counter()-start,bytes=db.stat().st_size,memory=memory()))
    print(label,'COMPLETE',flush=True)

def insert(con,batch):
    with con:
        con.executemany('INSERT INTO r(rowid,eid,business_name,business_address,country,n,a,s,ak) VALUES(?,?,?,?,?,?,?,?,?)',batch)
        con.executemany('INSERT INTO lex(rowid,n,a) VALUES(?,?,?)',((r[0],r[5],r[6]) for r in batch))
        con.executemany('INSERT INTO tri(rowid,n) VALUES(?,?)',((r[0],r[5]) for r in batch))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','split','index']);p.add_argument('--n',type=int,default=6000)
    p.add_argument('--source',type=int,choices=[2,3],default=2);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    if args.stage=='audit':audit()
    elif args.stage=='split':split(args.n)
    else:build(args.source,args.limit)
