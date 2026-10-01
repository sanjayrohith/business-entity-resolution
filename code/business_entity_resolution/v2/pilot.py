"""Retrieval policy assessment on fitting data, excluding both holdouts."""
from common import *
import gzip,argparse
from collections import Counter
from sparse_retrieval import retrieve
from experiment import slices

def assess(records,out):
    totals=Counter();hits=Counter();counts=np.zeros(len(records),int);miss=[];budgethits={k:Counter() for k in [10,20,50,100]};budgetcounts={k:np.zeros(len(records),int) for k in budgethits}
    for src in [2,3]:
        con=connection(src);truth={}
        for i,r in enumerate(records):
            q=view(r);truth[i]={}
            for e in r['truth']:
                if not e.startswith(f'S{src}-'):continue
                row=con.execute('SELECT rowid,eid,business_name,business_address,country FROM r WHERE eid=?',(e,)).fetchone()
                raw=dict(zip(['entity_id','business_name','business_address','country'],row[1:]));c=view(raw)
                truth[i][int(row[0])]=(e,slices(q,c),raw)
                totals.update(slices(q,c))
        with gzip.open(out/f's{src}_candidates.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                b=json.loads(line);i=b['q'];cands=set(map(int,b['candidates']));counts[i]+=len(cands)
                for k in budgethits:
                    keep={int(rid) for rid,rs in b['candidates'].items() if any(v[0]<=k for v in rs.values())}
                    budgetcounts[k][i]+=len(keep)
                    for rid,(eid,ss,raw) in truth[i].items():
                        if rid in keep:budgethits[k].update(ss)
                for rid,(eid,ss,raw) in truth[i].items():
                    if rid in cands:hits.update(ss)
                    else:miss.append({'query':records[i],'candidate':raw,'slices':ss})
        con.close()
    report={'recall':{s:{'total':v,'hit':hits[s],'recall':hits[s]/v} for s,v in totals.items()},
     'counts':{'mean':float(counts.mean()),'p99':float(np.quantile(counts,.99)),'pairs':int(counts.sum()),'max':int(counts.max())},'misses':miss,
     'budgets':{str(k):{'recall':{s:budgethits[k][s]/v for s,v in totals.items()},'mean':float(budgetcounts[k].mean()),'p99':float(np.quantile(budgetcounts[k],.99))} for k in budgethits}}
    save(out/'assessment.json',report);print(json.dumps({k:v for k,v in report.items() if k!='misses'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',action='store_true');p.add_argument('--n',type=int,default=500);p.add_argument('--revision',default='');p.add_argument('--policy',choices=['all','tiered'],default='all');a=p.parse_args()
    records=read(V1/'split_6000.json' if a.old else V2/'split.json')['records'];records=[r for r in records if r['fold']=='fit'][:a.n]
    out=V2/((f'pilot_old_{a.n}' if a.old else f'pilot_new_{a.n}')+a.revision)
    for src in [2,3]:retrieve(records,src,out,policy=a.policy)
    assess(records,out)
