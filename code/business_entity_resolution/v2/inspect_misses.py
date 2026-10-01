from common import *
import pickle
from sparse_retrieval import augment,ROUTES
records=[r for r in read(V2/'split.json')['records'] if r['fold']=='fit'][:500]
for m in read(V2/'pilot_new_500/assessment.json')['misses']:
    if m['query']['business_name'] not in ['Great Pediatrics','My Products Pvt Ltd','Gold Laxmi Investments Pvt Ltd','Zephis Yorkville LLC']:continue
    src=int(m['candidate']['entity_id'][1]);con=connection(src);idx=V2/f'index_s{src}';q=view(m['query']);c=view(m['candidate']);qi=next(i for i,r in enumerate(records) if r['entity_id']==m['query']['entity_id'])
    print(m['query']['business_name'],'=>',m['candidate']['business_name'],augment(c['nf']))
    for route in ROUTES:
        with open(idx/f'{route}.pkl','rb') as f:v=pickle.load(f)
        field='nf' if route.startswith('name') else 'af';a=v.transform([augment(q[field])]);b=v.transform([augment(c[field])]);sim=float((a@b.T).toarray()[0,0])
        z=np.load(V2/f'pilot_new_500/s{src}_{route}.npz');ids=z['rid'][qi];ss=z['score'][qi]
        names=[con.execute('SELECT business_name,business_address FROM r WHERE rowid=?',(int(i),)).fetchone() for i in ids[:2]]
        print(route,'true',round(sim,4),'cut',round(float(ss[-1]),4),'qnnz',a.nnz,'cnn',b.nnz,'top',names)
    con.close()
