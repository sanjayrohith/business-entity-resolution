"""Bounded name-only rescues on cross-script and missing-address target subsets."""
from common import *
import argparse,pickle,gc
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn
from anyascii import anyascii

class TranslitVectorizer:
    def __init__(self):
        self.v=HashingVectorizer(analyzer='char',ngram_range=(2,4),n_features=2**18,alternate_sign=False,binary=True,norm=None,lowercase=False,dtype=np.float32);self.idf=None
    def transform(self,texts):
        m=self.v.transform([anyascii(t).lower() for t in texts])
        if self.idf is not None:m.data*=self.idf[m.indices];m.eliminate_zeros();normalize(m,copy=False)
        return m

def build(src):
    out=V2/f'targeted_s{src}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    t=time.perf_counter();v=TranslitVectorizer();mats=[];ids=[];missing=[]
    for p in sorted((V2/f'index_s{src}').glob('*.parquet')):
        df=pl.read_parquet(p,columns=['rid','n','a']);sub=df.filter(pl.col('n').str.contains(r'[^\x00-\x7f]'))
        mats.append(v.transform(sub['n'].to_list()));ids.extend(sub['rid'].to_list());missing.extend(df.filter(pl.col('a')=='')['rid'].to_list())
    m=sparse.vstack(mats,format='csr');del mats;gc.collect();df=np.bincount(m.indices,minlength=2**18)
    v.idf=(1+np.log((len(ids)+1)/(df+1))).astype(np.float32);v.idf[df>len(ids)*.2]=0;v.idf[df==0]=0
    m.data*=v.idf[m.indices];m.eliminate_zeros();normalize(m,copy=False)
    sparse.save_npz(out/'translit.npz',m.T.tocsr(),compressed=False);np.save(out/'translit_ids.npy',np.asarray(ids,np.int32));np.save(out/'missing_ids.npy',np.asarray(missing,np.int32))
    with open(out/'translit.pkl','wb') as f:pickle.dump(v,f)
    save(out/'complete.json',{'seconds':time.perf_counter()-t,'nonlatin_records':len(ids),'missing_records':len(missing),'memory':memory()})
    print('targeted index',src,len(ids),time.perf_counter()-t,flush=True)

def retrieve(records,src,index,k=50):
    from sparse_retrieval import top_merge
    out=V2/f'targeted_s{src}';start=time.perf_counter();n=len(records);results={}
    with open(out/'translit.pkl','rb') as f:v=pickle.load(f)
    q=v.transform([fold(norm(r['business_name'])) for r in records]);m=sparse.load_npz(out/'translit.npz');mapping=np.load(out/'translit_ids.npy')
    c=sp_matmul_topn(q,m,top_n=k,threshold=.12,sort=True,n_threads=4);ids=np.zeros((n,k),np.int32);scores=np.zeros((n,k),np.float32)
    for i in range(n):
        lo,hi=c.indptr[i:i+2];ln=hi-lo;ids[i,:ln]=mapping[c.indices[lo:hi]];scores[i,:ln]=c.data[lo:hi]
    results['translit_char']=(ids,scores);del m,c
    with open(index/'name_char.pkl','rb') as f:v=pickle.load(f)
    q=v.transform([fold(norm(r['business_name'])) for r in records]);missing=np.load(out/'missing_ids.npy')
    ids=np.empty((n,0),np.int32);scores=np.empty((n,0),np.float32)
    for p in sorted(index.glob('*_name_char.npz')):
        offset=int(p.name.split('_')[0]);m=sparse.load_npz(p);mapping=missing[(missing>offset)&(missing<=offset+m.shape[1])]
        m=m[:,mapping-offset-1].tocsr();c=sp_matmul_topn(q,m,top_n=k,threshold=.1,sort=True,n_threads=4)
        ni=np.zeros((n,k),np.int32);ns=np.zeros((n,k),np.float32)
        for i in range(n):
            lo,hi=c.indptr[i:i+2];ln=hi-lo;ni[i,:ln]=mapping[c.indices[lo:hi]];ns[i,:ln]=c.data[lo:hi]
        ids,scores=top_merge(ids,scores,ni,ns,k);del m,c;gc.collect()
    results['missing_name_char']=(ids,scores)
    # Joint evidence retrieves common-name records whose two individual ranks are weak.
    with open(index/'address_word.pkl','rb') as f:av=pickle.load(f)
    aq=av.transform([fold(norm(r['business_address'])) for r in records])
    joint=sparse.hstack([q*.5,aq*.5],format='csr');ids=np.empty((n,0),np.int32);scores=np.empty((n,0),np.float32)
    for p in sorted(index.glob('*_name_char.npz')):
        offset=int(p.name.split('_')[0]);nm=sparse.load_npz(p);am=sparse.load_npz(index/f'{offset:08d}_address_word.npz');m=sparse.vstack([nm,am],format='csr');del nm,am
        c=sp_matmul_topn(joint,m,top_n=k,threshold=.08,sort=True,n_threads=4);ni=np.zeros((n,k),np.int32);ns=np.zeros((n,k),np.float32)
        for i in range(n):
            lo,hi=c.indptr[i:i+2];ln=hi-lo;ni[i,:ln]=c.indices[lo:hi]+offset+1;ns[i,:ln]=c.data[lo:hi]
        ids,scores=top_merge(ids,scores,ni,ns,k);del m,c;gc.collect()
    results['joint_name_address']=(ids,scores)
    return results,time.perf_counter()-start
if __name__=='__main__':
    from targeted import build
    p=argparse.ArgumentParser();p.add_argument('--source',type=int,required=True);a=p.parse_args();build(a.source)
