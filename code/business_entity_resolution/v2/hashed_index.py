"""Full-corpus hashed binary TF-IDF: no sample-vocabulary OOV gap."""
from common import *
import argparse,gc,pickle
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_retrieval import ROUTES,augment

def retrieval_text(s,address=False):
    s=augment(s)
    if address:s=re.sub(r'\d+',lambda m:str(int(m[0])),s)
    return s

class HashTfidf:
    def __init__(self,route,idf=None):
        self.route=route;self.idf=idf
        self.vectorizer=HashingVectorizer(n_features=2**20,alternate_sign=False,norm=None,binary=True,
          analyzer='char' if 'char' in route else 'word',ngram_range=(3,5) if route=='name_char' else ((3,3) if route=='address_char' else (1,1)),
          lowercase=False,token_pattern=r'(?u)\b\w+\b',dtype=np.float32)
    def transform(self,texts):
        m=self.vectorizer.transform([retrieval_text(t,self.route.startswith('address')) for t in texts])
        if self.idf is not None:
            m.data*=self.idf[m.indices];m.eliminate_zeros();normalize(m,copy=False)
        return m

def build(src):
    start=time.perf_counter();original=V2/f'index_s{src}';out=V2/f'index2_s{src}';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    raw=out/'binary';raw.mkdir(exist_ok=True)
    vecs={r:HashTfidf(r) for r in ROUTES};dfs={r:np.zeros(2**20,np.int64) for r in ROUTES};n=0
    for p in sorted(original.glob('*.parquet')):
        df=pl.read_parquet(p);n+=len(df)
        for route,v in vecs.items():
            dest=raw/f'{p.stem}_{route}.npz'
            if dest.exists():m=sparse.load_npz(dest)
            else:
                m=v.transform(df['n' if route.startswith('name') else 'a'].to_list());sparse.save_npz(dest,m,compressed=False)
            dfs[route]+=np.bincount(m.indices,minlength=2**20);del m
        del df;gc.collect();print('binary',src,n,round(time.perf_counter()-start,1),flush=True)
    for route,v in vecs.items():
        d=dfs[route];idf=(1+np.log((n+1)/(d+1))).astype(np.float32);idf[d>n*(.03 if 'char' in route else .1)]=0;idf[d==0]=0;v.idf=idf
        with open(out/f'{route}.pkl','wb') as f:pickle.dump(v,f)
        np.save(out/f'{route}_df.npy',d)
        for p in sorted(raw.glob(f'*_{route}.npz')):
            m=sparse.load_npz(p);m.data*=idf[m.indices];m.eliminate_zeros();normalize(m,copy=False)
            sparse.save_npz(out/p.name,m.T.tocsr(),compressed=False);del m;gc.collect()
        print('weighted',src,route,round(time.perf_counter()-start,1),memory(),flush=True)
    old=read(original/'manifest.json')
    save(out/'manifest.json',{'config':{'source':src,'version':'hashed-v2.1','source_fingerprint':old['config']['source_fingerprint'],
       'features':2**20,'char_max_df':.03,'word_max_df':.1,'idf':'full-corpus binary TF','address_digits':'strip leading zeros','normalization':'V1 plus local AnyAscii additional view'},
       'rows':n,'seconds':time.perf_counter()-start,'memory':memory(),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())})
if __name__=='__main__':
    # Ensure pickles have an importable module path when this script is __main__.
    from hashed_index import build
    p=argparse.ArgumentParser();p.add_argument('--source',type=int,required=True);a=p.parse_args();build(a.source)
