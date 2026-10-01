import unittest,tempfile
from common import *
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn
from sparse_retrieval import top_merge,augment
from pair_features import extras,FEATURES

class V2Tests(unittest.TestCase):
    def test_sparse_topk_matches_small_dense_reference(self):
        rng=np.random.default_rng(13);a=rng.random((7,13),dtype=np.float32);b=rng.random((13,29),dtype=np.float32)
        a[a<.8]=0;b[b<.8]=0
        c=sp_matmul_topn(sparse.csr_matrix(a),sparse.csr_matrix(b),top_n=4,threshold=0,sort=True)
        dense=a@b
        for i in range(7):
            np.testing.assert_allclose(sorted(c.data[c.indptr[i]:c.indptr[i+1]],reverse=True),sorted(dense[i][dense[i]>0],reverse=True)[:4],atol=1e-6)
    def test_partition_merge(self):
        ids,score=top_merge(np.array([[1,3]]),np.array([[.7,.9]]),np.array([[5,8]]),np.array([[.9,.2]]),2)
        self.assertEqual(ids.tolist(),[[3,5]])
    def test_unicode_kept_with_transliteration(self):
        self.assertIn('कृष्णा',augment('कृष्णा'));self.assertEqual(augment('normal'),'normal')
    def test_features_missing_and_open_country(self):
        a=view(dict(entity_id='S1-a',business_name='École SARL',business_address='12 Rue Lille',country='Unseen'))
        b=view(dict(entity_id='S2-b',business_name='Ecole',business_address='',country='Unseen'))
        self.assertTrue(np.isfinite(extras(a,b)).all());self.assertEqual(len(FEATURES),80)
    def test_new_split_isolation(self):
        if not (V2/'split.json').exists() or not (V1/'split_6000.json').exists():self.skipTest('Run V1/V2 split first (requires challenge data)')
        d=read(V2/'split.json')['records'];old=read(V1/'split_6000.json')['records'];oldids={r['entity_id'] for r in old}
        groups={};targets={};names={};addresses={};oldtargets={e for r in old for e in r['truth']}
        for r in d:
            self.assertNotIn(r['entity_id'],oldids);groups.setdefault(r['group'],set()).add(r['fold']);v=view(r)
            if v['s']:names.setdefault(v['s'],set()).add(r['fold'])
            if v['at']:addresses.setdefault(tuple(sorted(v['at'])),set()).add(r['fold'])
            for e in r['truth']:
                self.assertNotIn(e,oldtargets);targets.setdefault(e,set()).add(r['fold'])
        self.assertTrue(all(len(x)==1 for m in [groups,targets,names,addresses] for x in m.values()))
        self.assertEqual(len(d),10000)
    def test_mature_models_and_reload(self):
        import lightgbm as lgb,xgboost as xgb
        rng=np.random.default_rng(4);x=rng.random((400,3),dtype=np.float32);y=(x[:,0]+x[:,1]>1).astype(int)
        with tempfile.TemporaryDirectory() as d:
            l=lgb.LGBMClassifier(n_estimators=30,num_leaves=7,verbosity=-1,n_jobs=2).fit(x,y)
            p=Path(d)/'l.txt';l.booster_.save_model(str(p));r=lgb.Booster(model_file=str(p))
            np.testing.assert_allclose(l.predict_proba(x)[:,1],r.predict(x,num_threads=2),atol=1e-6)
            m=xgb.XGBClassifier(n_estimators=30,max_depth=3,n_jobs=2).fit(x,y);p=Path(d)/'x.ubj';m.save_model(p)
            r=xgb.XGBClassifier();r.load_model(p);np.testing.assert_allclose(m.predict_proba(x),r.predict_proba(x),atol=1e-6)
            self.assertGreater(((m.predict_proba(x)[:,1]>.5)==y).mean(),.95)
if __name__=='__main__':unittest.main()
