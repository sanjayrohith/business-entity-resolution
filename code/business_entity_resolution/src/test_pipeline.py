import json
import unittest
from collections import defaultdict
from pathlib import Path
import numpy as np
from core import *
from prepare import CACHE
from retrieval import Retriever,features,FEATURES

class PipelineTests(unittest.TestCase):
    def test_split_leakage(self):
        p=CACHE/'split_6000.json'
        if not p.exists():self.skipTest('Run prepare split first')
        d=json.loads(p.read_text(encoding='utf-8'));seen=set();groups=defaultdict(set);names=defaultdict(set);addrs=defaultdict(set)
        for r in d['records']:
            self.assertNotIn(r['entity_id'],seen);seen.add(r['entity_id']);groups[r['group']].add(r['fold']);v=view(r)
            if v['s']:names[v['s']].add(r['fold'])
            if v['at']:addrs[tuple(sorted(v['at']))].add(r['fold'])
        self.assertTrue(all(len(x)==1 for x in list(groups.values())+list(names.values())+list(addrs.values())))
        for fold in ['fit','tune','validation']:
            coverage={(r['country'],min(5,len(r['truth']))) for r in d['records'] if r['fold']==fold}
            self.assertEqual(len(coverage),12)
    def test_candidates_dedup_and_multiple_routes(self):
        if not (CACHE/'s2_smoke100000.json').exists():self.skipTest('Run smoke index first')
        rt=Retriever(2,100000);q=rt.get(1);found,_,_=rt.query(q,k=10,seed_budget=40)
        self.assertEqual(len(found),len(set(found)));self.assertIn(1,found)
        self.assertIn('exact_name',found[1]);self.assertIn('suffix_name',found[1])
        self.assertTrue(all(rt.get(i)['raw']['entity_id'].startswith('S2-') for i in found))
        rt.con.close();rt.get.cache_clear()
    def test_open_country_empty_and_french(self):
        a=dict(entity_id='S1-a',business_name='École de l’Étoile SARL',business_address='12 Rue de la Liberté, Lille',country='Unseen Country')
        b=dict(a,entity_id='S3-b',business_address='')
        q,c=view(a),view(b);f=features(q,c,{},0,0,0,3,{})
        self.assertEqual(len(f),len(FEATURES));self.assertTrue(np.isfinite(f).all());self.assertEqual(f[25],1)
        self.assertEqual(q['raw']['business_name'],a['business_name'])
    def test_model_feature_budget(self):
        self.assertEqual(len(FEATURES),len(set(FEATURES)))

if __name__=='__main__':unittest.main()
