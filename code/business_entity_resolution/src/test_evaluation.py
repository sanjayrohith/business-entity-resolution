import unittest
import numpy as np
from core import metrics
from experiment import eval_arrays

class EvaluationTests(unittest.TestCase):
    def test_vectorized_threshold_equals_set_metric(self):
        data=[{'entity_id':'S1-a','truth':[],'fold':'tune'},
              {'entity_id':'S1-b','truth':['S2-b','S3-b'],'fold':'tune'},
              {'entity_id':'S1-c','truth':['S2-c'],'fold':'tune'},
              {'entity_id':'S1-d','truth':[],'fold':'validation'}]
        q=np.array([0,1,1,1,2,3]);e=np.array(['S2-x','S2-b','S3-b','S2-z','S2-w','S2-other'])
        y=np.array([0,1,1,0,0,0]);score=np.array([.4,.9,.8,.3,.2,.99])
        for threshold in [0,.3,.5,.85,1.]:
            expected=metrics({r['entity_id']:set(r['truth']) for r in data if r['fold']=='tune'},
                             {r['entity_id']:{str(e[j]) for j in range(len(e)) if q[j]==i and score[j]>=threshold} for i,r in enumerate(data) if r['fold']=='tune'})
            actual=eval_arrays(data,'tune',q,e,y,score,threshold)
            for key in ['macro_f0_5','macro_precision','macro_recall','false_positive_links','false_negative_links','singleton_accuracy','average_links']:
                self.assertAlmostEqual(expected[key],actual[key])
    def test_empty_candidate_sets(self):
        data=[{'truth':[],'fold':'tune'},{'truth':['S2-a'],'fold':'tune'}]
        result=eval_arrays(data,'tune',np.array([],dtype=int),np.array([]),np.array([]),np.array([]),.6)
        self.assertEqual(result['macro_f0_5'],.5);self.assertEqual(result['false_negative_links'],1)

if __name__=='__main__':unittest.main()
