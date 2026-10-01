import unittest
import numpy as np
from models import Logistic,HistogramBoost
from retrieval import Retriever

class Tests(unittest.TestCase):
    def test_logistic_and_reload(self):
        x=np.tile(np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32),(100,1));y=x[:,0]
        m=Logistic().fit(x,y,np.ones(len(x),dtype=np.float32),steps=100)
        self.assertTrue(np.all((m.predict(x)>.5)==y))
        np.testing.assert_allclose(m.predict(x),Logistic.load(m.state()).predict(x),atol=1e-7)
    def test_boost_nonlinearity_and_reload(self):
        x=np.tile(np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32),(100,1));y=((x[:,0]+x[:,1])>1).astype(float)
        m=HistogramBoost().fit(x,y,np.ones(len(x)),trees=25)
        self.assertTrue(np.all((m.predict(x)>.5)==y))
        np.testing.assert_allclose(m.predict(x),HistogramBoost.load(m.state()).predict(x),atol=1e-7)
    def test_safe_fts_quotes(self):
        self.assertEqual(Retriever.term('a"b'),'"a""b"')

if __name__=='__main__':unittest.main()
