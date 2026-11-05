"""Integration contract tests; synthetic rows only except fixed split metadata."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import readout as r

class ReadoutContract(unittest.TestCase):
    def test_source_scaling_matches_original_without_pca(self):
        x=np.random.default_rng(42).normal(size=(63,216));x[:,24]=3.;x[:,25]*=1e-10
        actual=r.fit_scaler(x);_,old=r.scale.fit_transform(x)
        for key in actual:np.testing.assert_array_equal(actual[key],old[key])
        z=r.head.transform(x,actual);np.testing.assert_allclose(z.mean(0),0,atol=1e-12)
        self.assertEqual(actual['src_scale'][0],1.)
        self.assertGreater(actual['src_scale'][1],1e-5)

    def test_deployment_ignores_groups_and_certifies(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(55,216));y=.4+.1*x[:,0]+.02*rng.normal(size=55)
        g=np.array([0]*30+[1]*15+[2]*10);t=r.fit_scaler(x);z=r.head.transform(x,t)
        b,c,cert=r.head.fit(z,y,g,10.,True);self.assertTrue(cert['certified'])
        r.math_audit.check_source(z,y,g,b,c,cert,10.)
        coef,offset=r.head.deployment(t,b,c)
        np.testing.assert_allclose(x@coef+offset,z@b+c,atol=1e-10,rtol=1e-10)

    def test_original_alpha_tie_breaks(self):
        self.assertEqual(r.head.choose([.5]*5,[1]*5),4)
        self.assertEqual(r.head.choose([.5]*5,[1,1,.9,1,1]),2)
        self.assertEqual(r.head.choose([.5,.6,.5,.5,.5],[1,.9,.1,.1,.1]),1)

    def test_fixed_counts_and_scope(self):
        plan=r.read(r.R55/'plan.json')
        self.assertEqual(len(plan['folds']),16)
        self.assertEqual(set(i['setting'] for i in plan['folds']),set(r.SETTINGS))
        self.assertEqual(len(r.VARIANTS),6)
        self.assertEqual(sum(len(i['inner'])*5 for i in plan['folds'])*len(r.VARIANTS),2400)
        self.assertEqual(sum(len(i['test_ids']) for i in plan['folds'])*len(r.VARIANTS),22788)
        self.assertEqual({int(s[-2:]) for s in r.VARIANTS},{43,44})
        for item in plan['folds']:
            self.assertFalse(set(item['fit_ids'])&set(item['test_ids']))
            coverage=np.zeros(len(item['fit_ids']),int)
            for split in item['inner']:
                self.assertFalse(set(split['train'])&set(split['valid']));coverage[split['valid']]+=1
            self.assertTrue(np.all(coverage==1))

if __name__=='__main__':unittest.main()
