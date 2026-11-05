import unittest
from collections import Counter
from build_balanced import allocate,maximum_coverage,observed_runs,LENGTHS
class Tests(unittest.TestCase):
 def test_coverage_exhaustive(self):
  for n in range(1501):
   counts=Counter({l:(l*17+n)%31 for l in LENGTHS});result=allocate(n,counts)
   possible=max([0]+[min(n,60*k) for k in range(1,n//50+1)])
   self.assertEqual(sum(result),possible);self.assertTrue(all(l in LENGTHS for l in result));self.assertLessEqual(sum(result),n)
 def test_missing_and_true_zero(self):
  seq='A'*171;tokens=['0']*171;tokens[55]='NULL'
  self.assertEqual(list(observed_runs(seq,tokens)),[(0,55),(56,171)])
 def test_ambiguous_gap(self):
  self.assertEqual(list(observed_runs('U'*51+'N'+'G'*60,['.2']*112)),[(0,51),(52,112)])
 def test_deterministic_equal_counts_with_flexible_run(self):
  a=Counter({l:0 for l in LENGTHS});b=a.copy()
  self.assertEqual(allocate(6050,a),allocate(6050,b))
  self.assertLessEqual(max(a.values())-min(a.values()),1)
if __name__=='__main__':unittest.main()
