import unittest,numpy as np
from linearfold_backend import parse
class Tests(unittest.TestCase):
 def test_lengths_and_state(self):
  seqs=['A'*50,'G'*60];raw=('A'*50+'\n'+'.'*50+' (0.0)\n'+'G'*60+'\n'+'('*3+'.'*54+')'*3+' (-1.2)\n').encode()
  a=parse(raw,seqs);self.assertEqual(a.shape,(2,60));self.assertTrue((a[0,:50]==1).all());self.assertTrue((a[0,50:]==0).all());self.assertEqual(a[1].sum(),54)
 def test_reject_bad_output(self):
  for text in ['UU\n.. (0)\n','AA\n)( (0)\n','AA\n. (0)\n','AA\n.. (nan)\n']:
   with self.assertRaises(AssertionError):parse(text.encode(),['AA'])
if __name__=='__main__':unittest.main()
