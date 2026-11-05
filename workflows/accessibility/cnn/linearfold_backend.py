from pathlib import Path
import sys,os,subprocess,json,shutil,re
import numpy as np
from common import ROOT,sha,atomic_json
SOURCE=Path(os.environ.get('DUPLEXFM_LINEARFOLD_ROOT','external_tools/LinearFold-c3ee9bd80c06c2fc39a7bb7ae5e77b9566227cac')).resolve()
LOCAL=SOURCE
EXPECTED='194a299124fd9816c2e267b9f552e922b3ff122247153557a679296e4e6bbd77'
ARGS=['100','0','0','0','0','0','5.0','','0','2']
def prepare():
 assert (LOCAL/'bin/linearfold_v').is_file(), 'Set DUPLEXFM_LINEARFOLD_ROOT to the declared LinearFold source/build'
 assert sha(LOCAL/'bin/linearfold_v')==EXPECTED, 'Historical binary differs; register and validate a new build rather than relabelling it'
 return dict(binary_sha256=EXPECTED,args=ARGS)

def parse(raw,seqs):
 lines=[l.strip() for l in raw.decode('ascii').splitlines() if l.strip()];assert len(lines)==2*len(seqs)
 a=np.zeros((len(seqs),60),dtype='uint8');pattern=re.compile(r'^([().]+)\s+\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)$')
 for i,s in enumerate(seqs):
  assert lines[2*i]==s;match=pattern.fullmatch(lines[2*i+1]);assert match and len(match[1])==len(s)
  assert np.isfinite(float(match[2]));depth=0
  for j,c in enumerate(match[1]):
   depth+=(c=='(')-(c==')');assert depth>=0;a[i,j]=c=='.'
  assert depth==0
 return a
def fold_block(item):
 idx,ss=item;seqs=[bytes(s).decode() for s in ss]
 p=subprocess.run([str(LOCAL/'bin/linearfold_v')]+ARGS,input=('\n'.join(seqs)+'\n').encode(),capture_output=True,timeout=90)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
 return idx,parse(p.stdout,seqs)
