from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cnn'))
METHODS=['PairAC_original','rebuilt_mRNABERT','sequence','structure','LinearFold_V_raw']
DATA_SHA='1c6157293616224ecdacd9cfbf76387ac65f67243ff412fde251dff29a961719'
