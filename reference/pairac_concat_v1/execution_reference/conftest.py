"""Review-only import setup, also supports embedded Python ignoring PYTHONPATH."""
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(root/'ablations/seeds43_44/code')]
import torch
torch.set_num_threads(1)
