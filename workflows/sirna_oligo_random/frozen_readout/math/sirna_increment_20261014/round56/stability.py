"""Reuse the locked R51 mathematics; change only the fixed model identities."""
import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parent.parent/'round51/stability.py'
s=importlib.util.spec_from_file_location('_r56_locked51_math',P);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
COMPARISONS=old.COMPARISONS
MODELS=('TD24','Core55','Full55','Initial55','TD55','Core50')
qualification,statistics,paired,interval,macro=old.qualification,old.statistics,old.paired,old.interval,old.macro
