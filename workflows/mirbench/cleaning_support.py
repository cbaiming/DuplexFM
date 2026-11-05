"""Verbatim cleaner support; optional progress reporting does not import Torch."""
from __future__ import annotations
from typing import Optional
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

def normalize_rna(seq: str) -> str:
    if pd.isna(seq):
        return ""
    return str(seq).strip().upper().replace(" ", "").replace("T", "U")


class _NoOpProgress:
    def __init__(self, iterable=None):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, _: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def make_progress(
    iterable=None,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    disable: bool = False,
    leave: bool = False,
    unit: str = "it",
    unit_scale: bool = False,
):
    if disable or tqdm is None:
        return _NoOpProgress(iterable)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        unit=unit,
        unit_scale=unit_scale,
    )
