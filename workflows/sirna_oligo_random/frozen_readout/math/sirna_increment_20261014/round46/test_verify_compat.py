from pathlib import Path
import numpy as np
import pytest
from verify_compat import patched,OLD,NEW


def test_one_dimensional_intercept():
    state={'a':{'inner_intercept':np.arange(5.)}}
    exec(patched(OLD),state)
    np.testing.assert_array_equal(state['tdintercept'],np.arange(5.))
    for shape in ((5,25),(4,)):
        with pytest.raises(AssertionError):exec(patched(OLD),{'a':{'inner_intercept':np.zeros(shape)}})


def test_exact_replacement_and_compile_only():
    original=(Path(__file__).resolve().parent/'verify.py').read_text(encoding='utf-8')
    result=patched(original)
    assert result.replace(NEW,OLD)==original
    assert len([i for i,(a,b) in enumerate(zip(original.splitlines(),result.splitlines())) if a!=b])==1
    compile(result,'not_executed.py','exec')


def test_wrong_source_rejected():
    with pytest.raises(ValueError):patched('no site')
    with pytest.raises(ValueError):patched(OLD+';'+OLD)
