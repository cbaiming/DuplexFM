"""Never install mismatched foundations or overwrite a user's existing files."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('fetch_backbones', Path(__file__).resolve().parents[1]/'scripts/fetch_public_backbones.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_mismatch_rejected_before_install_and_existing_file_preserved(tmp_path):
    source, target = tmp_path/'download', tmp_path/'assets/config.json'
    source.write_bytes(b'upstream current revision')
    with pytest.raises(ValueError, match='Historical SHA256 mismatch'):
        m.checked_copy(source, target, hashlib.sha256(b'historical').hexdigest())
    assert not target.exists()
    digest = m.sha(source)
    m.checked_copy(source, target, digest)
    assert m.verify(target.parent, {'config.json':digest})[0]['sha256'] == digest
    target.write_bytes(b'user existing data')
    with pytest.raises(ValueError, match='Existing destination differs'):
        m.checked_copy(source, target, digest)
    assert target.read_bytes() == b'user existing data'


@pytest.mark.parametrize('name', ['../secret', '/absolute', 'a/../../escape', 'drive:file', 'a\\b'])
def test_remote_filename_cannot_escape_output(name):
    with pytest.raises(ValueError, match='safe relative'):
        m.relative_name(name)


def test_advertised_weight_mismatch_stops_before_large_download(tmp_path, monkeypatch):
    info = SimpleNamespace(sha='a' * 40, siblings=[
        SimpleNamespace(rfilename='model.safetensors', lfs={'sha256':'0' * 64})])
    def download(**kwargs):
        raise AssertionError('A mismatched large weight must not be downloaded')
    fake = SimpleNamespace(HfApi=lambda:SimpleNamespace(model_info=lambda **kwargs:info),
                           hf_hub_download=download)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake)
    with pytest.raises(ValueError, match='upstream metadata'):
        m.fetch(tmp_path/'model', 'publisher/model', 'main', '', {'model.safetensors':'1' * 64})
    assert not (tmp_path/'model').exists()
