"""Released inference state is accepted only by its registered bytes and seed."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reference/pairac_concat_v1/source_full/code'))
SPEC = importlib.util.spec_from_file_location('compact_source_inference', ROOT/'scripts/reproduce_mirna.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture():
    item=m.read(ROOT/'MODEL_REGISTRY.json')['compact_inference_checkpoints']['42']
    checkpoint=dict(step=32728,original_sha256=item['original_sha256'],
                    trainable_parameters={str(i):None for i in range(132)},fusion_buffers={})
    return item,checkpoint,SimpleNamespace(seed=42,assets=Path('/unused'))


def test_registered_compact_accepts_only_matching_source_identity():
    item,checkpoint,args=fixture()
    result=m.validate_checkpoint_contract(checkpoint,args,item['sha256'])
    assert result['registered_compact_identity'] and not result['training_resume_supported']
    args.seed=43
    with pytest.raises(ValueError,match='different source seed'):
        m.validate_checkpoint_contract(checkpoint,args,item['sha256'])


@pytest.mark.parametrize('field,value',[('step',32727),('original_sha256','0'*64),('trainable_parameters',{})])
def test_registered_compact_rejects_bad_endpoint_origin_or_inventory(field,value):
    item,checkpoint,args=fixture()
    checkpoint[field]=value
    with pytest.raises(ValueError,match='metadata or tensor inventory'):
        m.validate_checkpoint_contract(checkpoint,args,item['sha256'])
