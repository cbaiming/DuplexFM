"""Guards against wrong source endpoints, misalignment and label leakage."""
import copy
import contextlib
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/export_mirna_supervision_control.py'
SPEC = importlib.util.spec_from_file_location('export_mirna_control', PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def checkpoint_fixture(tmp_path):
    models, references = [], {}
    for seed in m.SEEDS:
        for name, state, endpoint, step in (('full', 'trained', 'terminal.pt', 32728),
                                             ('initial', 'initial', 'initial.pt', 0)):
            path = tmp_path / f'{name}{seed}.pt'
            path.write_bytes(f'{name}/{seed}'.encode())
            original = f'{seed:02d}' * 32 if state == 'initial' else 'a' * 64
            references[f'runs/seed{seed}/{endpoint}'] = original
            models.append(dict(id=f'{name}{seed}', path=path.name, seed=seed, step=step,
                               original_sha256=original, sha256=m.sha(path),
                               available=True, tensor_bitwise_restored=True))
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps(dict(models=models)), encoding='utf-8')
    return registry, dict(model_checkpoint_sha256_from_existing_verified_reference=references), models


def test_registry_requires_all_exact_preupdate_and_terminal_identities(tmp_path):
    path, reference, models = checkpoint_fixture(tmp_path)
    sources = m.checkpoint_sources(path, reference)
    assert [(row['state'], row['seed']) for row in sources] == [
        (state, seed) for seed in m.SEEDS for state in ('trained', 'initial')]
    for field, replacement in (('step', 1), ('original_sha256', '0' * 64),
                                ('tensor_bitwise_restored', False), ('available', False), ('seed', 43)):
        changed = copy.deepcopy(models)
        changed[1][field] = replacement
        path.write_text(json.dumps(dict(models=changed)), encoding='utf-8')
        with pytest.raises(ValueError, match='Unverified'):
            m.checkpoint_sources(path, reference)


def test_compact_checkpoint_validates_original_sha_and_actual_step():
    identity = dict(checkpoint_sha256='a' * 64, compact_sha256='b' * 64, step=0, state='initial')
    saved = dict(original_sha256='a' * 64, step=0, trainable_parameters={'x': object()}, fusion_buffers={})
    m.validate_checkpoint(saved, identity)
    for field, replacement in (('step', 32728), ('original_sha256', 'c' * 64), ('trainable_parameters', {})):
        changed = dict(saved, **{field: replacement})
        with pytest.raises(ValueError):
            m.validate_checkpoint(changed, identity)


def test_complete_initial_checkpoint_must_not_have_optimizer_updates():
    identity = dict(checkpoint_sha256='a' * 64, compact_sha256='a' * 64, step=0, state='initial')
    saved = dict(state=dict(step=0, seen_rows=0), optimizer=dict(state={}),
                 trainable_parameters={'x': object()}, fusion_buffers={})
    m.validate_checkpoint(saved, identity)
    saved['optimizer']['state'] = {'x': 1}
    with pytest.raises(ValueError, match='optimizer'):
        m.validate_checkpoint(saved, identity)


class FakeStore:
    def __init__(self, root=None):
        self.guide_ids = np.array([0, 1])
        self.target_ids = np.array([0, 1])
        self.guide_lengths = np.array([19, 19])
        self.target_lengths = np.array([57, 57])
        self.guide_tokens = np.array([[0] * 19, [1] * 19])
        self.target_tokens = np.array([[2] * 57, [3] * 57])
        self.closed = False

    def __len__(self):
        return 2

    def close(self):
        self.closed = True

    def verify_integrity(self):
        return dict(manifest_sha256='a' * 64, files_sha256={'accessibility.npy': 'b' * 64})

    def collate(self, indices):
        return dict(labels=np.array([0.1, 0.9])[indices], row_indices=indices,
                    guide_sequences=['A' * 19 for _ in indices], marker=indices)


def rows_csv(tmp_path, ids=('b', 'a')):
    path = tmp_path / 'rows.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id'])
        writer.writeheader()
        writer.writerows(dict(sample_id=value) for value in ids)
    return path


def test_sequence_alignment_restores_benchmark_order_and_rejects_wrong_sequence(tmp_path):
    store = FakeStore()
    rows = [dict(sample_id='a', guide='C' * 19, target57='U' * 57),
            dict(sample_id='b', guide='A' * 19, target57='G' * 57)]
    path = rows_csv(tmp_path)
    assert m.align_store(store, path, rows) == [('a', 1), ('b', 0)]
    rows[0]['target57'] = 'A' * 57
    with pytest.raises(ValueError, match='sequence differs'):
        m.align_store(store, path, rows)


def test_duplicate_store_ids_and_file_mutation_rejected(tmp_path):
    path = rows_csv(tmp_path, ('a', 'a'))
    with pytest.raises(ValueError, match='Duplicate'):
        m.align_store(FakeStore(), path, [])
    digest = m.sha(path)
    path.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA256 differs'):
        m.require_hash(path, digest)


def test_labels_and_row_indices_never_cross_model_boundary():
    batch = m.feature_batch(FakeStore(), [1, 0])
    assert 'labels' not in batch and 'row_indices' not in batch
    np.testing.assert_array_equal(batch['marker'], [1, 0])


def test_extract_restores_global_order_across_stores_and_removes_labels(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=contextlib.nullcontext))

    class Tensor:
        def __init__(self, values):
            self.values = values

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Model:
        def source192(self, batch, core_start):
            assert core_start == 19 and 'labels' not in batch and 'row_indices' not in batch
            return Tensor(np.repeat(batch['marker'][:, None], 192, axis=1).astype(np.float32))

    aligned = [(1, 1), (0, 0), (1, 0), (0, 1)]
    result = m.extract(Model(), [FakeStore(), FakeStore()], aligned, 1, 'ignored', lambda batch, _: batch)
    np.testing.assert_array_equal(result[:, 0], [1, 0, 0, 1])


def test_open_stores_rejects_duplicate_or_missing_coverage_and_closes_handles(tmp_path):
    (tmp_path / 'manifest.json').write_text('{}', encoding='utf-8')
    path = rows_csv(tmp_path)
    item = dict(root=str(tmp_path), rows_csv=str(path), rows_sha256=m.sha(path),
                manifest_sha256=m.sha(tmp_path / 'manifest.json'))
    rows = [dict(sample_id='a', cohort='one', guide='C' * 19, target57='U' * 57),
            dict(sample_id='b', cohort='one', guide='A' * 19, target57='G' * 57)]
    created = []

    def create_store(root):
        store = FakeStore(root)
        created.append(store)
        return store

    with pytest.raises(ValueError, match='multiple PairStores'):
        m.open_stores([item, item], rows, create_store)
    assert all(store.closed for store in created)
    rows.append(dict(sample_id='missing', cohort='one', guide='A' * 19, target57='G' * 57))
    with pytest.raises(ValueError, match='cover benchmark'):
        m.open_stores([item], rows, create_store)
    assert all(store.closed for store in created)


def test_parity_is_elementwise_and_rejects_bf16_sized_drift():
    reference = np.ones((3, 192), dtype=np.float32)
    actual = reference.copy()
    actual[0, 0] += 5e-5
    assert m.parity_report(actual, reference)['passed']
    actual[0, 0] += 0.001
    with pytest.raises(ValueError, match='fail released-bank parity'):
        m.parity_report(actual, reference)
    with pytest.raises(ValueError, match='only tighten'):
        m.parity_report(reference, reference, atol=0.1)
    with pytest.raises(ValueError, match='float32'):
        m.parity_report(reference.astype(np.float64), reference)


def test_feature_shape_nonfinite_order_digest_and_no_output_overwrite(tmp_path):
    with pytest.raises(ValueError, match='features'):
        m.validate_features(np.zeros((2, 191), np.float32), 2)
    values = np.zeros((2, 192), np.float32)
    values[1, 1] = np.nan
    with pytest.raises(ValueError, match='features'):
        m.validate_features(values, 2)
    assert m.row_ids_sha([{'sample_id': 'a'}, {'sample_id': 'b'}]) != m.row_ids_sha(
        [{'sample_id': 'b'}, {'sample_id': 'a'}])
    with pytest.raises(FileExistsError, match='already exists'):
        m.run(tmp_path / 'missing_config.json', tmp_path, tmp_path)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{"assets":"one","assets":"two"}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        m.read(path)
