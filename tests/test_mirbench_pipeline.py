"""Preparation/source-wrapper contracts, with no encoder loading or training."""
import ast
import hashlib
import importlib.util
import json
import argparse
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

prepare = module('public_prepare_mirbench', ROOT / 'scripts/prepare_mirbench.py')
source = module('public_reproduce_mirna', ROOT / 'scripts/reproduce_mirna.py')


class MirbenchPipelineTests(unittest.TestCase):
    def test_extracted_cleaner_functions_match_recorded_original_source(self):
        provenance = prepare.read(ROOT / 'workflows/mirbench/provenance.json')
        for filename, field in [('cleaning.py', 'functions_utf8_lf_sha256'),
                                ('cleaning_support.py', 'support_definitions_utf8_lf_sha256')]:
            text = (ROOT / 'workflows/mirbench' / filename).read_text(encoding='utf-8')
            lines = text.splitlines(keepends=True)
            found = {}
            for node in ast.parse(text).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    code = ''.join(lines[node.lineno - 1:node.end_lineno])
                    found[node.name] = hashlib.sha256(code.encode()).hexdigest()
            self.assertEqual(found, provenance[field])
        contract = prepare.read(ROOT / 'data/mirbench_v7/preparation_contract.json')
        self.assertEqual(prepare.sha(ROOT / 'data/mirbench_v7/fixed_split_manifest.json'), contract['fixed_split_sha256'])

    def test_first_occurrence_mapping_fixed_family_split_and_no_physics(self):
        def frame(rows):
            return pd.DataFrame(rows, columns=['gene', 'noncodingRNA', 'noncodingRNA_fam', 'label', 'source_dataset'])
        order = ['training', 'testA', 'testB']
        frames = {'training': frame([('C' * 50, 'ACGU', 'famA', 1, 'train.gz'),
                                     ('A' * 50, 'UGCA', 'famB', 0, 'train.gz')]),
                  'testA': frame([('C' * 50, 'UGCA', 'famB', 1, 'testA.gz')]),
                  'testB': frame([('G' * 50, 'ACGU', 'famA', 0, 'testB.gz')])}
        split = {'group_assignments': [{'noncodingRNA_fam': 'famA', 'split': 'train'},
                                       {'noncodingRNA_fam': 'famB', 'split': 'val'}]}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = prepare.build_arrays(frames, order, split, output)
            np.testing.assert_array_equal(np.load(output / 'data/target_ids.npy'), [0, 1, 0, 2])
            np.testing.assert_array_equal(np.load(output / 'data/guide_ids.npy'), [0, 1, 1, 0])
            np.testing.assert_array_equal(np.load(output / 'data/indices_train.npy'), [0])
            np.testing.assert_array_equal(np.load(output / 'data/indices_val.npy'), [1])
            np.testing.assert_array_equal(np.load(output / 'data/indices_testA.npy'), [2])
            self.assertEqual(np.load(output / 'data/target_tokens.npy').dtype, np.uint8)
            self.assertEqual(np.load(output / 'data/guide_lengths.npy').dtype, np.int16)
            self.assertEqual(np.load(output / 'data/target_ids.npy').dtype, np.int32)
            self.assertEqual(np.load(output / 'data/labels.npy').dtype, np.float32)
            self.assertEqual(result['rows'], 4)
            self.assertFalse((output / 'physics').exists())
            self.assertFalse((output / 'manifest.json').exists())
            expected = {'files_sha256': dict(result['files_sha256'])}
            prepare.verify_arrays(output, expected)
            (output / 'data/labels.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'array SHA-256 differs'):
                prepare.verify_arrays(output, expected)

    def test_missing_family_assignment_and_noncanonical_real_base_fail(self):
        with self.assertRaisesRegex(ValueError, 'Noncanonical'):
            prepare.encode(['ACGN'], 4)
        frame = pd.DataFrame([('A' * 50, 'ACGU', 'unassigned', 1, 'train.gz')],
                             columns=['gene', 'noncodingRNA', 'noncodingRNA_fam', 'label', 'source_dataset'])
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'family missing'):
            prepare.build_arrays({'train': frame}, ['train'], {'group_assignments': []}, Path(folder))

    def test_tokens_only_must_be_explicit_and_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            existing = Path(folder)
            with self.assertRaises(FileExistsError):
                prepare.prepare(existing, existing, tokens_only=True)
            with self.assertRaisesRegex(ValueError, 'explicitly choose --tokens-only'):
                prepare.prepare(existing, existing / 'new')

    def test_one_pinned_official_test_cleaning_without_source_training(self):
        path = ROOT / 'data/mirbench_v7/raw/AGO2_CLASH_Hejret2023_test.tsv.gz'
        contract = prepare.read(ROOT / 'data/mirbench_v7/preparation_contract.json')
        frame, audit = prepare.load_frame(path, 'AGO2_CLASH_Hejret2023_test', contract['files']['AGO2_CLASH_Hejret2023_test'])
        self.assertEqual(len(frame), 1018)
        self.assertEqual(int(frame.label.sum()), 495)
        self.assertEqual(audit['duplicate_row_count'], 0)
        values = prepare.encode(pd.unique(frame.gene), 50)
        self.assertEqual(values.shape[1], 50)
        self.assertTrue(np.isin(values, range(4)).all())
        # Independent known global test offset verifies the archived row-index bytes.
        buffer = io.BytesIO()
        np.save(buffer, np.arange(2495535 + 954, 2495535 + 954 + 1018), allow_pickle=False)
        historical = prepare.read(ROOT / 'data/mirbench_v7/historical_inputs.json')
        self.assertEqual(hashlib.sha256(buffer.getvalue()).hexdigest(),
                         historical['files_sha256']['data/indices_AGO2_CLASH_Hejret2023_test.npy'])

    def test_source_checkpoint_hash_rejected_before_encoder_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'wrong.pt'
            checkpoint.write_bytes(b'not the registered checkpoint')
            args = argparse.Namespace(device='cpu', amp='fp32', checkpoint=checkpoint,
                                      checkpoint_sha256=None, seed=42, assets=Path(folder))
            with patch('duplexfm_mechanism.model.PairACModel') as constructor:
                with self.assertRaisesRegex(ValueError, 'Checkpoint SHA-256 differs'):
                    source.load_model(args)
                constructor.assert_not_called()

    def test_new_checkpoint_seed_and_protocol_rejected_before_encoder_construction(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'new.pt'
            args = argparse.Namespace(device='cpu', amp='fp32', checkpoint=checkpoint,
                                      checkpoint_sha256=None, seed=42, assets=Path(folder))
            with patch('duplexfm_mechanism.model.PairACModel') as constructor:
                wrong_seed = {'state': {'step': 32728}, 'signature': {
                    'inputs': {'version': 'portable_PairAC_source_execution_v1', 'source_seed': 43},
                    'options': {'seed': 43}}}
                torch.save(wrong_seed, checkpoint)
                args.checkpoint_sha256 = source.sha(checkpoint)
                with self.assertRaisesRegex(ValueError, 'source seed differs'):
                    source.load_model(args)
                changed_protocol = {'state': {'step': 32728}, 'signature': {
                    'inputs': {'version': 'portable_PairAC_source_execution_v1', 'source_seed': 42,
                               'original_contract': {'reference_epochs': 99}},
                    'options': {'seed': 42}}}
                torch.save(changed_protocol, checkpoint)
                args.checkpoint_sha256 = source.sha(checkpoint)
                with self.assertRaisesRegex(ValueError, 'source protocol differs'):
                    source.load_model(args)
                unsigned = {'state': {'step': 32728}}
                torch.save(unsigned, checkpoint)
                args.checkpoint_sha256 = source.sha(checkpoint)
                with self.assertRaisesRegex(ValueError, 'missing portable source execution'):
                    source.load_model(args)
                constructor.assert_not_called()

    def test_new_execution_contract_accepts_own_complete_signature(self):
        from duplexfm_mechanism.protocol import contract, source_config
        args = argparse.Namespace(seed=42, assets=Path('/explicit/models'))
        spec = contract()
        foundation = source.read(source.CODE / 'foundation_reference.json')
        foundation['foundation_reference_verified'] = True
        inputs = {'version': 'portable_PairAC_source_execution_v1', 'source_seed': 42,
                  'original_contract': spec,
                  'model_config': {key: value for key, value in source_config(args.assets).items() if not key.endswith('_model')},
                  'input_arrays_sha256': source.read(ROOT / 'data/mirbench_v7/historical_inputs.json')['files_sha256'],
                  'source_code_sha256': {path.relative_to(source.CODE).as_posix(): source.sha(path) for path in source.CODE.rglob('*.py')},
                  'foundation': foundation}
        options = {'seed': 42, 'reference_rows': spec['reference_train_rows'], 'batch_size': spec['batch'],
                   'eval_batch_size': spec['eval_batch'], 'max_updates': spec['updates'],
                   'learning_rate': spec['learning_rate'], 'lora_learning_rate': spec['lora_learning_rate'],
                   'weight_decay': spec['weight_decay'], 'clip_grad_norm': spec['clip_grad_norm'],
                   'checkpoint_steps': 500, 'backup_seconds': 3600, 'amp': 'bf16', 'micro_batch_size': 512}
        saved = {'state': {'step': 32728, 'seen_rows': 8 * spec['reference_train_rows'],
                           'last_validation_step': 32728,
                           'history': [{'reference_epoch': index, 'step': index * (32728 // 8)} for index in range(1, 9)]},
                 'signature': {'inputs': inputs, 'options': options},
                 'scientific_checkpoint_selection': 'fixed_terminal_not_best_validation'}
        self.assertTrue(source.validate_checkpoint_contract(saved, args, '0' * 64)['new_execution_contract_verified'])
        saved['signature']['options']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'training options differ'):
            source.validate_checkpoint_contract(saved, args, '0' * 64)
        saved['signature']['options']['learning_rate'] /= 2
        saved['state']['history'][-1]['reference_epoch'] = 99
        with self.assertRaisesRegex(ValueError, 'terminal epoch/validation cursor differs'):
            source.validate_checkpoint_contract(saved, args, '0' * 64)

    def test_registered_historical_checkpoint_cannot_be_relabelled_to_another_seed(self):
        contract = source.read(ROOT / 'data/mirbench_v7/preparation_contract.json')
        args = argparse.Namespace(seed=42)
        with self.assertRaisesRegex(ValueError, 'different source seed'):
            source.validate_checkpoint_contract({'state': {'step': 32728}}, args,
                contract['source_terminal_sha256']['43'])

    def test_source_inference_rejects_unsafe_partition_before_model_load(self):
        for name in ['../train', 'x/y', 'x\\y', 'C:train', '.', '']:
            with self.assertRaises(ValueError):
                source.safe_partition(name)
        self.assertEqual(source.safe_partition('AGO2_CLASH_Hejret2023_test'), 'AGO2_CLASH_Hejret2023_test')


if __name__ == '__main__':
    unittest.main()
