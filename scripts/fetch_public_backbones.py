#!/usr/bin/env python3
"""Fetch public RNA-FM/mRNABERT files and require historical SHA256 identities.

Online mode requires huggingface_hub. --verify-only uses the standard library.
Repository/prefix overrides allow the same historical bytes to be hosted later
at an author-provided mirror, without changing the model's reference hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'reference/pairac_concat_v1/source_full/code/foundation_reference.json'
MODELS = {'rnafm': ('guide', 'multimolecule/rnafm'), 'mrnabert': ('target', 'YYLY66/mRNABERT')}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_name(value):
    item = PurePosixPath(value)
    if not value or '\\' in value or item.is_absolute() or '..' in item.parts or ':' in value:
        raise ValueError('Expected a safe relative Hugging Face path')
    return item.as_posix()


def checked_copy(source, target, expected):
    source, target = Path(source), Path(target)
    if sha(source) != expected:
        raise ValueError('Historical SHA256 mismatch: ' + target.name +
                         '. Supply a matching historical snapshot or a mirror of the recorded bytes.')
    if target.exists():
        if sha(target) != expected:
            raise ValueError('Existing destination differs; use a new output directory: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.verified-copy.tmp')
    try:
        shutil.copyfile(source, temporary)
        if sha(temporary) != expected:
            raise ValueError('File changed during copy')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def verify(folder, expected):
    checks = []
    for name, digest in expected.items():
        path = Path(folder) / relative_name(name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Missing or mismatched historical foundation file: ' + name)
        checks.append(dict(file=name, bytes=path.stat().st_size, sha256=digest))
    return checks


def fetch(folder, repo, revision, prefix, expected):
    from huggingface_hub import HfApi, hf_hub_download
    prefix = relative_name(prefix.strip('/')) + '/' if prefix else ''
    info = HfApi().model_info(repo_id=repo, revision=revision, files_metadata=True)
    commit = info.sha
    if not commit:
        raise ValueError('Could not resolve an immutable upstream commit')
    for item in info.siblings or []:
        name = item.rfilename.removeprefix(prefix)
        lfs = getattr(item, 'lfs', None)
        advertised = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if name in expected and advertised and advertised != expected[name]:
            raise ValueError('Historical SHA256 mismatch in upstream metadata: ' + name +
                             '. Use a matching historical snapshot or author mirror; large file not downloaded.')
    for name, digest in expected.items():
        source = hf_hub_download(repo_id=repo, revision=commit, filename=prefix + relative_name(name))
        checked_copy(source, Path(folder) / name, digest)
    # Keep upstream notices when available in the selected repository.
    listed = {item.rfilename for item in (info.siblings or [])}
    notices = []
    for name in ('license.md', 'license-faq.md', 'LICENSE', 'LICENSE.md'):
        if prefix + name in listed:
            source = hf_hub_download(repo_id=repo, revision=commit, filename=prefix + name)
            checked_copy(source, Path(folder) / name, sha(source))
            notices.append(name)
    return dict(repo_id=repo, revision=commit, prefix=prefix,
                files=verify(folder, expected), upstream_notices=notices)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Foundation root with rnafm/ and mrnabert/ subdirectories')
    parser.add_argument('--model', nargs='+', choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument('--verify-only', action='store_true')
    for model, (_, repo) in MODELS.items():
        parser.add_argument('--' + model + '-repo', default=repo)
        parser.add_argument('--' + model + '-revision', default='main')
        parser.add_argument('--' + model + '-prefix', default='')
    args = parser.parse_args(argv)
    reference = json.loads(REFERENCE.read_text(encoding='utf-8-sig'))
    results = {}
    for model in dict.fromkeys(args.model):
        kind, _ = MODELS[model]
        expected = reference[kind]['sha256']
        destination = args.output.resolve() / model
        if args.verify_only:
            results[model] = dict(files=verify(destination, expected), model_loaded=False)
        else:
            results[model] = fetch(destination, getattr(args, model + '_repo'),
                                   getattr(args, model + '_revision'),
                                   getattr(args, model + '_prefix'), expected)
    report = dict(passed=True, historical_reference_sha256=sha(REFERENCE),
                  models=results, model_execution=False)
    if not args.verify_only:
        (args.output / 'public_backbones_receipt.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
