"""Fetch only official processed in-vivo icSHAPE data and its GEO SOFT metadata.

No model, training, inference, cohort selection or performance calculation.
Standard library only; immutable final files, bounded transfer, gzip CRC readback.
"""
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent / 'gse74353'
ASSETS = [
    ('GSE74353_family.soft.gz',
     'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74353/soft/GSE74353_family.soft.gz',
     4 * 1024**2),
    ('GSE74353_HS_293T_icSHAPE_InVivo_BaseReactivities.txt.gz',
     'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74353/suppl/GSE74353_HS_293T_icSHAPE_InVivo_BaseReactivities.txt.gz',
     32 * 1024**2),
]

def verify(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(1024**2):
            sha.update(data)
    expanded = 0
    with gzip.open(path, 'rb') as f:
        while data := f.read(1024**2):
            expanded += len(data)
            if expanded > 1024**3:
                raise ValueError('Unexpected expanded size >1GiB')
    return dict(bytes=path.stat().st_size, sha256=sha.hexdigest(),
                gzip_crc_read_to_eof=True, expanded_bytes=expanded)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    receipt = ROOT / 'download_receipt.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        for a in saved['assets']:
            actual = verify(ROOT / a['filename'])
            assert actual['sha256'] == a['sha256']
        print('Existing completed receipt verified; no download.', flush=True)
        return
    started = time.monotonic()
    report = dict(schema='gse74353_official_processed_download_v1',
                  started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  official_origin='https://ftp.ncbi.nlm.nih.gov', assets=[],
                  raw_reads_downloaded=False, model_execution=False)
    try:
        for filename, url, limit in ASSETS:
            path = ROOT / filename
            if path.exists():
                raise FileExistsError(f'Unreceipted existing asset: {path}')
            partial = ROOT / (filename + '.part')
            request = urllib.request.Request(url, headers={'User-Agent':'DuplexFM-public-data-audit/1.0'})
            with urllib.request.urlopen(request, timeout=45) as response:
                if not response.geturl().startswith('https://ftp.ncbi.nlm.nih.gov/'):
                    raise ValueError('Unexpected redirect away from official origin')
                length = response.headers.get('Content-Length')
                if length and int(length) > limit:
                    raise ValueError('Content-Length exceeds bound')
                total = 0
                download_hash = hashlib.sha256()
                with partial.open('xb') as f:
                    while data := response.read(1024**2):
                        total += len(data)
                        if total > limit:
                            raise ValueError('Transfer exceeds bound')
                        download_hash.update(data)
                        f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                if length and total != int(length):
                    raise ValueError('Content-Length mismatch')
                headers = {k: response.headers.get(k) for k in
                           ['Content-Type','Content-Length','Last-Modified','ETag']}
            with partial.open('rb') as f:
                if f.read(2) != b'\x1f\x8b':
                    raise ValueError('Response is not a gzip file')
            checked = verify(partial)
            if checked['sha256'] != download_hash.hexdigest():
                raise ValueError('Independent SHA readback mismatch')
            partial.rename(path)
            report['assets'].append(dict(filename=filename, url=url,
                http_headers=headers, independent_sha_readback=True, **checked))
            print(json.dumps(report['assets'][-1]), flush=True)
        report.update(status='download_verified', elapsed_seconds=time.monotonic()-started,
                      completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        with receipt.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    except Exception as exc:
        report.update(status='failed', error=repr(exc), elapsed_seconds=time.monotonic()-started)
        failure = ROOT / ('download_failure_' + str(time.time_ns()) + '.json')
        failure.write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise

if __name__ == '__main__':
    main()
