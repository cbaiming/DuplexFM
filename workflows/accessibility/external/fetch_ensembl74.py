"""Fetch the declared historical transcript reference, not a genome or model."""
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent/'gse74353'/'reference'
BASE = 'https://ftp.ensembl.org/pub/release-74/fasta/homo_sapiens/cdna/'
NAME = 'Homo_sapiens.GRCh37.74.cdna.all.fa.gz'

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    receipt = ROOT/'download_receipt.json'
    if receipt.exists():
        raise FileExistsError('Reference already received; inspect receipt instead of redownloading')
    report = dict(schema='ensembl74_reference_download_v1',assets=[],
                  started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    started = time.monotonic()
    try:
        for name,limit in [('CHECKSUMS',10000),('README',100000),(NAME,96*1024**2)]:
            path = ROOT/name
            if path.exists():
                raise FileExistsError(path)
            partial = ROOT/(name+'.part')
            url=BASE+name
            sha=hashlib.sha256()
            total=0
            with urllib.request.urlopen(url,timeout=45) as r:
                assert r.geturl().startswith('https://ftp.ensembl.org/')
                size=r.headers.get('Content-Length')
                if size and int(size)>limit:
                    raise ValueError('Reference exceeds transfer cap')
                headers={k:r.headers.get(k) for k in ['Content-Length','Last-Modified','Content-Type']}
                with partial.open('xb') as f:
                    while data:=r.read(1024**2):
                        total+=len(data)
                        if total>limit: raise ValueError('Transfer too large')
                        sha.update(data);f.write(data)
                    f.flush();os.fsync(f.fileno())
                if size: assert total==int(size)
            if name.endswith('.gz'):
                expanded=0
                with gzip.open(partial,'rb') as f:
                    while block:=f.read(1024**2):
                        expanded+=len(block)
                        if expanded>1024**3: raise ValueError('Expanded reference too large')
            else: expanded=None
            with partial.open('rb') as f:
                disk=hashlib.file_digest(f,'sha256').hexdigest()
            assert disk==sha.hexdigest()
            partial.rename(path)
            report['assets'].append(dict(filename=name,url=url,bytes=total,sha256=disk,
                http_headers=headers,gzip_crc_read_to_eof=name.endswith('.gz'),expanded_bytes=expanded))
            print(json.dumps(report['assets'][-1]),flush=True)
        report.update(status='download_verified',wall_seconds=time.monotonic()-started,
                      completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      publisher_checksum_note='CHECKSUMS preserved; independent SHA and gzip CRC checked. Publisher sum validation is separate.')
        with receipt.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
        (ROOT/('failure_'+str(time.time_ns())+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise

if __name__=='__main__': main()
