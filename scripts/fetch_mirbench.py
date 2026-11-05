"""Obtain the v7 files referenced by official miRBench, with publisher checksums.

Official code: https://github.com/katarinagresova/miRBench/tree/v1.0.3
The official package distributes data through Zenodo record 20540907;
raw datasets are not GitHub source-code blobs. No model weights are downloaded.
"""
import argparse, gzip, hashlib, json, os, shutil, urllib.parse, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def md5(path):
    h=hashlib.md5()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return 'md5:'+h.hexdigest()

def validate_file(path,item):
    if path.stat().st_size!=item['bytes']or md5(path)!=item['publisher_checksum']:
        raise ValueError('Official size/MD5 differs: '+item['filename'])
    if item.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Recorded SHA256 differs: '+item['filename'])
    with gzip.open(path,'rb')as f:
        for _ in iter(lambda:f.read(8<<20),b''):pass

def fetch_one(item,destination,bundled):
    name=item['filename']
    if Path(name).name!=name or '/'in name or '\\'in name or not name.endswith('.tsv.gz'):
        raise ValueError('Unsafe official filename')
    target=destination/name
    if target.exists():
        validate_file(target,item)
        return dict(file=name,status='existing verified')
    partial=target.with_suffix(target.suffix+'.part')
    if partial.exists():raise FileExistsError('Preserve and inspect the existing partial download before retrying')
    source=bundled/name
    if source.exists():
        validate_file(source,item)
        shutil.copyfile(source,partial)
        status='copied included verified file'
    else:
        url=item['url'];u=urllib.parse.urlparse(url)
        if u.scheme!='https'or u.netloc!='zenodo.org'or not u.path.startswith('/api/records/20540907/files/'):
            raise ValueError('Unexpected official dataset URL')
        request=urllib.request.Request(url,headers={'User-Agent':'DuplexFM-reproduction/2026.11.05'})
        with urllib.request.urlopen(request,timeout=60)as response,partial.open('xb')as out:
            final=urllib.parse.urlparse(response.geturl())
            if final.scheme!='https'or final.netloc!='zenodo.org':raise ValueError('Unexpected dataset redirect')
            if 'text/html'in (response.headers.get('Content-Type','')):raise ValueError('Server returned HTML rather than dataset')
            count=0
            while b:=response.read(8<<20):
                count+=len(b)
                if count>item['bytes']:raise ValueError('Official file-size bound exceeded')
                out.write(b)
            out.flush();os.fsync(out.fileno())
        status='downloaded and verified'
    validate_file(partial,item)
    partial.rename(target)
    return dict(file=name,status=status)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--destination',type=Path,default=ROOT/'external_data/mirbench_v7')
    p.add_argument('--file',action='append',dest='filenames',help='Optional official filename; repeat to select several')
    a=p.parse_args();record=json.loads((ROOT/'data/mirbench_v7/SOURCE.json').read_text('utf-8'))
    if not a.download:
        print(json.dumps(record,indent=2));return
    wanted=set(a.filenames or [r['filename']for r in record['files']])
    if not wanted <= {r['filename']for r in record['files']}:p.error('Choose an official filename recorded in SOURCE.json')
    a.destination.mkdir(parents=True,exist_ok=True);done=[]
    for item in record['files']:
        if item['filename']in wanted:
            done.append(fetch_one(item,a.destination,ROOT/'data/mirbench_v7/raw'))
            print(json.dumps(done[-1]),flush=True)
    print(json.dumps(dict(official_repository=record['official_repository'],record_id=20540907,
                         completed=done,weights_downloaded=False),indent=2))

if __name__=='__main__':main()
