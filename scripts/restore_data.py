"""Restore a declared, SHA-verified multipart dataset (standard library only)."""
import argparse,hashlib,json,os,shutil,tarfile,tempfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def logical(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or str(p)!=name:
        raise ValueError('Unsafe archive path')
    return p

def restore(index_path,destination):
    index_path=Path(index_path).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Use a new output directory; existing data is never overwritten')
    index=json.loads(index_path.read_text(encoding='utf-8'))
    if index['schema']!='duplexfm_multipart_tar_gzip_v1':raise ValueError('Unknown dataset schema')
    parts=index['parts'];records=index['members']
    names=[r['path'] for r in records]
    if len(names)!=len(set(names)) or not names:raise ValueError('Duplicate/empty archive inventory')
    for name in names:logical(name)
    for r in parts:
        logical(r['path']);p=index_path.parent/r['path']
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
            raise ValueError('Part integrity failure: '+r['path'])
    total=sum(r['bytes'] for r in records)
    if total!=index['uncompressed_bytes'] or total>2*2**30:raise ValueError('Expanded-size bound')
    if sum(r['bytes'] for r in parts)!=index['archive_bytes']:raise ValueError('Archive-size bound')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free<total+index['archive_bytes']+2**20:
        raise RuntimeError('Insufficient free disk space')
    with tempfile.TemporaryDirectory(prefix='duplexfm_restore_',dir=destination.parent) as temp:
        archive=Path(temp)/'data.tar.gz'
        with archive.open('xb') as out:
            for r in parts:
                with (index_path.parent/r['path']).open('rb') as f:shutil.copyfileobj(f,out,8<<20)
            out.flush();os.fsync(out.fileno())
        if digest(archive)!=index['archive_sha256']:raise ValueError('Assembled archive SHA mismatch')
        expected={r['path']:r for r in records}
        # Verify the complete tar inventory before writing a single member.
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers()
            if len(members)!=len(expected) or {m.name for m in members}!=set(expected):raise ValueError('Archive inventory mismatch')
            for m in members:
                logical(m.name)
                if not m.isfile() or m.size!=expected[m.name]['bytes']:raise ValueError('Nonregular file or size mismatch')
            stage=Path(temp)/'restored';stage.mkdir()
            for m in members:
                p=stage.joinpath(*logical(m.name).parts);p.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as src,p.open('xb') as out:
                    remaining=m.size
                    while remaining:
                        b=src.read(min(8<<20,remaining))
                        if not b:raise ValueError('Truncated member')
                        out.write(b);remaining-=len(b)
                if digest(p)!=expected[m.name]['sha256']:raise ValueError('Restored member SHA mismatch')
            # Same-filesystem rename publishes only a completely verified directory.
            stage.rename(destination)
    return dict(passed=True,dataset=index['dataset'],files=len(records),bytes=total,all_sha256_verified=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='icshape_source')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.dataset!='icshape_source':p.error('Only the declared icshape_source multipart dataset is supported')
    print(json.dumps(restore(a.root/'data/large'/a.dataset/'index.json',a.output),indent=2))

if __name__=='__main__':main()
