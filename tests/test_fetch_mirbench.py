import gzip,hashlib,importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('fetch_official',Path(__file__).resolve().parents[1]/'scripts/fetch_mirbench.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(tmp_path):
    bundled=tmp_path/'bundled';dest=tmp_path/'dest';bundled.mkdir();dest.mkdir()
    payload=gzip.compress(b'gene\tnoncodingRNA\tlabel\nAAAA\tUUUU\t1\n',mtime=0)
    item=dict(filename='fake.tsv.gz',bytes=len(payload),publisher_checksum='md5:'+hashlib.md5(payload).hexdigest(),
              url='https://zenodo.org/api/records/20540907/files/fake.tsv.gz/content')
    return bundled,dest,payload,item

def test_existing_verified_source_is_copied_into_one_destination(tmp_path):
    b,d,p,item=fixture(tmp_path);(b/item['filename']).write_bytes(p)
    assert m.fetch_one(item,d,b)['status']=='copied included verified file'
    assert (d/item['filename']).read_bytes()==p
    assert m.fetch_one(item,d,b)['status']=='existing verified'

def test_wrong_existing_file_is_not_overwritten(tmp_path):
    b,d,p,item=fixture(tmp_path);(d/item['filename']).write_bytes(b'wrong')
    with pytest.raises(ValueError):m.fetch_one(item,d,b)
    assert (d/item['filename']).read_bytes()==b'wrong'

def test_partial_is_preserved_not_silently_replaced(tmp_path):
    b,d,p,item=fixture(tmp_path);q=d/'fake.tsv.gz.part';q.write_bytes(b'partial')
    with pytest.raises(FileExistsError):m.fetch_one(item,d,b)
    assert q.read_bytes()==b'partial'

def test_untrusted_url_and_path_are_rejected_without_network(tmp_path):
    b,d,p,item=fixture(tmp_path);item['url']='https://untrusted.invalid/file'
    with pytest.raises(ValueError,match='Unexpected'):m.fetch_one(item,d,b)
    item['filename']='../escape.tsv.gz'
    with pytest.raises(ValueError,match='Unsafe'):m.fetch_one(item,d,b)
