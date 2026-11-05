"""Synthetic archive safety and integrity tests; no real data or model training."""
import hashlib,importlib.util,io,json,tarfile,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('restore_public_data',ROOT/'scripts/restore_data.py')
restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
def sha(b):return hashlib.sha256(b).hexdigest()
class TestRestore(unittest.TestCase):
    def fixture(self,root,name='data/a.txt'):
        raw=io.BytesIO()
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            info=tarfile.TarInfo(name);info.size=3;tar.addfile(info,io.BytesIO(b'abc'))
        archive=raw.getvalue();parts=[]
        for i,b in enumerate([archive[:11],archive[11:]]):
            p=root/f'archive.part{i:04d}';p.write_bytes(b);parts.append(dict(path=p.name,bytes=len(b),sha256=sha(b)))
        index=dict(schema='duplexfm_multipart_tar_gzip_v1',dataset='fake',parts=parts,
                   members=[dict(path=name,bytes=3,sha256=sha(b'abc'))],archive_sha256=sha(archive),archive_bytes=len(archive),uncompressed_bytes=3)
        p=root/'index.json';p.write_text(json.dumps(index));return p
    def test_roundtrip_and_existing_output(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);i=self.fixture(r);out=r/'out'
            self.assertTrue(restore.restore(i,out)['passed']);self.assertEqual((out/'data/a.txt').read_bytes(),b'abc')
            with self.assertRaises(FileExistsError):restore.restore(i,out)
    def test_tamper_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);i=self.fixture(r);(r/'archive.part0000').write_bytes(b'bad')
            with self.assertRaises(ValueError):restore.restore(i,r/'out')
            self.assertFalse((r/'out').exists())
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);i=self.fixture(r,'../escape.txt')
            with self.assertRaises(ValueError):restore.restore(i,r/'out')
            self.assertFalse((r/'escape.txt').exists())
    def test_declared_member_hash_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);i=self.fixture(r);d=json.loads(i.read_text());d['members'][0]['sha256']='0'*64;i.write_text(json.dumps(d))
            with self.assertRaises(ValueError):restore.restore(i,r/'out')
            self.assertFalse((r/'out').exists())
if __name__=='__main__':unittest.main()
