"""Physics-free portable inputs. Existing arrays are reused read-only, without BIO/Inta."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
PAD_TOKEN=4
TARGET_WIDTH=60
BASES=np.array(list('ACGU'))

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

class PairStore:
    def __init__(self,root):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8-sig'))
        if self.manifest.get('target_padding','collate_only_center')!='collate_only_center':
            raise ValueError('Expected original coordinates')
        names=('target_tokens','guide_tokens','guide_lengths','target_ids','guide_ids','labels')
        self.required={f'data/{n}.npy' for n in names}|{'accessibility.npy','data/indices_train.npy','data/indices_val.npy'}
        self.required.update('data/'+p.name for p in (self.root/'data').glob('indices_*.npy'))
        for n in names:setattr(self,n,np.load(self.root/'data'/f'{n}.npy',mmap_mode='r',allow_pickle=False))
        path=self.root/'data/target_lengths.npy'
        if path.exists():
            self.required.add('data/target_lengths.npy')
            self.target_lengths=np.load(path,mmap_mode='r',allow_pickle=False)
        else:self.target_lengths=np.full(len(self.target_tokens),self.target_tokens.shape[1],np.int64)
        self.accessibility=np.load(self.root/'accessibility.npy',mmap_mode='r',allow_pickle=False)
        self._validate()

    def __len__(self):return len(self.labels)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        for v in vars(self).values():
            if isinstance(v,np.memmap) and getattr(v,'_mmap',None) is not None:v._mmap.close()

    def _validate(self):
        for name in ('guide','target'):
            tokens,lengths=getattr(self,name+'_tokens'),getattr(self,name+'_lengths')
            if tokens.ndim!=2 or tokens.dtype.kind not in 'iu' or lengths.shape!=(len(tokens),):raise ValueError('Token dimensions')
            if lengths.dtype.kind not in 'iu' or np.any(lengths<1) or np.any(lengths>tokens.shape[1]):raise ValueError('True lengths')
            if name=='target' and (tokens.shape[1]>60 or np.any(lengths>60)):raise ValueError('Target >60; never crop')
            for lo in range(0,len(tokens),32768):
                v=tokens[lo:lo+32768];m=np.arange(tokens.shape[1])[None]<lengths[lo:lo+32768,None]
                if np.any((v[m]<0)|(v[m]>3)) or np.any(v[~m]!=4):raise ValueError('Unknown real base or non-PAD tail')
            ids=getattr(self,name+'_ids')
            if ids.shape!=(len(self),) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(tokens)):raise ValueError('Row IDs')
        if self.labels.ndim!=1 or not np.isfinite(self.labels).all():raise ValueError('Nonfinite labels')
        if self.accessibility.shape!=self.target_tokens.shape:raise ValueError('Accessibility coordinates')

    def verify_integrity(self):
        declared=self.manifest.get('files_sha256',{})
        if not self.required<=set(declared):raise ValueError('Required input SHA missing')
        actual={}
        for name in sorted(self.required):
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root.resolve()):raise ValueError('Input escapes root')
            actual[name]=sha256(path)
            if actual[name]!=declared[name]:raise ValueError('Input SHA mismatch: '+name)
        return {'files_sha256':actual,'verified_files':len(actual),'manifest_sha256':sha256(self.root/'manifest.json'),
                'unused_legacy_physics_arrays_opened':False}

    def _row_indices(self,ids,allow_empty=False):
        ids=np.asarray(ids)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or (not allow_empty and not len(ids)) or np.any(ids<0) or np.any(ids>=len(self)):
            raise ValueError('Invalid row selection')
        return ids.astype(np.int64,copy=False)

    def indices(self,name):
        if not name or Path(name).name!=name or name in ('.','..') or '\\' in name:raise ValueError('Unsafe partition')
        ids=self._row_indices(np.load(self.root/'data'/f'indices_{name}.npy',allow_pickle=False))
        if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate partition row')
        return ids

    def collate(self,ids):
        import torch
        ids=self._row_indices(ids)
        ti,gi=self.target_ids[ids],self.guide_ids[ids]
        tl,gl=np.asarray(self.target_lengths[ti],np.int64),np.asarray(self.guide_lengths[gi],np.int64)
        b,g=len(ids),int(gl.max())
        target=np.full((b,60),4,np.int64);guide=np.full((b,g),4,np.int64)
        tp=np.full_like(target,-1);gp=np.full_like(guide,-1);access=np.zeros((b,60),np.float32)
        ts,gs=[],[]
        for i,(tid,gid,nt,ng) in enumerate(zip(ti,gi,tl,gl)):
            left=(60-int(nt))//2;window=slice(left,left+int(nt))
            t=np.asarray(self.target_tokens[tid,:nt],np.int64);v=np.asarray(self.guide_tokens[gid,:ng],np.int64)
            a=np.asarray(self.accessibility[tid,:nt],np.float32)
            if not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid accessibility proxy')
            target[i,window]=t;guide[i,:ng]=v;tp[i,window]=np.arange(nt);gp[i,:ng]=np.arange(ng)
            access[i,window]=a;ts.append(''.join(BASES[t]));gs.append(''.join(BASES[v]))
        arrays={'target':target,'guide':guide,'target_pos':tp,'guide_pos':gp,'target_mask':tp>=0,'guide_mask':gp>=0,
                'target_lengths':tl.copy(),'guide_lengths':gl.copy(),'accessibility':access,
                'labels':np.array(self.labels[ids],np.float32,copy=True),'row_indices':ids.copy()}
        batch={k:torch.from_numpy(v) for k,v in arrays.items()}
        batch.update(target_sequences=ts,guide_sequences=gs)
        return batch
