"""PairAC16: learned ordered pairs, live cross, frozen accessibility; concat only."""
from __future__ import annotations
import importlib.abc
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F
from duplexfm_joint.model import CellNorm, LocalBlock, _diagonal_indices
from duplexfm_joint.backbones import JointModel


class PairACFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get('explicit_products') is not False or config.get('site_energy') is not False:
            raise ValueError('PairAC does not implement manual products or energy branches')
        fixed={'heads':8,'hidden_width':32,'pair_embedding_dim':4,'projection_dim':256,
               'local_accessibility':True,'accessibility_window':7,'architecture':'PairAC16'}
        if any(config.get(k)!=v for k,v in fixed.items()):
            raise ValueError('PairAC ablation requires the original fixed channel and width contract')
        self.ablation='full'  # nonparameter identity; checkpoint signature binds this condition
        self.heads, self.width = 8, 32
        self.input_channels, self.pair_dim = 16, 4
        self.pair_embedding = nn.Embedding(16, 4)
        nn.init.normal_(self.pair_embedding.weight, std=.02)
        self.log_cross_scale = nn.Parameter(torch.full((8,), math.log(math.expm1(4.))))
        self.stem = nn.Sequential(nn.Conv2d(16,32,1),CellNorm(32),nn.GELU())
        self.blocks = nn.ModuleList([LocalBlock(32,d,config['dropout']) for d in (1,1,2)])
        # Retain the pre-existing position regions and transfer interface;
        # these are positional priors, not a seed-match test or candidate gate.
        self.site_encoder = nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Dropout(config['dropout']))
        self.site_attention, self.output = nn.Linear(64,1), nn.Linear(64,1)
        nn.init.zeros_(self.site_attention.weight)
        nn.init.zeros_(self.site_attention.bias)

    @staticmethod
    def channel_names():
        return ([f'pair_{i}' for i in range(4)]+[f'cross_{i}' for i in range(8)]+
                ['access','access_local','guide_position','target_position'])

    def make_maps(self, batch):
        q,k=batch['q'],batch['k']
        b,t,d=q.shape; g=k.shape[1]
        if k.shape!=(b,g,d) or d%8:
            raise ValueError('Invalid Q/K dimensions')
        gm,tm=batch['guide_mask'].bool(),batch['target_mask'].bool()
        if gm.shape!=(b,g) or tm.shape!=(b,t):
            raise ValueError('Masks do not match sequence axes')
        if not gm.any(1).all() or not tm.any(1).all():
            raise ValueError('Empty observed sequence')
        valid=gm[:,:,None]&tm[:,None,:]
        # PAD never indexes the lookup, including arbitrary padded token values.
        guide=batch['guide'].masked_fill(~gm,0)
        target=batch['target'].masked_fill(~tm,0)
        if ((guide<0)|(guide>3)).any() or ((target<0)|(target>3)).any():
            raise ValueError('Unknown real nucleotide')
        pair_id=guide[:,:,None]*4+target[:,None,:]
        pairs=self.pair_embedding(pair_id).permute(0,3,1,2)
        q=q.masked_fill(~tm[:,:,None],0);k=k.masked_fill(~gm[:,:,None],0)
        c=torch.einsum('bghd,bthd->bhgt',k.reshape(b,g,8,d//8),q.reshape(b,t,8,d//8))/math.sqrt(d//8)
        c=torch.tanh(c.float()/(F.softplus(self.log_cross_scale)[None,:,None,None]+.01))
        a=batch['accessibility'].float().masked_fill(~tm,0)
        kernel=a.new_ones(1,1,7)
        background=(F.conv1d(a[:,None],kernel,padding=3)[:,0]/
                    F.conv1d(tm[:,None].to(a.dtype),kernel,padding=3)[:,0].clamp_min(1))
        local=(a-background).masked_fill(~tm,0)
        gp=batch['guide_pos'].float()/32
        tp=(batch['target_pos'].float()-(batch['target_lengths'][:,None].float()-1)/2)/30
        x=torch.cat([pairs,c,a[:,None,None,:].expand(-1,1,g,-1),
                     local[:,None,None,:].expand(-1,1,g,-1),
                     gp[:,None,:,None].expand(-1,1,-1,t),tp[:,None,None,:].expand(-1,1,g,-1)],1)
        if self.ablation not in ('full','no_pair','no_cross','no_access',
                                  'only_pair','only_cross','only_access'):
            raise ValueError('Unknown fixed expert mask')
        # masked_fill removes every route from the disabled input. All upstream
        # computations still execute, preserving forward dropout RNG consumption.
        disabled_channels={
            'no_pair':(0,1,2,3), 'no_cross':tuple(range(4,12)),
            'no_access':(12,13), 'only_pair':tuple(range(4,14)),
            'only_cross':(0,1,2,3,12,13), 'only_access':tuple(range(12))}
        if self.ablation in disabled_channels:
            disabled=torch.zeros(16,dtype=torch.bool,device=x.device)
            disabled[list(disabled_channels[self.ablation])]=True
            x=x.masked_fill(disabled[None,:,None,None],0)
        return x.masked_fill(~valid[:,None],0),valid

    def grid_features(self,batch):
        x,valid=self.make_maps(batch)
        mask=valid[:,None]
        h=self.stem(x).masked_fill(~mask,0)
        for block in self.blocks:h=block(h,mask)
        return h,valid

    @staticmethod
    def site_pool(h,valid,batch):
        b,c,g,t=h.shape
        j,clamped=_diagonal_indices(g,t,h.device)
        hd=h.gather(3,clamped[None,None].expand(b,c,-1,-1))
        vd=valid.gather(2,clamped[None].expand(b,-1,-1))&(j[None]>=0)&(j[None]<t)
        gp,gm=batch['guide_pos'],batch['guide_mask'].bool()
        regions=[(gp>=1)&(gp<=7),(gp>=8)&(gp<=11),(gp==0)|(gp>=12)]
        pieces=[]
        for region in regions:
            m=vd&(region&gm)[:,:,None];count=m.sum(1)
            mean=(hd*m[:,None]).sum(2)/count[:,None].clamp_min(1)
            maximum=hd.masked_fill(~m[:,None],torch.finfo(h.dtype).min).amax(2)
            maximum=torch.where(count[:,None]>0,maximum,torch.zeros_like(maximum))
            pieces.extend([mean,maximum])
        sites=torch.cat(pieces,1).transpose(1,2)
        minimum=torch.minimum(gm.sum(1),batch['target_mask'].sum(1)).clamp(max=6)
        allowed=vd.sum(1)>=minimum[:,None]
        fallback=vd.sum(1)==vd.sum(1).amax(1,keepdim=True)
        return sites,torch.where(allowed.any(1)[:,None],allowed,fallback)

    def encode_sites(self,batch):
        h,valid=self.grid_features(batch)
        sites,allowed=self.site_pool(h,valid,batch)
        return self.site_encoder(sites),allowed

    def forward_features(self,batch):
        z,allowed=self.encode_sites(batch)
        weights=self.site_attention(z).squeeze(-1).float().masked_fill(~allowed,-torch.inf).softmax(1)
        return (z*weights[:,:,None]).sum(1)

    def forward(self,batch):return self.output(self.forward_features(batch)).squeeze(-1)


class PairACModel(JointModel):
    def __init__(self,config,variant='seed42'):
        from .protocol import variant_parts
        condition,seed=variant_parts(variant)
        class NoOptionalFlash(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.endswith('flash_attn_triton'):
                    raise ImportError('Use the audited native PyTorch fallback')
        blocker=NoOptionalFlash();sys.meta_path.insert(0,blocker)
        try:super().__init__(config)
        finally:sys.meta_path.remove(blocker)
        self.variant=variant
        self.condition,self.source_seed=condition,seed
        self.fusion.ablation=condition

    def masked_parameter_names(self):
        names={n for n,p in self.named_parameters() if p.requires_grad}
        pair={'fusion.pair_embedding.weight'}
        cross={n for n in names if not n.startswith('fusion.')}|{'fusion.log_cross_scale'}
        if self.condition in ('no_pair','only_cross'):return pair
        if self.condition in ('no_cross','only_pair'):return cross
        if self.condition=='only_access':return pair|cross
        if self.condition in ('full','no_access'):return set()
        raise ValueError('Unknown fixed expert mask')

    def _encode_batch(self,sequences,target):
        if any(not s or set(s)-set('ACGU') for s in sequences):raise ValueError('Noncanonical sequence')
        tokenizer=self.target_tokenizer if target else self.guide_tokenizer
        texts=[' '.join(s.replace('U','T')) for s in sequences] if target else sequences
        tokens=tokenizer(texts,return_tensors='pt',padding=True,truncation=False,return_special_tokens_mask=True)
        real=tokens['attention_mask'].bool()&~tokens.pop('special_tokens_mask').bool()
        lengths=list(map(len,sequences))
        if real.sum(1).tolist()!=lengths or ((tokens['input_ids']==tokenizer.unk_token_id)&real).any():
            raise ValueError('Tokenizer nucleotide identity changed')
        positions=torch.zeros(len(sequences),max(lengths),dtype=torch.long)
        for i,mask in enumerate(real):positions[i,:lengths[i]]=mask.nonzero(as_tuple=True)[0]
        device=next(self.parameters()).device
        kwargs={k:v.to(device) for k,v in tokens.items() if k in ('input_ids','attention_mask','token_type_ids')}
        if not target:kwargs.pop('token_type_ids',None)
        encoder=self.target_encoder if target else self.guide_encoder
        output=encoder(**kwargs)
        hidden=output.last_hidden_state if hasattr(output,'last_hidden_state') else output[0]
        projection=self.target_projection if target else self.guide_projection
        qk=self.q_projection if target else self.k_projection
        hidden=qk(projection(hidden))
        return hidden.gather(1,positions.to(device)[:,:,None].expand(-1,-1,hidden.shape[-1]))

    def interaction_inputs(self,batch):
        result=dict(batch)
        for name,is_target,key in [('target',True,'q'),('guide',False,'k')]:
            encoded=self._encode_batch(batch[name+'_sequences'],is_target)
            positions=batch[name+'_pos'].clamp_min(0)
            values=encoded.gather(1,positions[:,:,None].expand(-1,-1,encoded.shape[-1]))
            result[key]=values.masked_fill(~batch[name+'_mask'][:,:,None],0)
        return result

    def source192(self,batch,core_start=19):
        """Explicit siRNA design-site interface, not a miRNA candidate restriction."""
        if not all(len(s)==19 for s in batch['guide_sequences']):raise ValueError('Expected guide19')
        encoded=self.interaction_inputs(batch)
        h,valid=self.fusion.grid_features(encoded)
        sites,allowed=self.fusion.site_pool(h,valid,encoded)
        gi=batch['guide_pos']==0;tj=batch['target_pos']==core_start+18
        if not gi.any(1).all() or not tj.any(1).all():raise ValueError('Missing design-site coordinates')
        offset=gi.long().argmax(1)+tj.long().argmax(1)
        if not allowed.gather(1,offset[:,None]).all():raise ValueError('Invalid design site')
        return sites.gather(1,offset[:,None,None].expand(-1,1,192))[:,0]
