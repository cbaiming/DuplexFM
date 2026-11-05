"""Audited source transformation: skip disabled SINGLE-module computations.

Apply only to a new release copy. No tensors, parameter shapes or shared layers
are added or removed. Historical full/no_* behavior is byte-for-byte retained.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

BASE_MODEL_SHA256 = "b033311299a516f7c64f438a5c8a350ba98cb5c53413b4a7b156d611415c750a"
SINGLE_CONDITIONS = ("only_pair", "only_cross", "only_access")

FAST_MAPS = '''    def _single_module_maps(self, batch):
        """Compute active evidence only; retain the exact 16-channel interface."""
        condition = self.ablation
        if condition not in ('only_pair', 'only_cross', 'only_access'):
            raise ValueError('Fast maps require a registered single-module condition')
        gm,tm=batch['guide_mask'].bool(),batch['target_mask'].bool()
        if gm.ndim!=2 or tm.ndim!=2 or gm.shape[0]!=tm.shape[0]:
            raise ValueError('Masks do not match sequence axes')
        b,g=gm.shape;t=tm.shape[1]
        if not gm.any(1).all() or not tm.any(1).all():
            raise ValueError('Empty observed sequence')
        valid=gm[:,:,None]&tm[:,None,:]
        # float32 matches the retained original channels, including under AMP.
        # Disabled evidence is never inspected, evaluated, or gathered here.
        def zeros(channels):
            return torch.zeros((b,channels,g,t),device=gm.device,dtype=torch.float32)
        if condition=='only_pair':
            guide=batch['guide'].masked_fill(~gm,0)
            target=batch['target'].masked_fill(~tm,0)
            if guide.shape!=(b,g) or target.shape!=(b,t):
                raise ValueError('Nucleotide axes do not match masks')
            if ((guide<0)|(guide>3)).any() or ((target<0)|(target>3)).any():
                raise ValueError('Unknown real nucleotide')
            pair_id=guide[:,:,None]*4+target[:,None,:]
            pairs=self.pair_embedding(pair_id).permute(0,3,1,2)
        else:
            pairs=zeros(4)
        if condition=='only_cross':
            q,k=batch['q'],batch['k']
            if q.ndim!=3 or q.shape[:2]!=(b,t):
                raise ValueError('Invalid Q/K dimensions')
            d=q.shape[2]
            if k.shape!=(b,g,d) or d%8:
                raise ValueError('Invalid Q/K dimensions')
            q=q.masked_fill(~tm[:,:,None],0);k=k.masked_fill(~gm[:,:,None],0)
            c=torch.einsum('bghd,bthd->bhgt',k.reshape(b,g,8,d//8),q.reshape(b,t,8,d//8))/math.sqrt(d//8)
            c=torch.tanh(c.float()/(F.softplus(self.log_cross_scale)[None,:,None,None]+.01))
        else:
            c=zeros(8)
        if condition=='only_access':
            a=batch['accessibility'].float().masked_fill(~tm,0)
            if a.shape!=(b,t):
                raise ValueError('Accessibility axes do not match masks')
            kernel=a.new_ones(1,1,7)
            background=(F.conv1d(a[:,None],kernel,padding=3)[:,0]/
                        F.conv1d(tm[:,None].to(a.dtype),kernel,padding=3)[:,0].clamp_min(1))
            local=(a-background).masked_fill(~tm,0)
            access=a[:,None,None,:].expand(-1,1,g,-1)
            contrast=local[:,None,None,:].expand(-1,1,g,-1)
        else:
            access=zeros(1);contrast=zeros(1)
        gp=batch['guide_pos'].float()/32
        tp=(batch['target_pos'].float()-(batch['target_lengths'][:,None].float()-1)/2)/30
        x=torch.cat([pairs,c,access,contrast,
                     gp[:,None,:,None].expand(-1,1,-1,t),tp[:,None,None,:].expand(-1,1,g,-1)],1)
        return x.masked_fill(~valid[:,None],0),valid

'''

ENCODER_EARLY = '''        if self.condition in ('only_pair','only_access'):
            # Both sequence masks and coordinate priors remain available. No
            # foundation, adapter or projection forward is needed here.
            result=dict(batch)
            result.pop('q',None);result.pop('k',None)
            return result
'''


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def generated_bytes(source: bytes) -> bytes:
    if sha256(source) != BASE_MODEL_SHA256:
        raise ValueError("Fastpath base model SHA mismatch")
    text = source.decode("utf-8").replace("\r\n", "\n")
    marker = "    def make_maps(self, batch):\n"
    if text.count(marker) != 1:
        raise ValueError("make_maps source anchor mismatch")
    text = text.replace(marker, FAST_MAPS + marker +
        "        if self.ablation in ('only_pair','only_cross','only_access'):\n"
        "            return self._single_module_maps(batch)\n"
        "        return self._legacy_make_maps(batch)\n\n"
        "    def _legacy_make_maps(self, batch):\n", 1)
    marker = "    def interaction_inputs(self,batch):\n"
    if text.count(marker) != 1:
        raise ValueError("interaction_inputs source anchor mismatch")
    text = text.replace(marker, marker + ENCODER_EARLY +
        "        return self._legacy_interaction_inputs(batch)\n\n"
        "    def _legacy_interaction_inputs(self,batch):\n", 1)
    return text.encode("utf-8")


def apply_to(code_root: str | Path) -> dict:
    root = Path(code_root).resolve()
    path = root / "duplexfm_mechanism/model.py"
    if "H100_remote" in root.parts or root == Path(__file__).resolve().parents[1] / "release/code":
        raise ValueError("Refuse mutation of original or historical release")
    receipt_path = root / "fastpath_patch_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if sha256(path.read_bytes()) != receipt["model_after_sha256"]:
            raise ValueError("Already patched model SHA mismatch")
        return receipt
    source = path.read_bytes()
    patched = generated_bytes(source)
    originals = root / "fastpath_originals"
    originals.mkdir(exist_ok=False)
    (originals / "model.py").write_bytes(source)
    receipt = {
        "version": 1, "purpose": "single-module disabled-compute elimination",
        "model_before_sha256": sha256(source), "model_after_sha256": sha256(patched),
        "conditions": list(SINGLE_CONDITIONS),
        "parameters_and_shapes_unchanged": True,
        "retained_channels": 16, "shared_convolution_and_pooling_unchanged": True,
        "source192_interface_unchanged": True,
        "disabled_encoder_execution": False,
        "dropout_stream_matches_historical_training": False,
        "training_restart": "all nine conditions fresh from exact original per-seed task initial",
    }
    path.write_bytes(patched)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("code_root")
    args = parser.parse_args()
    print(json.dumps(apply_to(args.code_root), indent=2))
