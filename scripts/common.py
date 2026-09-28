import importlib.util
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT.parent
PREVIOUS = PS / 'ultraflash_decoder_aware_balanced_loss_v2'
spec = importlib.util.spec_from_file_location('balanced_reuse', PREVIOUS / 'scripts/common.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT = ROOT
for name in dir(base):
    if not name.startswith('_') and name not in ('ROOT','PS'):
        globals()[name] = getattr(base, name)
STEPS = (0,250,500,750,1000)
RUN = ROOT / 'runs/main'
AUDIT = ROOT / 'audit'
FRAMES = list(range(17))
_original_gt_frames=base.gt_frames

def gt_frames(row):
    if row['split']=='train':
        cached=PREVIOUS/'cache/gt_train'/f"{row['clip_id']}.pt"
        if cached.is_file():return torch.load(cached,map_location='cpu',weights_only=True)
    return _original_gt_frames(row)

base.gt_frames=gt_frames

def write_json(path,data):
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def write_csv(path,data):
    import os
    data=list(data)
    assert data
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    os.replace(tmp,path)

class ConditionProjection(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        assert isinstance(conv, torch.nn.Conv3d)
        assert tuple(conv.weight.shape)==(1536,32,1,2,2) and conv.groups==1
        self.register_buffer('latent_weight', conv.weight[:,:16].detach().clone())
        self.condition_weight=torch.nn.Parameter(conv.weight[:,16:].detach().float().clone())
        self.register_buffer('bias',conv.bias.detach().clone())
        self.stride,self.padding,self.dilation,self.groups=conv.stride,conv.padding,conv.dilation,conv.groups

    @property
    def weight(self):
        return torch.cat((self.latent_weight,self.condition_weight.to(self.latent_weight.dtype)),dim=1)

    def forward(self,x):
        return F.conv3d(x,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)

def load_models(with_losses=False):
    up=base.load_up(train=False)
    sr=base.load_sr()
    assert base.SR_CKPT.resolve().name=='2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth'
    sr.patch_embedding=ConditionProjection(sr.patch_embedding)
    vae=base.load_vae() if with_losses else None
    lp=base.load_lpips() if with_losses else None
    return up,sr,vae,lp

def parameters(sr):
    result=[p for p in sr.parameters() if p.requires_grad]
    assert len(result)==1 and result[0] is sr.patch_embedding.condition_weight
    return result

def forward_losses(row,up,sr,vae,lp,seed_value,only=None):
    x,gt=base.latent_pair(row)
    with torch.no_grad(): c=base.run_up(up,x)
    z=base.refine(sr,c,seed_value)
    if only in ('latent','temporal'):
        loss=(F.mse_loss(z.float(),gt.float()) if only=='latent' else
              F.l1_loss(z[:,:,1:].float()-z[:,:,:-1].float(),gt[:,:,1:].float()-gt[:,:,:-1].float()))
        return {only:loss},{only:WEIGHTS[only]*loss},WEIGHTS[only]*loss
    rgb_gt=base.gt_tensor(row)
    rgb=base.decode_train(vae,z)
    assert tuple(rgb.shape)==(1,3,17,480,864)
    return base.losses(z,gt,rgb,rgb_gt,lp,FRAMES)

def atomic_save(path,data):
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(data,tmp);os.replace(tmp,path)

def checkpoint(step): return RUN/f'step_{step:06d}.pt'

def restore(sr,step):
    state=torch.load(checkpoint(step),map_location='cpu',weights_only=False)
    with torch.no_grad(): sr.patch_embedding.condition_weight.copy_(state['condition_weight'])
    return state

def frozen_hash(sr):
    import hashlib
    h=hashlib.sha256()
    for name,v in sorted(sr.state_dict().items()):
        if name=='patch_embedding.condition_weight': continue
        h.update(name.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def identity(up,sr,vae):
    return {'upsampler':base.tensor_checksum(up),'sr_frozen':frozen_hash(sr),
            'latent_projection':__import__('hashlib').sha256(sr.patch_embedding.latent_weight.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            'wan':base.tensor_checksum(vae)}

def settings():
    return {'weights':WEIGHTS,'lpips_frames':FRAMES,'seed':SEED,'steps':list(STEPS),
        'lr':1e-4,'betas':[.9,.99],'weight_decay':1e-4,'grad_clip':1.,
        'upsampler':str(base.checkpoint_path()),'sr_checkpoint':str(base.SR_CKPT.resolve()),
        'trainable':'patch_embedding.condition_weight','condition_master_dtype':'float32',
        'forward_weight_dtype':'bfloat16','bias_frozen':True}

def ensure():
    for folder in ('audit','logs','metrics','runs/main','outputs'): (ROOT/folder).mkdir(parents=True,exist_ok=True)
