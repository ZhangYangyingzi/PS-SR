"""Six-depth state-aware side adapter on the frozen UltraFlash streaming SR DiT."""
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT.parent
SOURCE = PS / 'ultraflash_condition_projection_scale100_v1'
_previous_spec = importlib.util.spec_from_file_location('prior_condition_common', SOURCE / 'scripts/common.py')
previous = importlib.util.module_from_spec(_previous_spec)
_previous_spec.loader.exec_module(previous)

BASE = previous.base
SEED = 2026
INDICES = (4, 9, 14, 19, 24, 29)
STEPS = (0, 250, 500, 750, 1000)
ADAPTER_DIM = 384
HEADS = 6
WEIGHTS = {'latent': .5, 'temporal': .1, 'rgb': .2, 'lpips': .15, 'hf': 1.}
KEYS = ('psnr', 'ssim', 'lpips', 'dists', 'latent_mse', 'latent_mae')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def write_csv(path, values):
    values = list(values)
    assert values
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    os.replace(tmp, path)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def rows(split):
    assert split in ('train', 'val')
    return previous.rows(split)


def initial_condition_checkpoint():
    candidates = sorted((SOURCE / 'runs/main').glob('step_*.pt'))
    matches = []
    for path in candidates:
        if path.stem != 'step_001500':
            continue
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('step') == 1500 and 'condition_weight' in state:
            matches.append(path)
    assert len(matches) == 1, f'Expected one actual Train100 step1500 checkpoint, found {matches}'
    return matches[0]


def initial_config():
    path = initial_condition_checkpoint()
    return {'name': 'ultraflash_state_aware_adapter_6depth_v1',
            'source_train100_manifest': str(previous.MANIFEST),
            'source_condition_checkpoint': str(path.resolve()),
            'upsampler_checkpoint': str(BASE.checkpoint_path()),
            'sr_checkpoint': str(BASE.SR_CKPT.resolve()),
            'wan_checkpoint': str(BASE.WAN_CKPT),
            'seed': SEED, 'steps': list(STEPS), 'batch_size': 1,
            'adapter_indices': list(INDICES), 'adapter_dim': ADAPTER_DIM,
            'adapter_heads': HEADS,
            'state_aware_fusion': 'static base_proj + zero_out(cross_attention(query=static, context=[dynamic,static]))',
            'dynamic_source': 'current SR DiT block output h',
            'time_conditioning': None,
            'loss_weights': WEIGHTS, 'lpips_frames': list(range(17)),
            'optimizer': {'name': 'AdamW', 'betas': [.9,.99], 'weight_decay': 1e-4,
                          'condition_lr': 1e-4, 'adapter_lr': 5e-5,
                          'grad_clip': 1., 'scheduler': None},
            'precision': 'frozen BF16 base; FP32 adapter and condition projection parameters with BF16 autocast forward'}


def split_audit():
    groups = {s: [r for r in csv.DictReader(previous.MANIFEST.open()) if r['split'] == s]
              for s in ('train', 'val', 'test')}
    vids = {s: {r['video_id'] for r in group} for s, group in groups.items()}
    for s, n, v in (('train', 400, 100), ('val', 24, 6), ('test', 24, 6)):
        assert len(groups[s]) == n and len(vids[s]) == v
        counts = collections.Counter(r['video_id'] for r in groups[s])
        assert all(value == 4 for value in counts.values())
        assert len({r['clip_id'] for r in groups[s]}) == n
    assert not vids['train'] & vids['val']
    assert not vids['train'] & vids['test']
    assert not vids['val'] & vids['test']
    result = {'source_manifest': str(previous.MANIFEST),
              'train_videos': 100, 'train_clips': 400,
              'val_videos': 6, 'val_clips': 24,
              'test_videos': 6, 'test_clips': 24,
              'train_val_overlap': 0, 'train_test_overlap': 0, 'val_test_overlap': 0,
              'test_executed': False,
              'train_clip_ids': [r['clip_id'] for r in groups['train']],
              'val_clip_ids': [r['clip_id'] for r in groups['val']],
              'test_clip_ids': [r['clip_id'] for r in groups['test']]}
    write_json(ROOT / 'audit/split_audit.json', result)
    return result


class ControlBlock(nn.Module):
    def __init__(self):
        super().__init__()
        d = ADAPTER_DIM
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(approximate='tanh'), nn.Linear(4*d, d))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        def heads(t): return t.reshape(b, n, HEADS, d//HEADS).transpose(1, 2)
        y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=0., is_causal=False)
        x = x + self.out(y.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.norm2(x))


class StateAwareAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        d = ADAPTER_DIM
        self.base_proj = nn.Conv3d(16, d, kernel_size=(1,2,2), stride=(1,2,2))
        self.dynamic_proj = nn.Linear(1536, d)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.attn_out = nn.Linear(d, d)
        self.zero_out = nn.Linear(d, d)
        nn.init.zeros_(self.zero_out.weight)
        nn.init.zeros_(self.zero_out.bias)
        self.control_blocks = nn.ModuleDict({str(i): ControlBlock() for i in INDICES})
        self.zero_connectors = nn.ModuleDict({str(i): nn.Linear(d, 1536, bias=False) for i in INDICES})
        for connector in self.zero_connectors.values(): nn.init.zeros_(connector.weight)
        self.static_tokens = None
        self.shape_trace = []

    def set_condition(self, c):
        # c is the frozen-upsampler condition for the current 2/2/1-frame chunk.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feature = self.base_proj(c)
        self.static_tokens = feature.flatten(2).transpose(1, 2)
        if len(self.shape_trace) < 12:
            self.shape_trace.append({'stage': 'static', 'condition': list(c.shape),
                                     'static_feature': list(self.static_tokens.shape)})

    def forward(self, idx, h):
        static = self.static_tokens
        assert static is not None and static.shape[:2] == h.shape[:2]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dynamic = self.dynamic_proj(h)
            context = torch.cat((dynamic, static), dim=1)
            q = self.query(static)
            k = self.key(context)
            v = self.value(context)
            b, n, d = q.shape
            def heads(t): return t.reshape(b, -1, HEADS, d//HEADS).transpose(1, 2)
            attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=0., is_causal=False)
            attended = attended.transpose(1, 2).reshape(b, n, d)
            fused = static + self.zero_out(self.attn_out(attended))
            control = self.control_blocks[str(idx)](fused)
            residual = self.zero_connectors[str(idx)](control)
        if len(self.shape_trace) < 50:
            self.shape_trace.append({'stage': 'block', 'index': idx, 'base_state': list(h.shape),
                'dynamic_feature': list(dynamic.shape), 'static_feature': list(static.shape),
                'query': list(q.shape), 'context': list(context.shape),
                'fused': list(fused.shape), 'control': list(control.shape),
                'residual': list(residual.shape)})
        return h + residual


class SideRunner:
    def __init__(self, sr, adapter):
        self.sr = sr
        self.adapter = adapter
        self.hooks = []
        for idx in INDICES:
            def inject(module, inputs, output, index=idx):
                h, k, v = output
                return self.adapter(index, h), k, v
            self.hooks.append(sr.blocks[idx].register_forward_hook(inject))

    def close(self):
        for hook in self.hooks: hook.remove()
        self.hooks.clear()

    def refine(self, condition, seed_value):
        from sr.stream_forward import _stream_forward_causal
        generator = torch.Generator(device='cuda').manual_seed(seed_value)
        context = [torch.zeros(512, 4096, device='cuda', dtype=torch.bfloat16)]
        cache, outputs = None, []
        self.sr.clear_cross_kv()
        for start, stop in ((0,2), (2,4), (4,5)):
            part = condition[:, :, start:stop]
            self.adapter.set_condition(part)
            noise = torch.randn(part.shape, device='cuda', dtype=torch.bfloat16, generator=generator)
            flow, cache = _stream_forward_causal(model=self.sr, x_list=[noise[0]],
                t=torch.full((1,), 1000., device='cuda'), context=context,
                y_list=[part[0]], temporal_offset=start, kv_len=3, cache_state=cache)
            outputs.append(noise - torch.stack(flow))
        self.sr.clear_cross_kv()
        self.adapter.static_tokens = None
        return torch.cat(outputs, 2)


def load_bundle(with_losses=False):
    up, sr, _, _ = previous.load_models()
    path = initial_condition_checkpoint()
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['step'] == 1500
    with torch.no_grad(): sr.patch_embedding.condition_weight.copy_(state['condition_weight'])
    assert torch.equal(sr.patch_embedding.condition_weight.detach().cpu(), state['condition_weight'])
    assert sr.patch_embedding.condition_weight.numel() == 98304
    adapter = StateAwareAdapter().cuda().eval()
    runner = SideRunner(sr, adapter)
    vae = BASE.load_vae() if with_losses else None
    lp = BASE.load_lpips() if with_losses else None
    return up, sr, adapter, runner, vae, lp


def latent_pair(row):
    if row['split'] == 'train':
        path = SOURCE / 'data/cache/train' / f"{row['clip_id']}.pt"
    else:
        path = previous.OLD_CACHE / row['split'] / f"{row['clip_id']}.pt"
    item = torch.load(path, map_location='cpu', weights_only=False)
    return item['z_lr'].unsqueeze(0).cuda().bfloat16(), item['z_hr'].unsqueeze(0).cuda().bfloat16()


def forward_losses(row, bundle, seed_value):
    up, sr, adapter, runner, vae, lp = bundle
    x, gt = latent_pair(row)
    with torch.no_grad(): c = BASE.run_up(up, x)
    pred = runner.refine(c, seed_value)
    rgb_gt = BASE.gt_tensor(row)
    rgb = BASE.decode_train(vae, pred)
    assert tuple(rgb.shape) == (1,3,17,480,864)
    raw, weighted, total = BASE.losses(pred, gt, rgb, rgb_gt, lp, list(range(17)))
    return raw, weighted, total


def frozen_identity(up, sr, vae):
    return previous.identity(up, sr, vae)


def adapter_checksum(adapter):
    return BASE.tensor_checksum(adapter)


def checkpoint_path(step): return ROOT / 'runs' / f'step_{step:04d}.pt'


def save_checkpoint(path, step, sr, adapter, optimizer=None):
    value = {'step': step, 'condition_weight': sr.patch_embedding.condition_weight.detach().cpu(),
             'adapter': {k: v.detach().cpu() for k,v in adapter.state_dict().items()},
             'config': initial_config()}
    if optimizer is not None:
        value['optimizer'] = optimizer.state_dict()
        value['rng'] = {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def restore_checkpoint(path, sr, adapter):
    state = torch.load(path, map_location='cpu', weights_only=False)
    with torch.no_grad(): sr.patch_embedding.condition_weight.copy_(state['condition_weight'])
    adapter.load_state_dict(state['adapter'], strict=True)
    return state


def optimizer_for(sr, adapter):
    return torch.optim.AdamW([
        {'params': [sr.patch_embedding.condition_weight], 'lr': 1e-4},
        {'params': list(adapter.parameters()), 'lr': 5e-5}],
        betas=(.9,.99), weight_decay=1e-4)


def trainable(sr, adapter):
    return [sr.patch_embedding.condition_weight, *adapter.parameters()]
