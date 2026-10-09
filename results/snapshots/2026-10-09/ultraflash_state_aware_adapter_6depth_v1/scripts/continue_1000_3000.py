"""Strict continuation of the existing six-depth experiment; never accesses Test."""
import argparse
import collections
import gc
import hashlib
import math
import time

from common import *

CONT = ROOT / 'continuation_1000_3000'
SOURCE_CKPT = checkpoint_path(1000)
MILESTONES = (1500, 2000, 2500, 3000)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_state():
    candidates = sorted((ROOT / 'runs').glob('*.pt'))
    matches = []
    for path in candidates:
        if path.name != 'step_1000.pt':
            continue
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value.get('step') == 1000 and 'condition_weight' in value and 'adapter' in value:
            matches.append((path, value))
    assert len(matches) == 1, f'Expected one step1000 training checkpoint; found {len(matches)}'
    return matches[0]


def audit():
    path, state = source_state()
    record = {'checkpoint_path': str(path.resolve()), 'checkpoint_size': path.stat().st_size,
              'checkpoint_sha256': sha256(path), 'global_step': state['step'],
              'has_model_state': 'condition_weight' in state and 'adapter' in state,
              'has_condition_projection': tuple(state['condition_weight'].shape) == (1536,16,1,2,2),
              'has_adapter_state': bool(state['adapter']),
              'has_optimizer_state': bool(state.get('optimizer', {}).get('state')),
              'has_scaler_state': 'scaler' in state or 'grad_scaler' in state,
              'has_scheduler_state': 'scheduler' in state,
              'has_rng_state': 'rng' in state,
              'optimizer_state_parameter_count': len(state.get('optimizer', {}).get('state', {}))}
    write_json(ROOT / 'audit/resume_step1000_audit.json', record)
    if not record['has_optimizer_state']:
        write_json(ROOT / 'audit/resume_failure.json',
                   {'reason': 'optimizer state missing', 'checkpoint_path': str(path)})
        print('RESUME_BLOCKED:\noptimizer state missing', flush=True)
        raise SystemExit(2)
    assert record['has_model_state'] and record['has_condition_projection'] and record['has_adapter_state']
    assert state['config'] == initial_config()
    manifest = Path(previous.MANIFEST)
    original = json.loads((ROOT / 'audit/split_audit.json').read_text())
    assert str(manifest) == original['source_manifest']
    train, val = rows('train'), rows('val')
    t_ids, v_ids = {r['video_id'] for r in train}, {r['video_id'] for r in val}
    assert len(train) == 400 and len(t_ids) == 100 and len(val) == 24 and len(v_ids) == 6
    assert len(t_ids & v_ids) == 0
    split = {'manifest_path': str(manifest), 'manifest_sha256': sha256(manifest),
             'original_manifest_path': original['source_manifest'],
             'original_manifest_sha256': sha256(Path(original['source_manifest'])),
             'manifest_sha256_matches_original': True,
             'train_videos': 100, 'train_clips': 400, 'val_videos': 6, 'val_clips': 24,
             'train_val_overlap': 0}
    write_json(ROOT / 'audit/resume_split_audit.json', split)
    return state


def load_restored(with_losses):
    seed_all()
    bundle = load_bundle(with_losses)
    up, sr, adapter, runner, vae, lp = bundle
    state = restore_checkpoint(SOURCE_CKPT, sr, adapter)
    opt = optimizer_for(sr, adapter)
    opt.load_state_dict(state['optimizer'])
    groups = opt.param_groups
    assert len(groups) == 2 and groups[0]['lr'] == 1e-4 and groups[1]['lr'] == 5e-5
    assert all(tuple(g['betas']) == (.9,.99) and g['weight_decay'] == 1e-4 for g in groups)
    assert all(not p.requires_grad for p in up.parameters())
    assert all(not p.requires_grad for name,p in sr.named_parameters()
               if name != 'patch_embedding.condition_weight')
    assert sr.patch_embedding.condition_weight.requires_grad
    assert all(p.requires_grad for p in adapter.parameters())
    assert tuple(INDICES) == (4,9,14,19,24,29)
    if vae is not None:
        assert all(not p.requires_grad for p in vae.parameters())
    write_json(ROOT / 'audit/resumed_optimizer.json',
               {'restored': True, 'groups': [{'lr':g['lr'], 'betas':list(g['betas']),
                 'weight_decay':g['weight_decay'], 'parameter_count':sum(p.numel() for p in g['params'])}
                 for g in groups], 'state_parameter_count':len(opt.state),
                'scheduler':None, 'grad_scaler':None})
    write_json(ROOT / 'audit/pre_resume_integrity.json',
               {'upsampler_frozen':True, 'latent_side_projection_frozen':True,
                'sr_dit_original_backbone_frozen':True, 'wan_frozen':True,
                'condition_projection_trainable':True, 'adapter_trainable':True,
                'adapter_indices':list(INDICES), 'frozen_identity':frozen_identity(up,sr,vae) if vae else None})
    return bundle, opt, state


def equivalence():
    from evaluate import metric_class
    bundle, _, state = load_restored(False)
    up, sr, adapter, runner, _, _ = bundle
    vae = BASE.load_vae()
    metric = metric_class()()
    row = rows('val')[0]
    old = next(r for r in csv.DictReader((ROOT / 'metrics/val_step1000_per_clip.csv').open())
               if r['clip_id'] == row['clip_id'])
    output = CONT / 'audit/resume_equivalence_clip.mp4'
    with torch.inference_mode():
        x, gt = latent_pair(row)
        c = BASE.run_up(up, x)
        pred = runner.refine(c, SEED)
        cpu, target = pred.float().cpu(), gt.float().cpu()
        values = {'latent_mse':float((cpu-target).square().mean()),
                  'latent_mae':float((cpu-target).abs().mean())}
        rgb = vae.decode(pred, device='cuda', tiled=False)
        BASE.write_video(BASE.video_frames(rgb), output)
        values.update(metric(BASE.gt_frames(row).numpy(), BASE.read_video(output)))
    diffs = {key:abs(float(values[key])-float(old[key])) for key in KEYS}
    max_diff = max(diffs.values())
    old_output = ROOT / 'outputs/val/step_1000' / row['video_id'] / f"clip_{row['clip_index']}.mp4"
    old_rgb = BASE.read_video(old_output)
    new_rgb = BASE.read_video(output)
    old_rgb = np.stack(old_rgb).astype(np.float32)
    new_rgb = np.stack(new_rgb).astype(np.float32)
    assert old_rgb.shape == new_rgb.shape == (17,480,864,3)
    max_rgb_diff = float(np.max(np.abs(old_rgb-new_rgb)))
    passed = max_diff <= 1e-4 and max_rgb_diff <= 1.
    record = {'pass':passed, 'clip_id':row['clip_id'], 'metric_diffs':diffs,
              'metric_max_diff':max_diff, 'max_abs_rgb_diff':max_rgb_diff,
              'max_abs_latent_diff':None, 'mean_abs_latent_diff':None,
              'latent_cache_available':False, 'old_rgb_path':str(old_output),
              'new_rgb_path':str(output)}
    write_json(ROOT / 'audit/resume_equivalence.json', record)
    if not passed:
        write_json(ROOT / 'audit/resume_failure.json',
                   {'reason':'step1000 equivalence mismatch', 'details':record})
        raise RuntimeError('Resume equivalence failed')
    return record


def restore_rng(state):
    if 'rng' not in state:
        return False
    value = state['rng']
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    torch.cuda.set_rng_state_all(value['cuda'])
    return True


def grad_norm(parameters):
    return math.sqrt(sum(float(torch.sum(p.grad.detach().float().square()))
                         for p in parameters if p.grad is not None))


def save_new(path, step, sr, adapter, optimizer):
    if path.exists():
        existing = torch.load(path, map_location='cpu', weights_only=False)
        assert existing.get('step') == step, f'Existing checkpoint has wrong step: {path}'
        assert torch.equal(existing['condition_weight'], sr.patch_embedding.condition_weight.detach().cpu())
        current = adapter.state_dict()
        assert set(existing['adapter']) == set(current)
        assert all(torch.equal(existing['adapter'][key], value.detach().cpu())
                   for key,value in current.items()), f'Existing checkpoint differs: {path}'
        return
    save_checkpoint(path, step, sr, adapter, optimizer)


def train():
    bundle, opt, source = load_restored(True)
    up, sr, adapter, runner, vae, lp = bundle
    before = frozen_identity(up,sr,vae)
    source_condition = source['condition_weight'].to(sr.patch_embedding.condition_weight.device)
    source_adapter_hash = hashlib.sha256()
    for key, value in sorted(source['adapter'].items()):
        source_adapter_hash.update(key.encode())
        source_adapter_hash.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    source_adapter_hash = source_adapter_hash.hexdigest()
    latest = CONT / 'runs/latest.pt'
    if latest.exists():
        state = restore_checkpoint(latest,sr,adapter)
        opt.load_state_dict(state['optimizer'])
        assert 1000 < state['step'] <= 3000
        start = state['step'] + 1
    else:
        state = source
        start = 1001
    rng_restored = restore_rng(state)
    write_json(CONT / 'audit/resume_runtime.json',
               {'source_step':1000, 'starting_step':start, 'optimizer_restored':True,
                'grad_scaler_restored':False, 'grad_scaler_used_in_original':False,
                'scheduler_restored':False, 'scheduler_used_in_original':False,
                'rng_state_restored':rng_restored})
    train_rows = rows('train')
    history_path = ROOT / 'metrics/train_history_1000_3000.csv'
    history = list(csv.DictReader(history_path.open())) if history_path.exists() else []
    history = [r for r in history if int(r['step']) < start]
    monitor_path = ROOT / 'audit/gradient_monitor_1000_3000.json'
    monitor = json.loads(monitor_path.read_text()) if monitor_path.exists() else {}
    for step in range(start,3001):
        opt.zero_grad(set_to_none=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            raw, weighted, loss = forward_losses(train_rows[(step-1)%len(train_rows)],bundle,SEED+step)
        assert bool(torch.isfinite(loss)), f'non-finite loss at {step}'
        loss.backward()
        if step in (1001,1500,2000,2500,3000):
            items = dict(adapter.named_parameters())
            embedding = [p for name,p in items.items() if not name.startswith(('control_blocks.','zero_connectors.'))]
            blocks = [p for name,p in items.items() if name.startswith('control_blocks.')]
            connectors = [p for name,p in items.items() if name.startswith('zero_connectors.')]
            monitor[str(step)] = {'condition_projection_grad_norm':grad_norm([sr.patch_embedding.condition_weight]),
                                  'state_aware_embedding_grad_norm':grad_norm(embedding),
                                  'control_block_grad_norm':grad_norm(blocks),
                                  'zero_connector_grad_norm':grad_norm(connectors),
                                  'total_trainable_grad_norm':grad_norm(trainable(sr,adapter)),
                                  'nan_count':0,'inf_count':0}
            assert all(math.isfinite(v) for k,v in monitor[str(step)].items() if k.endswith('norm'))
            write_json(monitor_path,monitor)
        grad = float(torch.nn.utils.clip_grad_norm_(trainable(sr,adapter),1.,error_if_nonfinite=True))
        assert math.isfinite(grad)
        opt.step()
        assert all(bool(torch.isfinite(p).all()) for p in trainable(sr,adapter))
        entry = {'step':step,'total_loss':float(loss.detach()),
                 **{f'{key}_loss':float(value.detach()) for key,value in raw.items()},
                 'projection_lr':opt.param_groups[0]['lr'],'adapter_lr':opt.param_groups[1]['lr'],
                 'grad_norm':grad}
        history.append(entry)
        if step % 10 == 0 or step == start:
            write_csv(history_path,history)
            print(f'train {step}/3000 loss={entry["total_loss"]:.8f} grad={grad:.8f}',flush=True)
        if step % 25 == 0 and step not in MILESTONES:
            save_checkpoint(latest,step,sr,adapter,opt)
        if step in MILESTONES:
            save_new(checkpoint_path(step),step,sr,adapter,opt)
            save_checkpoint(latest,step,sr,adapter,opt)
            assert frozen_identity(up,sr,vae) == before
            print(f'checkpoint step{step}',flush=True)
        del raw,weighted,loss
        gc.collect(); torch.cuda.empty_cache()
    write_csv(history_path,history)
    after = frozen_identity(up,sr,vae)
    integrity = {'upsampler_unchanged':before['upsampler']==after['upsampler'],
                 'latent_side_projection_unchanged':before['latent_projection']==after['latent_projection'],
                 'sr_dit_backbone_unchanged':before['sr_frozen']==after['sr_frozen'],
                 'wan_unchanged':before['wan']==after['wan'],
                 'condition_projection_changed_since_step1000':not torch.equal(source_condition,sr.patch_embedding.condition_weight),
                 'adapter_changed_since_step1000':adapter_checksum(adapter)!=source_adapter_hash,
                 'adapter_indices':list(INDICES),'before':before,'after':after}
    write_json(ROOT / 'audit/final_integrity_step3000.json',integrity)
    assert all(integrity[k] for k in ('upsampler_unchanged','latent_side_projection_unchanged',
        'sr_dit_backbone_unchanged','wan_unchanged',
        'condition_projection_changed_since_step1000','adapter_changed_since_step1000'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('audit','equivalence','train'))
    args=parser.parse_args()
    if args.mode=='audit': audit()
    elif args.mode=='equivalence': equivalence()
    else: train()


if __name__=='__main__': main()
