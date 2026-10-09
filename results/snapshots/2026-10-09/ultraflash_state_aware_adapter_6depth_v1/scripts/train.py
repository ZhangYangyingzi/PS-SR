"""Resume joint condition-projection and six-depth adapter training to a fixed step."""
import argparse
import gc
import math

from common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--until', type=int, choices=STEPS[1:], required=True)
    args = parser.parse_args()
    assert json.loads((ROOT / 'audit/pretrain_pass.json').read_text())['all_required_audits_passed']
    seed_all()
    bundle = load_bundle(True)
    up, sr, adapter, runner, vae, lp = bundle
    before = frozen_identity(up, sr, vae)
    opt = optimizer_for(sr, adapter)
    latest = ROOT / 'runs/latest.pt'
    source = latest if latest.exists() else checkpoint_path(0)
    state = restore_checkpoint(source, sr, adapter)
    initial = torch.load(checkpoint_path(0), map_location='cpu', weights_only=False)
    condition_before = initial['condition_weight'].to(sr.patch_embedding.condition_weight.device)
    initial_hash = hashlib.sha256()
    for key, value in sorted(initial['adapter'].items()):
        initial_hash.update(key.encode())
        initial_hash.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    adapter_before = initial_hash.hexdigest()
    assert state['config'] == initial_config()
    assert state['step'] <= args.until
    if 'optimizer' in state:
        opt.load_state_dict(state['optimizer'])
        rng = state['rng']
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])
    train_rows = rows('train')
    assert len(train_rows) == 400
    history_path = ROOT / 'metrics/train_history.csv'
    history = list(csv.DictReader(history_path.open())) if history_path.exists() else []
    history = [r for r in history if int(r['step']) <= state['step']]
    start = state['step'] + 1
    for step in range(start, args.until + 1):
        opt.zero_grad(set_to_none=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            raw, weighted, loss = forward_losses(train_rows[(step-1) % len(train_rows)], bundle, SEED + step)
        assert torch.isfinite(loss), f'non-finite loss at step {step}'
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(trainable(sr,adapter), 1., error_if_nonfinite=True))
        assert math.isfinite(grad), f'non-finite gradient at step {step}'
        opt.step()
        assert all(bool(torch.isfinite(p).all()) for p in trainable(sr,adapter)), f'non-finite parameter at step {step}'
        entry = {'step': step, 'total_loss': float(loss.detach()),
                 **{f'{key}_loss': float(value.detach()) for key,value in raw.items()},
                 'grad_norm': grad, 'condition_lr': 1e-4, 'adapter_lr': 5e-5}
        history.append(entry)
        if step % 10 == 0 or step == start:
            write_csv(history_path, history)
            print(f'train {step}/1000 loss={entry["total_loss"]:.8f} grad={grad:.8f}', flush=True)
        if step % 25 == 0:
            save_checkpoint(latest, step, sr, adapter, opt)
        if step in STEPS[1:]:
            save_checkpoint(checkpoint_path(step), step, sr, adapter, opt)
            after = frozen_identity(up,sr,vae)
            assert after == before
            write_json(ROOT / 'audit' / f'integrity_step{step:04d}.json',
                       {'step':step, 'before':before, 'after':after,
                        'upsampler_unchanged':True, 'latent_side_projection_unchanged':True,
                        'sr_dit_backbone_unchanged':True, 'wan_unchanged':True,
                        'condition_projection_changed': not torch.equal(condition_before, sr.patch_embedding.condition_weight),
                        'adapter_changed': adapter_checksum(adapter) != adapter_before})
        del raw, weighted, loss
        gc.collect(); torch.cuda.empty_cache()
    if args.until == 1000:
        after = frozen_identity(up,sr,vae)
        result = {'upsampler_unchanged': before['upsampler'] == after['upsampler'],
            'latent_side_projection_unchanged': before['latent_projection'] == after['latent_projection'],
            'sr_dit_backbone_unchanged': before['sr_frozen'] == after['sr_frozen'],
            'wan_unchanged': before['wan'] == after['wan'],
            'condition_projection_changed': not torch.equal(condition_before, sr.patch_embedding.condition_weight),
            'adapter_changed': adapter_checksum(adapter) != adapter_before,
            'before':before, 'after':after}
        write_json(ROOT / 'audit/final_integrity.json', result)
        assert all(result[k] for k in ('upsampler_unchanged','latent_side_projection_unchanged',
            'sr_dit_backbone_unchanged','wan_unchanged','condition_projection_changed','adapter_changed'))


if __name__ == '__main__': main()
