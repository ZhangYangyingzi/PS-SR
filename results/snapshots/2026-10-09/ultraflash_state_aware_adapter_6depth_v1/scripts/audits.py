"""Required split/model/zero-gradient/one-step audits before training."""
import argparse
import gc
import math
import traceback

from common import *


def prepare():
    for folder in ('configs','scripts','runs','metrics','audit','logs'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    config = initial_config()
    write_json(ROOT / 'configs/config.json', config)
    split_audit()
    print('Prepared:', ROOT, flush=True)


def norm(params):
    return math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in params if p.grad is not None))


def group_grads(adapter):
    groups = {'state_aware_embedding': [], 'control_blocks': [], 'zero_connectors': []}
    for name, parameter in adapter.named_parameters():
        if name.startswith('control_blocks.'):
            groups['control_blocks'].append(parameter)
        elif name.startswith('zero_connectors.'):
            groups['zero_connectors'].append(parameter)
        else:
            groups['state_aware_embedding'].append(parameter)
    return {key: norm(value) for key, value in groups.items()}


def grad_none(model):
    return all(p.grad is None for p in model.parameters())


def pretrain():
    seed_all()
    assert (ROOT / 'configs/config.json').is_file()
    assert (ROOT / 'audit/split_audit.json').is_file()
    up, sr, adapter, runner, _, _ = load_bundle(False)
    vae = BASE.load_vae()
    lp = BASE.load_lpips()
    bundle = (up, sr, adapter, runner, vae, lp)
    before = frozen_identity(up, sr, vae)
    condition_initial = sr.patch_embedding.condition_weight.detach().clone()
    adapter_initial = adapter_checksum(adapter)

    sr_count = sum(p.numel() for p in sr.parameters())
    condition_count = sr.patch_embedding.condition_weight.numel()
    adapter_count = sum(p.numel() for p in adapter.parameters())
    adapter_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    assert adapter_count == adapter_trainable
    assert len(sr.blocks) == 30 and condition_count == 98304
    assert len(adapter.control_blocks) == len(adapter.zero_connectors) == 6
    assert set(int(x) for x in adapter.control_blocks) == set(INDICES)
    assert [name for name,p in sr.named_parameters() if p.requires_grad] == ['patch_embedding.condition_weight']
    parameter_audit = {'total_sr_dit_params': sr_count,
        'frozen_backbone_params': sr_count - condition_count,
        'condition_projection_trainable_params': condition_count,
        'adapter_total_params': adapter_count,
        'adapter_trainable_params': adapter_trainable,
        'total_trainable_params': condition_count + adapter_trainable,
        'trainable_ratio_vs_sr_dit': (condition_count + adapter_trainable) / sr_count,
        'adapter_indices': list(INDICES)}
    write_json(ROOT / 'audit/parameter_audit.json', parameter_audit)

    requires = {'upsampler_frozen': all(not p.requires_grad for p in up.parameters()),
        'latent_projection_frozen': not sr.patch_embedding.latent_weight.requires_grad,
        'condition_projection_trainable': sr.patch_embedding.condition_weight.requires_grad,
        'base_self_attention_frozen': all(not p.requires_grad for b in sr.blocks for p in b.self_attn.parameters()),
        'base_cross_attention_frozen': all(not p.requires_grad for b in sr.blocks for p in b.cross_attn.parameters()),
        'base_ffn_frozen': all(not p.requires_grad for b in sr.blocks for p in b.ffn.parameters()),
        'base_norm_frozen': all(not p.requires_grad for b in sr.blocks for n in (b.norm1,b.norm2,b.norm3) for p in n.parameters()),
        'base_modulation_frozen': all(not b.modulation.requires_grad for b in sr.blocks),
        'base_head_frozen': all(not p.requires_grad for p in sr.head.parameters()),
        'wan_frozen': all(not p.requires_grad for p in vae.parameters()),
        'adapter_trainable': all(p.requires_grad for p in adapter.parameters()),
        'only_allowed_sr_parameter_trainable': [n for n,p in sr.named_parameters() if p.requires_grad] == ['patch_embedding.condition_weight']}
    write_json(ROOT / 'audit/requires_grad.json', requires)
    assert all(requires.values()), f'Requires-grad audit failed: {requires}'

    # Compare the same fixed Val clip, seed, frozen upsampler, and step1500 projection.
    val = rows('val')[0]
    x, _ = latent_pair(val)
    with torch.inference_mode():
        c = BASE.run_up(up, x)
        runner.close()
        original = BASE.refine(sr, c, SEED)
        runner = SideRunner(sr, adapter)
        bundle = (up, sr, adapter, runner, vae, lp)
        adapted = runner.refine(c, SEED)
        diff = (original.float() - adapted.float()).abs()
    zero_audit = {'clip_id': val['clip_id'], 'seed': SEED,
        'baseline': 'frozen step8000 Upsampler + original SR backbone + Train100 step1500 condition projection',
        'max_abs_diff_latent': float(diff.max()), 'mean_abs_diff_latent': float(diff.mean()),
        'exact_equal': bool(torch.equal(original, adapted)),
        'all_connectors_zero': all(int(torch.count_nonzero(m.weight)) == 0 for m in adapter.zero_connectors.values()),
        'output_shape': list(original.shape)}
    write_json(ROOT / 'audit/zero_init_equivalence.json', zero_audit)
    write_json(ROOT / 'audit/adapter_shapes.json', {'shapes': adapter.shape_trace})
    assert zero_audit['exact_equal'] and zero_audit['all_connectors_zero']
    del x, c, original, adapted, diff
    gc.collect(); torch.cuda.empty_cache()

    save_checkpoint(checkpoint_path(0), 0, sr, adapter)
    opt = optimizer_for(sr, adapter)
    train = rows('train')[0]
    opt.zero_grad(set_to_none=True)
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        raw, weighted, loss = forward_losses(train, bundle, SEED+1)
    assert torch.isfinite(loss)
    loss.backward()
    first_grads = group_grads(adapter)
    first_condition_grad = norm([sr.patch_embedding.condition_weight])
    assert first_condition_grad > 0 and first_grads['zero_connectors'] > 0
    assert grad_none(up) and grad_none(vae)
    assert all(p.grad is None for n,p in sr.named_parameters() if n != 'patch_embedding.condition_weight')
    clip_grad = float(torch.nn.utils.clip_grad_norm_(trainable(sr,adapter), 1., error_if_nonfinite=True))
    assert math.isfinite(clip_grad)
    opt.step()
    after_step = frozen_identity(up, sr, vae)
    condition_changed = not torch.equal(condition_initial, sr.patch_embedding.condition_weight)
    adapter_changed = adapter_checksum(adapter) != adapter_initial
    finite = all(torch.isfinite(p).all() for p in trainable(sr,adapter))
    smoke = {'clip_id': train['clip_id'], 'loss': float(loss.detach()),
        'grad_norm_before_clipping': clip_grad, 'loss_finite': True,
        'grad_finite': math.isfinite(clip_grad), 'parameters_finite': bool(finite),
        'condition_projection_changed': condition_changed,
        'adapter_parameters_changed': adapter_changed,
        'base_backbone_unchanged': before == after_step,
        'upsampler_unchanged': before['upsampler'] == after_step['upsampler'],
        'wan_unchanged': before['wan'] == after_step['wan']}
    write_json(ROOT / 'audit/optimizer_smoke.json', smoke)
    assert finite and condition_changed and adapter_changed and before == after_step
    del raw, weighted, loss
    gc.collect(); torch.cuda.empty_cache()

    # A zero connector blocks upstream gradients on its first backward. After the
    # one-step smoke update, a second real-clip backward audits upstream flow.
    opt.zero_grad(set_to_none=True)
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        raw, weighted, loss = forward_losses(train, bundle, SEED+1)
    assert torch.isfinite(loss)
    loss.backward()
    grads = group_grads(adapter)
    condition_grad = norm([sr.patch_embedding.condition_weight])
    frozen_sr_none = all(p.grad is None for n,p in sr.named_parameters() if n != 'patch_embedding.condition_weight')
    gradient = {'clip_id': train['clip_id'], 'measurement': 'real clip backward after isolated one-step connector smoke update',
        'initial_zero_connector_backward': {'condition_projection_grad_norm': first_condition_grad,
                                             **{k+'_grad_norm': v for k,v in first_grads.items()}},
        'condition_projection_grad_norm': condition_grad,
        'state_aware_embedding_grad_norm': grads['state_aware_embedding'],
        'control_blocks_grad_norm': grads['control_blocks'],
        'zero_connectors_grad_norm': grads['zero_connectors'],
        'frozen_backbone_grad_none': frozen_sr_none,
        'upsampler_grad_none': grad_none(up), 'wan_grad_none': grad_none(vae)}
    write_json(ROOT / 'audit/gradient_audit.json', gradient)
    assert condition_grad > 0 and all(v > 0 for v in grads.values())
    assert frozen_sr_none and grad_none(up) and grad_none(vae)

    opt.zero_grad(set_to_none=True)
    restore_checkpoint(checkpoint_path(0), sr, adapter)
    assert torch.equal(condition_initial, sr.patch_embedding.condition_weight)
    assert adapter_checksum(adapter) == adapter_initial
    assert all(int(torch.count_nonzero(m.weight)) == 0 for m in adapter.zero_connectors.values())
    assert frozen_identity(up,sr,vae) == before
    write_json(ROOT / 'audit/pretrain_pass.json', {'all_required_audits_passed': True,
              'smoke_update_reverted': True, 'step0_checkpoint': str(checkpoint_path(0))})
    print('Pre-train audits passed:', ROOT, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','pretrain'))
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except Exception:
        traceback.print_exc()
        write_json(ROOT / 'audit/pretrain_failure.json', {'stage': args.stage, 'traceback': traceback.format_exc()})
        print('AUDIT FAILURE:', args.stage, ROOT / 'audit/pretrain_failure.json', flush=True)
        raise
