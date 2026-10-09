"""One-checkpoint, preselected step-1000 final Test24 evaluation."""
import collections
import csv
import gc
import hashlib
import json
import math
import sys

from common import *
from evaluate import metric_class

STEP = 1000
METRICS_DIR = ROOT / 'metrics'
AUDIT_DIR = ROOT / 'audit'
SOURCE_METRICS = SOURCE / 'metrics'
OLD_MANIFEST = PS / 'latent_sr_v2_multivideo/manifests/clips.csv'
OLD_RESULTS = PS / 'latent_sr_v2_multivideo/results/test_per_clip.csv'
REFERENCE = {
    'adapter_step0': {'psnr': 25.03683322961177, 'ssim': 0.6769797356326652,
        'lpips': 0.18015048638278364, 'dists': 0.15577694466885397,
        'latent_mse': 0.054123218171298504, 'latent_mae': 0.17435261545081934},
    'train24_projection': {'psnr': 24.96228310670445, 'ssim': 0.6716822292118935,
        'lpips': 0.17946517571587772, 'dists': 0.154872219498251,
        'latent_mse': 0.056601271498948336, 'latent_mae': 0.1786820236593485},
    'step8000_baseline': {'psnr': 23.780474, 'ssim': 0.647606,
        'lpips': 0.201375, 'dists': 0.153097},
    'pixel_pssr': {'psnr': 24.98303, 'ssim': 0.7368301,
        'lpips': 0.2105204, 'dists': 0.1355408},
    'bicubic': {'psnr': 31.564989, 'ssim': 0.867143,
        'lpips': 0.214334, 'dists': 0.127346},
}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text())


def test_rows():
    return previous.rows('test')


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_checkpoint():
    matches = [p for p in ROOT.rglob('step_1000.pt') if p.is_file()]
    assert len(matches) == 1, matches
    return matches[0].resolve()


def preflight():
    assert not (METRICS_DIR / 'test_per_clip.csv').exists()
    assert not (METRICS_DIR / 'final_selection_before_test.json').exists()
    assert not (ROOT / 'outputs/test').exists()
    history = [{k: int(v) if k == 'step' else float(v) for k, v in row.items()}
               for row in read_csv(METRICS_DIR / 'val_history.csv')]
    assert [r['step'] for r in history] == [0, 250, 500, 750, 1000]
    val = next(r for r in history if r['step'] == STEP)
    assert val['lpips'] == min(r['lpips'] for r in history)
    selection = {'selected_step': STEP, 'selection_metric': 'val_macro_lpips',
        **{'selected_val_' + k: val[k] for k in KEYS},
        'test_used_for_current_checkpoint_selection': False,
        'test_split_seen_in_prior_experiments': True,
        'current_step1000_selected_using_test': False}
    write_json(METRICS_DIR / 'final_selection_before_test.json', selection)

    path = selected_checkpoint()
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['step'] == STEP
    condition = state['condition_weight']
    adapter = state['adapter']
    assert tuple(condition.shape) == (1536, 16, 1, 2, 2)
    assert torch.isfinite(condition).all()
    assert isinstance(adapter, dict) and all(torch.isfinite(t).all() for t in adapter.values())
    assert any(k.startswith('base_proj.') for k in adapter)
    assert any(k.startswith('dynamic_proj.') for k in adapter)
    assert any(k.startswith('query.') for k in adapter)
    assert any(k.startswith('zero_out.') for k in adapter)
    assert {int(k.split('.')[1]) for k in adapter if k.startswith('control_blocks.')} == set(INDICES)
    assert {int(k.split('.')[1]) for k in adapter if k.startswith('zero_connectors.')} == set(INDICES)
    assert list(state['config']['adapter_indices']) == list(INDICES)
    checkpoint = {'checkpoint_path': str(path), 'checkpoint_size_bytes': path.stat().st_size,
        'checkpoint_sha256': hash_file(path), 'step': STEP,
        'condition_weight_present': True, 'condition_weight_shape': list(condition.shape),
        'adapter_weight_count': len(adapter), 'state_aware_embedding_present': True,
        'state_aware_fusion_present': True, 'control_blocks_present': True,
        'zero_connectors_present': True, 'adapter_indices': list(INDICES),
        'all_weights_finite': True}
    write_json(AUDIT_DIR / 'final_test_checkpoint.json', checkpoint)

    groups = {s: previous.rows(s) for s in ('train', 'val', 'test')}
    vids = {s: {r['video_id'] for r in group} for s, group in groups.items()}
    clips = [r['clip_id'] for r in groups['test']]
    old = [r for r in read_csv(OLD_MANIFEST) if r['split'] == 'test']
    old_scored = read_csv(SOURCE_METRICS / 'test_per_clip.csv')
    pixel = [r for r in read_csv(OLD_RESULTS) if r['method'] == 'pixel_pssr']
    bicubic = [r for r in read_csv(OLD_RESULTS) if r['method'] == 'bicubic']
    counts = collections.Counter(r['video_id'] for r in groups['test'])
    assert len(groups['train']) == 400 and len(vids['train']) == 100
    assert len(groups['val']) == 24 and len(vids['val']) == 6
    assert len(groups['test']) == 24 and len(vids['test']) == 6
    assert len(set(clips)) == 24 and all(v == 4 for v in counts.values())
    assert not (vids['train'] & vids['test']) and not (vids['val'] & vids['test'])
    assert {r['video_id'] for r in old} == vids['test']
    for rows_ in (old, old_scored, pixel, bicubic):
        assert len(rows_) == 24 and {r['clip_id'] for r in rows_} == set(clips)
    assert all((previous.OLD_CACHE / 'test' / f'{clip}.pt').is_file() for clip in clips)
    split = {'source_manifest': str(previous.MANIFEST), 'old_manifest': str(OLD_MANIFEST),
        'train_videos': 100, 'train_clips': 400, 'val_videos': 6, 'val_clips': 24,
        'test_videos': 6, 'test_clips': 24, 'clips_per_video': dict(sorted(counts.items())),
        'train_test_overlap': 0, 'val_test_overlap': 0,
        'video_id_set_identical': True, 'clip_id_set_identical': True,
        'pixel_pssr_same_test24': True, 'bicubic_same_test24': True,
        'test_clip_ids': clips, 'duplicate_clips': 0}
    write_json(AUDIT_DIR / 'final_test_split.json', split)


def module_hashes(up, sr, vae):
    return {'upsampler': BASE.tensor_checksum(up),
        'latent_side_projection': hashlib.sha256(sr.patch_embedding.latent_weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        'self_attention': [BASE.tensor_checksum(b.self_attn) for b in sr.blocks],
        'cross_attention': [BASE.tensor_checksum(b.cross_attn) for b in sr.blocks],
        'ffn': [BASE.tensor_checksum(b.ffn) for b in sr.blocks],
        'norm': [[BASE.tensor_checksum(n) for n in (b.norm1, b.norm2, b.norm3)] for b in sr.blocks],
        'modulation': [hashlib.sha256(b.modulation.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for b in sr.blocks],
        'head': BASE.tensor_checksum(sr.head), 'wan': BASE.tensor_checksum(vae),
        'whole_sr_backbone': previous.frozen_hash(sr)}


def run_test():
    assert read_json(METRICS_DIR / 'final_selection_before_test.json')['selected_step'] == STEP
    assert read_json(AUDIT_DIR / 'final_test_checkpoint.json')['checkpoint_sha256'] == hash_file(selected_checkpoint())
    expected = test_rows()
    seed_all()
    up, sr, adapter, runner, _, _ = load_bundle(False)
    vae = BASE.load_vae()
    prior_integrity = read_json(AUDIT_DIR / 'final_integrity.json')['after']
    before = frozen_identity(up, sr, vae)
    assert before == prior_integrity
    before_modules = module_hashes(up, sr, vae)
    initial_condition = sr.patch_embedding.condition_weight.detach().clone()
    state = restore_checkpoint(selected_checkpoint(), sr, adapter)
    assert state['step'] == STEP
    assert torch.equal(sr.patch_embedding.condition_weight.detach().cpu(), state['condition_weight'])
    assert all(torch.equal(v.detach().cpu(), state['adapter'][k]) for k, v in adapter.state_dict().items())
    assert tuple(sorted(int(x) for x in adapter.control_blocks)) == INDICES
    after = frozen_identity(up, sr, vae)
    after_modules = module_hashes(up, sr, vae)
    assert before == after == prior_integrity and before_modules == after_modules
    assert not torch.equal(initial_condition, sr.patch_embedding.condition_weight)
    assert all(not p.requires_grad for p in up.parameters())
    assert all(not p.requires_grad for p in vae.parameters())
    assert [n for n, p in sr.named_parameters() if p.requires_grad] == ['patch_embedding.condition_weight']
    integrity = {'checkpoint_step': STEP, 'before': before, 'after': after,
        'upsampler_unchanged': True, 'latent_side_projection_unchanged': True,
        'self_attention_unchanged': True, 'cross_attention_unchanged': True,
        'ffn_unchanged': True, 'norm_unchanged': True, 'modulation_unchanged': True,
        'head_unchanged': True, 'sr_dit_backbone_unchanged': True, 'wan_unchanged': True,
        'condition_projection_changed': True, 'condition_projection_restored': True,
        'adapter_restored': True, 'adapter_indices': list(INDICES),
        'module_hashes': after_modules}
    write_json(AUDIT_DIR / 'final_test_integrity.json', integrity)
    metric = metric_class()()
    records = []
    for index, row in enumerate(expected):
        output = ROOT / 'outputs/test/step_1000' / row['video_id'] / f"clip_{row['clip_index']}.mp4"
        assert not output.exists(), output
        with torch.inference_mode():
            x, gt = latent_pair(row)
            c = BASE.run_up(up, x)
            pred = runner.refine(c, SEED + index)
            assert tuple(pred.shape) == (1, 16, 5, 60, 108) and bool(torch.isfinite(pred).all())
            cpu, target = pred.float().cpu(), gt.float().cpu()
            latent_mse = float((cpu-target).square().mean())
            latent_mae = float((cpu-target).abs().mean())
            rgb = vae.decode(pred, device='cuda', tiled=False)
            assert tuple(rgb.shape) == (1, 3, 17, 480, 864) and bool(torch.isfinite(rgb).all())
            BASE.write_video(BASE.video_frames(rgb), output)
            gt_frames = BASE.gt_frames(row).numpy()
            decoded = BASE.read_video(output)
            assert len(gt_frames) == len(decoded) == 17
            assert all(a.shape == b.shape == (480, 864, 3) for a, b in zip(gt_frames, decoded))
            values = metric(gt_frames, decoded)
        record = {'clip_id': row['clip_id'], 'video_id': row['video_id'], 'dataset': row['dataset'],
                  **values, 'latent_mse': latent_mse, 'latent_mae': latent_mae}
        assert all(math.isfinite(float(record[k])) for k in KEYS)
        records.append(record)
        write_csv(METRICS_DIR / 'test_per_clip.csv', records)
        print(f'test step1000 {index+1}/24', file=sys.stderr, flush=True)
        del x, gt, c, pred, cpu, target, rgb, gt_frames, decoded
        gc.collect(); torch.cuda.empty_cache()
    assert len(records) == len(expected) == 24
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record['video_id']].append(record)
    assert len(grouped) == 6 and all(len(v) == 4 for v in grouped.values())
    videos = [{'video_id': video, **{k: sum(float(r[k]) for r in group)/4 for k in KEYS}}
              for video, group in sorted(grouped.items())]
    macro = {k: sum(float(r[k]) for r in videos)/6 for k in KEYS}
    assert all(math.isfinite(v) for v in macro.values())
    write_csv(METRICS_DIR / 'test_per_video.csv', videos)
    write_json(METRICS_DIR / 'test_macro.json', macro)


def comparison_rows(old, new, id_key):
    a = {r[id_key]: r for r in old}
    b = {r[id_key]: r for r in new}
    assert a.keys() == b.keys()
    result = []
    for ident in sorted(a):
        row = {id_key: ident}
        for k in KEYS:
            x, y = float(a[ident][k]), float(b[ident][k])
            row['step0_' + k] = x
            row['adapter_' + k] = y
            row['delta_' + k] = y-x
        result.append(row)
    return result


def improved_counts(items):
    return {k + '_improved': sum(r['delta_' + k] > 0 if k in ('psnr', 'ssim')
                                else r['delta_' + k] < 0 for r in items) for k in KEYS}


def summarize():
    clips = read_csv(METRICS_DIR / 'test_per_clip.csv')
    videos = read_csv(METRICS_DIR / 'test_per_video.csv')
    macro = read_json(METRICS_DIR / 'test_macro.json')
    expected = test_rows()
    assert len(clips) == 24 and len(videos) == 6
    assert len({r['clip_id'] for r in clips}) == 24
    assert {r['clip_id'] for r in clips} == {r['clip_id'] for r in expected}
    grouped = collections.defaultdict(list)
    for row in clips: grouped[row['video_id']].append(row)
    assert len(grouped) == 6 and all(len(v) == 4 for v in grouped.values())
    for row in videos:
        for k in KEYS:
            assert abs(sum(float(v[k]) for v in grouped[row['video_id']])/4 - float(row[k])) < 1e-10
    for k in KEYS:
        assert abs(sum(float(v[k]) for v in videos)/6 - macro[k]) < 1e-10
    assert all(math.isfinite(float(r[k])) for r in clips for k in KEYS)
    old_clips = read_csv(SOURCE_METRICS / 'test_per_clip.csv')
    old_videos = read_csv(SOURCE_METRICS / 'test_per_video.csv')
    old_macro = read_json(SOURCE_METRICS / 'test_macro.json')
    for k in KEYS:
        assert abs(old_macro[k]-REFERENCE['adapter_step0'][k]) < 1e-12
    vcmp = comparison_rows(old_videos, videos, 'video_id')
    ccmp = comparison_rows(old_clips, clips, 'clip_id')
    vcounts, ccounts = improved_counts(vcmp), improved_counts(ccmp)
    write_csv(METRICS_DIR / 'adapter_vs_step0_per_video.csv', vcmp)
    write_json(METRICS_DIR / 'adapter_vs_step0_video_counts.json', vcounts)
    write_csv(METRICS_DIR / 'adapter_vs_step0_per_clip.csv', ccmp)
    write_json(METRICS_DIR / 'adapter_vs_step0_clip_counts.json', ccounts)
    reference = dict(REFERENCE)
    reference['adapter_step0'] = old_macro
    comparison = {'references': reference, 'adapter_step1000': macro}
    for name, base in reference.items():
        comparison['adapter_step1000_minus_' + name] = {
            'delta_' + k: macro[k]-base[k] if k in base else None for k in KEYS}
    write_json(METRICS_DIR / 'final_comparison.json', comparison)
    val0 = read_json(METRICS_DIR / 'val_step0000_macro.json')
    val1000 = read_json(METRICS_DIR / 'val_step1000_macro.json')
    val_test = {'step0_val': val0, 'step1000_val': val1000,
        'step0_test': old_macro, 'step1000_test': macro,
        'val_improvement_step1000_minus_step0': {'delta_'+k: val1000[k]-val0[k] for k in KEYS},
        'test_improvement_step1000_minus_step0': {'delta_'+k: macro[k]-old_macro[k] for k in KEYS}}
    write_json(METRICS_DIR / 'val_test_summary.json', val_test)
    checkpoint = read_json(AUDIT_DIR / 'final_test_checkpoint.json')
    split = read_json(AUDIT_DIR / 'final_test_split.json')
    integrity = read_json(AUDIT_DIR / 'final_test_integrity.json')
    selection = read_json(METRICS_DIR / 'final_selection_before_test.json')
    summary = {'selected_checkpoint': checkpoint['checkpoint_path'],
        'selection': selection, 'val_step0': val0, 'val_step1000': val1000,
        'test_step0_reference': old_macro, 'test_step1000': macro,
        'macro_deltas': {k:v for k,v in comparison.items() if k.startswith('adapter_step1000_minus_')},
        'per_video_improvement_counts': vcounts, 'per_clip_improvement_counts': ccounts,
        'split_audit': split, 'checkpoint_path': checkpoint['checkpoint_path'],
        'checkpoint_sha256': checkpoint['checkpoint_sha256'], 'integrity': integrity,
        'completion': {'test_videos': 6, 'test_clips': 24, 'missing_clips': 0,
            'duplicate_clips': 0, 'nan_metrics': 0, 'inf_metrics': 0,
            'aggregation': 'clip -> 4-clip video mean -> 6-video macro',
            'rgb_frames': 17, 'rgb_resolution': [480,864], 'gt_preprocessing': 'same as Val'}}
    write_json(METRICS_DIR / 'final_raw_summary.json', summary)
    return summary, comparison


def print_final(summary, comparison):
    selection = summary['selection']; split = summary['split_audit']
    val = summary['val_step1000']; test = summary['test_step1000']
    old = summary['test_step0_reference']; integrity = summary['integrity']
    def metrics(v, latent=True):
        for k in KEYS if latent else KEYS[:4]:
            label = {'latent_mse':'latent MSE','latent_mae':'latent MAE'}.get(k,k.upper())
            print(f'{label}: {v[k]}')
    print('FINAL SELECTION')
    print('Selected step:', selection['selected_step'])
    print('Selection metric:', selection['selection_metric'])
    print('Current checkpoint selected using Test:', selection['current_step1000_selected_using_test'])
    print('\nTEST SPLIT')
    print('Videos:', split['test_videos']); print('Clips:', split['test_clips'])
    print('Train-Test overlap:', split['train_test_overlap'])
    print('Val-Test overlap:', split['val_test_overlap'])
    print('\nVAL STEP1000'); metrics(val)
    print('\nTEST STEP1000'); metrics(test)
    print('\nTEST STEP0 REFERENCE'); metrics(old)
    print('\nSTEP1000 - STEP0')
    for k in KEYS:
        label = {'latent_mse':'latent MSE','latent_mae':'latent MAE'}.get(k,k.upper())
        print(f'delta {label}: {test[k]-old[k]}')
    for title, key in (('TRAIN24 PROJECTION','train24_projection'),
                       ('STEP8000 BASELINE','step8000_baseline'),
                       ('PIXEL PS-SR','pixel_pssr'), ('BICUBIC','bicubic')):
        print('\n'+title); metrics(REFERENCE[key], latent=False)
    for title, key in (('PER-VIDEO STEP1000 VS STEP0','per_video_improvement_counts'),
                       ('PER-CLIP STEP1000 VS STEP0','per_clip_improvement_counts')):
        print('\n'+title)
        for k in KEYS:
            label = {'latent_mse':'latent MSE','latent_mae':'latent MAE'}.get(k,k.upper())
            print(f'{label} improved: {summary[key][k+"_improved"]}')
    print('\nINTEGRITY')
    for label, key in (('Upsampler unchanged','upsampler_unchanged'),
                       ('Latent-side projection unchanged','latent_side_projection_unchanged'),
                       ('SR DiT backbone unchanged','sr_dit_backbone_unchanged'),
                       ('Wan unchanged','wan_unchanged'),
                       ('Condition projection restored','condition_projection_restored'),
                       ('Adapter restored','adapter_restored'), ('Adapter indices','adapter_indices')):
        print(f'{label}: {integrity[key]}')
    print('\nNaN:', summary['completion']['nan_metrics'])
    print('Inf:', summary['completion']['inf_metrics'])
    print('\nCheckpoint SHA256:', summary['checkpoint_sha256'])
    print('Results directory:', ROOT)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('preflight','run_test','summarize')
    if sys.argv[1] == 'preflight': preflight()
    elif sys.argv[1] == 'run_test': run_test()
    else:
        summary, comparison = summarize()
        print_final(summary, comparison)
