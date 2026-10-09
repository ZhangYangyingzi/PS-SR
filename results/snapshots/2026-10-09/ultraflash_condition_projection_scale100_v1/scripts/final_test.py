"""Run the frozen step-1500 Train100 final test and factual comparisons."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import math
from pathlib import Path

from common import *

STEP = 1500
OLD = PS / 'ultraflash_condition_projection_adaptation_v1'
OLD_MANIFEST = PS / 'latent_sr_v2_multivideo/manifests/clips.csv'
KEYS = ('psnr', 'ssim', 'lpips', 'dists', 'latent_mse', 'latent_mae')
BASELINE = {'psnr': 23.780474, 'ssim': .647606, 'lpips': .201375,
            'dists': .153097, 'latent_mse': None, 'latent_mae': None}


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def preflight():
    ensure()
    history = [{k: int(v) if k == 'step' else float(v) for k, v in row.items()}
               for row in read_csv(ROOT / 'metrics/val_history.csv')]
    assert [r['step'] for r in history] == [0, 500, 1000, 1500, 2000]
    chosen = next(r for r in history if r['step'] == STEP)
    assert chosen['lpips'] == min(r['lpips'] for r in history)
    selection = {'selected_step': STEP, 'selection_metric': 'val_macro_lpips',
                 'selected_val_psnr': 26.554084, 'selected_val_ssim': .624022,
                 'selected_val_lpips': .211276, 'selected_val_dists': .184572,
                 'selected_val_latent_mse': .055172,
                 'selected_val_latent_mae': .177618,
                 'test_seen_before_selection': False}
    for key in KEYS:
        assert abs(chosen[key] - selection['selected_val_' + key]) < 0.000001
    assert not (ROOT / 'metrics/test_per_clip.csv').exists()
    assert not any((ROOT / 'outputs/test').glob('step_*/*/*.mp4'))
    write_json(ROOT / 'metrics/final_selection_before_test.json', selection)

    candidates = [p for p in RUN.glob('*.pt') if p.name == 'step_001500.pt']
    assert len(candidates) == 1
    path = candidates[0].resolve()
    state = torch.load(path, map_location='cpu', weights_only=False)
    weight = state['condition_weight']
    assert state['step'] == STEP and tuple(weight.shape) == (1536, 16, 1, 2, 2)
    assert weight.numel() == 98304 and torch.isfinite(weight).all()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    cp = {'checkpoint_path': str(path), 'checkpoint_size_bytes': path.stat().st_size,
          'checkpoint_sha256': digest.hexdigest(), 'step': state['step'],
          'condition_weight_present': True, 'condition_weight_shape': list(weight.shape),
          'condition_weight_finite': True, 'restorable_shape': True}
    write_json(AUDIT / 'final_test_checkpoint.json', cp)

    groups = {split: rows(split) for split in ('train', 'val', 'test')}
    videos = {split: {r['video_id'] for r in group} for split, group in groups.items()}
    test = groups['test']
    old = [r for r in read_csv(OLD_MANIFEST) if r['split'] == 'test']
    counts = collections.Counter(r['video_id'] for r in test)
    ids = [r['clip_id'] for r in test]
    old_ids = [r['clip_id'] for r in old]
    assert len(groups['train']) == 400 and len(videos['train']) == 100
    assert len(groups['val']) == 24 and len(videos['val']) == 6
    assert len(test) == 24 and len(videos['test']) == 6
    assert all(v == 4 for v in counts.values()) and len(set(ids)) == 24
    assert not videos['train'] & videos['test']
    assert not videos['val'] & videos['test']
    assert not videos['train'] & videos['val']
    assert videos['test'] == {r['video_id'] for r in old}
    assert set(ids) == set(old_ids) and len(old_ids) == len(set(old_ids)) == 24
    assert all((OLD_CACHE / 'test' / f"{r['clip_id']}.pt").is_file() for r in test)
    audit = {'train_videos': 100, 'train_clips': 400, 'val_videos': 6,
             'val_clips': 24, 'test_videos': 6, 'test_clips': 24,
             'clips_per_test_video': dict(sorted(counts.items())),
             'train_test_overlap': 0, 'val_test_overlap': 0,
             'train_val_overlap': 0, 'old_test_manifest': str(OLD_MANIFEST),
             'test_video_ids_identical_to_train24': True,
             'test_clip_ids_identical_to_train24': True,
             'test_clip_ids': ids, 'duplicate_test_clips': 0}
    write_json(AUDIT / 'final_test_split.json', audit)


def generate():
    import gc
    assert (ROOT / 'metrics/final_selection_before_test.json').is_file()
    assert (AUDIT / 'final_test_split.json').is_file()
    seed_all()
    up, sr, _, _ = load_models()
    vae = load_vae()
    before = identity(up, sr, vae)
    original = sr.patch_embedding.condition_weight.detach().clone()
    state = restore(sr, STEP)
    assert state['step'] == STEP
    assert torch.equal(sr.patch_embedding.condition_weight.cpu(), state['condition_weight'])
    after = identity(up, sr, vae)
    changed = not torch.equal(original, sr.patch_embedding.condition_weight)
    prior = json.loads((AUDIT / 'integrity_step1500.json').read_text())['after']
    assert before == after == prior
    names = [n for n, p in sr.named_parameters() if p.requires_grad]
    assert names == ['patch_embedding.condition_weight']
    assert sr.patch_embedding.condition_weight.numel() == 98304 and changed
    integrity = {'checkpoint_step': STEP, 'before': before, 'after': after,
                 'upsampler_unchanged': True,
                 'latent_side_projection_unchanged': True,
                 'attention_unchanged': True, 'ffn_unchanged': True,
                 'norm_unchanged': True, 'sr_dit_backbone_unchanged': True,
                 'wan_unchanged': True,
                 'condition_side_projection_changed': True,
                 'only_adapted_parameter': 'patch_embedding.condition_weight',
                 'trainable_params': 98304,
                 'condition_max_abs_diff': float((sr.patch_embedding.condition_weight - original).abs().max())}
    write_json(AUDIT / 'final_test_integrity.json', integrity)
    for index, row in enumerate(rows('test')):
        output = ROOT / 'outputs/test' / f'step_{STEP:06d}' / row['video_id'] / f"clip_{row['clip_index']}.mp4"
        latent_path = output.with_suffix('.pt')
        assert not output.exists() and not latent_path.exists()
        with torch.inference_mode():
            x, gt = latent_pair(row)
            pred = refine(sr, run_up(up, x), SEED + index)
            assert tuple(pred.shape) == (1, 16, 5, 60, 108) and torch.isfinite(pred).all()
            cpu, target = pred.float().cpu(), gt.float().cpu()
            rgb = vae.decode(pred, device='cuda', tiled=False)
            assert tuple(rgb.shape) == (1, 3, 17, 480, 864) and torch.isfinite(rgb).all()
            write_video(video_frames(rgb), output)
            atomic_save(latent_path, {'z_hr_hat': pred.half().cpu(),
                        'latent_mse': float((cpu - target).square().mean()),
                        'latent_mae': float((cpu - target).abs().mean())})
        print(f'test generate step{STEP}: {index + 1}/24', flush=True)
        del x, gt, pred, rgb, cpu, target
        gc.collect()
        torch.cuda.empty_cache()


def score():
    spec = importlib.util.spec_from_file_location('balanced_metrics', PREVIOUS / 'scripts/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.RUN = RUN
    module.STEPS = STEPS
    module.checkpoint = checkpoint
    assert not (ROOT / 'metrics/test_per_clip.csv').exists()
    expected = rows('test')
    assert len(expected) == 24
    for row in expected:
        output = ROOT / 'outputs/test' / f'step_{STEP:06d}' / row['video_id'] / f"clip_{row['clip_index']}.mp4"
        assert output.is_file() and output.with_suffix('.pt').is_file()
    module.score('test', STEP)
    scored = read_csv(ROOT / 'metrics/test_per_clip.csv')
    assert len(scored) == 24 and {r['clip_id'] for r in scored} == {r['clip_id'] for r in expected}
    assert len({r['clip_id'] for r in scored}) == 24
    assert all(math.isfinite(float(r[k])) for r in scored for k in KEYS)


def comparison_rows(old_rows, new_rows, field, keys, include_latent=False):
    old = {r[field]: r for r in old_rows}
    new = {r[field]: r for r in new_rows}
    assert old.keys() == new.keys()
    metrics = KEYS if include_latent else KEYS[:4]
    out = []
    for ident in sorted(new):
        record = {field: ident}
        for k in metrics:
            a, b = float(old[ident][k]), float(new[ident][k])
            record['train24_' + k] = a
            record['train100_' + k] = b
            record['delta_' + k] = b - a
        out.append(record)
    return out


def summarize():
    new_clips = read_csv(ROOT / 'metrics/test_per_clip.csv')
    new_videos = read_csv(ROOT / 'metrics/test_per_video.csv')
    new_macro = json.loads((ROOT / 'metrics/test_macro.json').read_text())
    old_clips = read_csv(OLD / 'metrics/test_per_clip.csv')
    old_videos = read_csv(OLD / 'metrics/test_per_video.csv')
    old_macro = json.loads((OLD / 'metrics/test_macro.json').read_text())
    assert len(new_clips) == len(old_clips) == 24
    assert len(new_videos) == len(old_videos) == 6
    assert {r['video_id'] for r in new_videos} == {r['video_id'] for r in old_videos}
    assert all(math.isfinite(float(r[k])) for r in new_clips for k in KEYS)
    assert all(math.isfinite(float(new_macro[k])) for k in KEYS)
    by_video = collections.defaultdict(list)
    for r in new_clips:
        by_video[r['video_id']].append(r)
    assert len(by_video) == 6 and all(len(v) == 4 for v in by_video.values())
    for r in new_videos:
        for k in KEYS:
            calc = sum(float(v[k]) for v in by_video[r['video_id']]) / 4
            assert abs(calc - float(r[k])) < 1e-10
    for k in KEYS:
        calc = sum(float(r[k]) for r in new_videos) / 6
        assert abs(calc - float(new_macro[k])) < 1e-10

    comparison = {'step8000_baseline': BASELINE,
                  'train24_condition_projection': old_macro,
                  'train100_condition_projection': new_macro}
    for name, a, b in (('train24_minus_baseline', old_macro, BASELINE),
                       ('train100_minus_baseline', new_macro, BASELINE),
                       ('train100_minus_train24', new_macro, old_macro)):
        comparison[name] = {'delta_' + k: None if b[k] is None else a[k] - b[k] for k in KEYS}
    write_json(ROOT / 'metrics/final_comparison.json', comparison)

    video_comp = comparison_rows(old_videos, new_videos, 'video_id', KEYS)
    clip_comp = comparison_rows(old_clips, new_clips, 'clip_id', KEYS)
    write_csv(ROOT / 'metrics/train100_vs_train24_per_video.csv', video_comp)
    write_csv(ROOT / 'metrics/train100_vs_train24_per_clip.csv', clip_comp)
    def counts(items):
        return {k + '_improved': sum((r['delta_' + k] > 0 if k in ('psnr', 'ssim')
                                      else r['delta_' + k] < 0) for r in items) for k in KEYS[:4]}
    video_counts, clip_counts = counts(video_comp), counts(clip_comp)
    write_json(ROOT / 'metrics/train100_vs_train24_counts.json', video_counts)
    write_json(ROOT / 'metrics/train100_vs_train24_clip_counts.json', clip_counts)
    selection = json.loads((ROOT / 'metrics/final_selection_before_test.json').read_text())
    checkpoint_audit = json.loads((AUDIT / 'final_test_checkpoint.json').read_text())
    split = json.loads((AUDIT / 'final_test_split.json').read_text())
    integrity = json.loads((AUDIT / 'final_test_integrity.json').read_text())
    history = [{k: int(v) if k == 'step' else float(v) for k, v in row.items()}
               for row in read_csv(ROOT / 'metrics/val_history.csv')]
    write_json(ROOT / 'metrics/final_raw_summary.json', {
        'selected_checkpoint': checkpoint_audit['checkpoint_path'],
        'selection': selection, 'val_history': history,
        'train100_test_macro': new_macro, 'train24_test_macro': old_macro,
        'step8000_baseline_test': BASELINE,
        'macro_deltas': {k: comparison[k] for k in
                         ('train24_minus_baseline', 'train100_minus_baseline',
                          'train100_minus_train24')},
        'per_video_improvement_counts': video_counts,
        'per_clip_improvement_counts': clip_counts,
        'split_audit': split,
        'checkpoint_sha256': checkpoint_audit['checkpoint_sha256'],
        'integrity': integrity,
        'completion': {'test_clips': 24, 'test_videos': 6, 'missing_clips': 0,
                       'duplicate_clips': 0, 'nan_metrics': 0, 'inf_metrics': 0,
                       'aggregation': 'clip -> four-clip video mean -> six-video macro'}})

    def metric_lines(values, latent=True):
        for k in KEYS if latent else KEYS[:4]:
            print(('latent MSE' if k == 'latent_mse' else 'latent MAE' if k == 'latent_mae' else k.upper()) + ':', values[k])
    print('FINAL SELECTION')
    print('Selected step:', STEP)
    print('Selection metric:', selection['selection_metric'])
    print('Test unseen before selection:', selection['test_seen_before_selection'])
    print('\nTEST SPLIT')
    print('Videos:', split['test_videos'])
    print('Clips:', split['test_clips'])
    print('Train-Test overlap:', split['train_test_overlap'])
    print('Val-Test overlap:', split['val_test_overlap'])
    print('\nTRAIN100 FINAL TEST')
    metric_lines(new_macro)
    print('\nSTEP8000 BASELINE')
    metric_lines(BASELINE, False)
    print('\nTRAIN24 CONDITION PROJECTION')
    metric_lines(old_macro)
    for title, key, latent in (('TRAIN100 - BASELINE', 'train100_minus_baseline', False),
                               ('TRAIN100 - TRAIN24', 'train100_minus_train24', True)):
        print('\n' + title)
        for k in KEYS if latent else KEYS[:4]:
            label = 'delta latent MSE' if k == 'latent_mse' else 'delta latent MAE' if k == 'latent_mae' else 'delta ' + k.upper()
            print(label + ':', comparison[key]['delta_' + k])
    for title, value in (('PER-VIDEO TRAIN100 VS TRAIN24', video_counts),
                         ('PER-CLIP TRAIN100 VS TRAIN24', clip_counts)):
        print('\n' + title)
        for k in KEYS[:4]: print(k.upper() + ' improved:', value[k + '_improved'])
    print('\nINTEGRITY')
    for label, key in (('Upsampler unchanged', 'upsampler_unchanged'),
                       ('Latent-side projection unchanged', 'latent_side_projection_unchanged'),
                       ('SR DiT backbone unchanged', 'sr_dit_backbone_unchanged'),
                       ('Wan unchanged', 'wan_unchanged'),
                       ('Condition projection changed', 'condition_side_projection_changed'),
                       ('Only adapted parameter', 'only_adapted_parameter')):
        print(label + ':', integrity[key])
    print('\nCheckpoint SHA256:', checkpoint_audit['checkpoint_sha256'])
    print('\nResults directory:', ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('preflight', 'generate', 'score', 'summarize'))
    args = parser.parse_args()
    globals()[args.stage]()
