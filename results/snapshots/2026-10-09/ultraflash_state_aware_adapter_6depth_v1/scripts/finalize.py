"""Verify required files and print the requested raw experiment fields only."""
from common import *


def main():
    config = json.loads((ROOT / 'configs/config.json').read_text())
    split = json.loads((ROOT / 'audit/split_audit.json').read_text())
    params = json.loads((ROOT / 'audit/parameter_audit.json').read_text())
    zero = json.loads((ROOT / 'audit/zero_init_equivalence.json').read_text())
    grad = json.loads((ROOT / 'audit/gradient_audit.json').read_text())
    integrity = json.loads((ROOT / 'audit/final_integrity.json').read_text())
    required_audits = ('parameter_audit','requires_grad','zero_init_equivalence',
        'gradient_audit','optimizer_smoke','split_audit','final_integrity')
    assert all((ROOT / 'audit' / f'{name}.json').is_file() for name in required_audits)
    assert all(checkpoint_path(step).is_file() for step in STEPS)
    nan_count = inf_count = 0
    macros = {}
    for step in STEPS:
        clips_path = ROOT / 'metrics' / f'val_step{step:04d}_per_clip.csv'
        videos_path = ROOT / 'metrics' / f'val_step{step:04d}_per_video.csv'
        macro_path = ROOT / 'metrics' / f'val_step{step:04d}_macro.json'
        clips = list(csv.DictReader(clips_path.open()))
        videos = list(csv.DictReader(videos_path.open()))
        macro = json.loads(macro_path.read_text())
        assert len(clips) == 24 and len(videos) == 6
        assert len({r['clip_id'] for r in clips}) == 24
        assert len({r['video_id'] for r in clips}) == 6
        grouped = collections.defaultdict(list)
        for record in clips: grouped[record['video_id']].append(record)
        assert all(len(group) == 4 for group in grouped.values())
        for record in videos:
            for key in KEYS:
                assert abs(float(record[key]) - sum(float(r[key]) for r in grouped[record['video_id']])/4) < 1e-10
        for key in KEYS:
            assert abs(macro[key] - sum(float(r[key]) for r in videos)/6) < 1e-10
        for record in clips:
            for key in KEYS:
                value = float(record[key])
                nan_count += math.isnan(value)
                inf_count += math.isinf(value)
        macros[step] = macro
    assert nan_count == inf_count == 0
    assert len(list(csv.DictReader((ROOT / 'metrics/val_history.csv').open()))) == 5

    print('EXPERIMENT')
    print('Directory:', ROOT)
    print('Train videos:', split['train_videos'])
    print('Train clips:', split['train_clips'])
    print('Val videos:', split['val_videos'])
    print('Val clips:', split['val_clips'])
    print('Test executed:', split['test_executed'])
    print('\nMODEL')
    print('SR DiT blocks:', 30)
    print('Adapter blocks:', len(INDICES))
    print('Adapter indices:', list(INDICES))
    print('Condition projection trainable params:', params['condition_projection_trainable_params'])
    print('Adapter trainable params:', params['adapter_trainable_params'])
    print('Total trainable params:', params['total_trainable_params'])
    print('Trainable ratio:', params['trainable_ratio_vs_sr_dit'])
    print('\nZERO INIT')
    print('max_abs_diff:', zero['max_abs_diff_latent'])
    print('mean_abs_diff:', zero['mean_abs_diff_latent'])
    print('\nGRADIENT AUDIT')
    print('Condition projection grad norm:', grad['condition_projection_grad_norm'])
    print('State-aware embedding grad norm:', grad['state_aware_embedding_grad_norm'])
    print('Control block grad norm:', grad['control_blocks_grad_norm'])
    print('Zero connector grad norm:', grad['zero_connectors_grad_norm'])
    print('Frozen backbone grad:', None if grad['frozen_backbone_grad_none'] else 'present')
    print('Upsampler grad:', None if grad['upsampler_grad_none'] else 'present')
    print('Wan grad:', None if grad['wan_grad_none'] else 'present')
    print('\nVAL HISTORY')
    for step in STEPS:
        print('\nstep' + str(step))
        for key in KEYS:
            label = {'psnr':'PSNR','ssim':'SSIM','lpips':'LPIPS','dists':'DISTS',
                     'latent_mse':'latent MSE','latent_mae':'latent MAE'}[key]
            print(label + ':', macros[step][key])
    print('\nINTEGRITY')
    for label, key in (('Upsampler unchanged','upsampler_unchanged'),
                       ('Latent-side projection unchanged','latent_side_projection_unchanged'),
                       ('SR DiT backbone unchanged','sr_dit_backbone_unchanged'),
                       ('Wan unchanged','wan_unchanged'),
                       ('Condition projection changed','condition_projection_changed'),
                       ('Adapter changed','adapter_changed')):
        print(label + ':', integrity[key])
    print('\nNaN:', nan_count)
    print('Inf:', inf_count)
    print('\nResults directory:', ROOT)


if __name__ == '__main__': main()
