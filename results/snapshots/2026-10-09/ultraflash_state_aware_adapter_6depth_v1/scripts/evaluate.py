"""Full Val24 generation and metrics at a fixed checkpoint; Test is inaccessible."""
import argparse
import gc

from common import *


def metric_class():
    path = PS / 'ultraflash_decoder_aware_balanced_loss_v2/scripts/evaluate.py'
    spec = importlib.util.spec_from_file_location('balanced_metric_impl', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PSSR_SITE = BASE.PSSR_SITE
    module.METRICS = BASE.METRICS
    return module.Metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=STEPS, required=True)
    args = parser.parse_args()
    step = args.step
    assert checkpoint_path(step).is_file()
    assert json.loads((ROOT / 'audit/pretrain_pass.json').read_text())['all_required_audits_passed']
    seed_all()
    up, sr, adapter, runner, _, _ = load_bundle(False)
    state = restore_checkpoint(checkpoint_path(step), sr, adapter)
    assert state['step'] == step
    vae = BASE.load_vae()
    metric = metric_class()()
    records = []
    expected = rows('val')
    assert len(expected) == 24
    for index, row in enumerate(expected):
        output = ROOT / 'outputs/val' / f'step_{step:04d}' / row['video_id'] / f"clip_{row['clip_index']}.mp4"
        if output.exists():
            raise RuntimeError(f'Refusing to reuse or overwrite Val output: {output}')
        with torch.inference_mode():
            x, gt = latent_pair(row)
            c = BASE.run_up(up, x)
            pred = runner.refine(c, SEED + index)
            assert tuple(pred.shape) == (1,16,5,60,108) and bool(torch.isfinite(pred).all())
            cpu, target = pred.float().cpu(), gt.float().cpu()
            latent_mse = float((cpu-target).square().mean())
            latent_mae = float((cpu-target).abs().mean())
            rgb = vae.decode(pred, device='cuda', tiled=False)
            assert tuple(rgb.shape) == (1,3,17,480,864) and bool(torch.isfinite(rgb).all())
            BASE.write_video(BASE.video_frames(rgb), output)
            gt_frames = BASE.gt_frames(row).numpy()
            values = metric(gt_frames, BASE.read_video(output))
        record = {'clip_id':row['clip_id'], 'video_id':row['video_id'], 'dataset':row['dataset'],
                  **values, 'latent_mse':latent_mse, 'latent_mae':latent_mae}
        assert all(math.isfinite(float(record[k])) for k in KEYS)
        records.append(record)
        write_csv(ROOT / 'metrics' / f'val_step{step:04d}_per_clip.csv', records)
        print(f'val step{step} {index+1}/24', flush=True)
        del x, gt, c, pred, cpu, target, rgb, gt_frames
        gc.collect(); torch.cuda.empty_cache()
    grouped = collections.defaultdict(list)
    for record in records: grouped[record['video_id']].append(record)
    assert len(grouped) == 6 and all(len(v) == 4 for v in grouped.values())
    videos = [{'video_id': video, **{key: sum(float(r[key]) for r in group)/4 for key in KEYS}}
              for video, group in sorted(grouped.items())]
    macro = {key: sum(row[key] for row in videos)/6 for key in KEYS}
    assert all(math.isfinite(value) for value in macro.values())
    write_csv(ROOT / 'metrics' / f'val_step{step:04d}_per_video.csv', videos)
    write_json(ROOT / 'metrics' / f'val_step{step:04d}_macro.json', macro)
    history_path = ROOT / 'metrics/val_history.csv'
    history = {int(r['step']): r for r in csv.DictReader(history_path.open())} if history_path.exists() else {}
    history[step] = {'step':step, **macro}
    write_csv(history_path, [history[k] for k in sorted(history)])


if __name__ == '__main__': main()
