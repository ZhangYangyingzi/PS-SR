"""Freeze the selected method and deterministically build a disjoint blind split."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT.parent
SOURCE = PS / 'ultraflash_state_aware_adapter_6depth_v1'
POOL = PS / 'latent_sr_v2_multivideo/data_audit/all_videos.csv'
SEED = 20261003
ID = re.compile(r'^[^/]+_[0-9a-f]{8}$')
CLIP = re.compile(r'^(.+_[0-9a-f]{8})_clip_[0-9]+$')
MEDIA = {'.mp4', '.mkv', '.mov', '.y4m', '.avi'}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def path_id(path):
    p = Path(path)
    return f'{p.stem}_{hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:8]}'


def video_ids(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('video_id', 'video_ids', 'selected_video_ids'):
                found |= video_ids(item)
            elif key in ('clip_id', 'clip_ids', 'train_clip_ids', 'val_clip_ids', 'test_clip_ids'):
                found |= video_ids(item)
            elif key in ('absolute_path', 'source_video', 'source_path') and isinstance(item, str):
                if Path(item).suffix.lower() in MEDIA:
                    found.add(path_id(item))
            else:
                found |= video_ids(item)
    elif isinstance(value, list):
        for item in value: found |= video_ids(item)
    elif isinstance(value, str):
        m = CLIP.match(value)
        if m: found.add(m.group(1))
        elif ID.match(value): found.add(value)
    return found


def candidate_files(experiment):
    paths = []
    for folder in (experiment, experiment / 'manifests', experiment / 'manifest',
                   experiment / 'audit', experiment / 'data', experiment / 'metrics'):
        if not folder.is_dir(): continue
        for p in folder.iterdir():
            if not p.is_file() or p.suffix.lower() not in ('.json', '.csv'): continue
            name = p.name.lower()
            if any(token in name for token in ('manifest', 'split', 'clips', 'per_clip')):
                if p.resolve() != POOL.resolve(): paths.append(p)
    runs = experiment / 'runs'
    if runs.is_dir():
        for folder in runs.iterdir():
            if folder.is_dir():
                paths.extend(p for p in folder.iterdir() if p.is_file() and p.suffix == '.csv'
                             and 'per_clip' in p.name.lower())
    return sorted(set(paths))


def inventory():
    rows = []
    used = set()
    for experiment in sorted(PS.iterdir()):
        if not experiment.is_dir() or experiment == ROOT: continue
        for path in candidate_files(experiment):
            try:
                if path.suffix == '.csv':
                    with path.open(newline='') as handle:
                        contents = list(csv.DictReader(handle))
                else:
                    contents = json.loads(path.read_text())
            except (UnicodeError, ValueError, OSError):
                continue
            if isinstance(contents, list) and contents and isinstance(contents[0], dict) and 'split' in contents[0]:
                groups = {}
                for item in contents: groups.setdefault(str(item['split']), []).append(item)
            elif isinstance(contents, dict) and any(k in contents for k in ('train','val','test')):
                groups = {k: contents[k] for k in ('train','val','test') if k in contents}
            else:
                groups = {'unspecified': contents}
            for split, value in groups.items():
                ids = video_ids(value)
                if not ids: continue
                rows.append({'experiment': experiment.name, 'split': split,
                    'source_file': str(path), 'video_ids': sorted(ids), 'video_count': len(ids)})
                used |= ids
    for experiment in ('latent_sr_v2_multivideo', 'ultraflash_zero_shot_multivideo',
            'ultraflash_upsampler_finetune_v1', 'ultraflash_downstream_aware_upsampler_v1',
            'ultraflash_condition_projection_adaptation_v1',
            'ultraflash_condition_projection_scale100_v1',
            'ultraflash_state_aware_adapter_6depth_v1'):
        assert any(r['experiment'] == experiment for r in rows), experiment
    result = {'inventory': rows, 'total_unique_previously_used_videos': len(used),
        'all_previously_used_video_ids': sorted(used)}
    save_json(ROOT / 'audit/previously_used_video_inventory.json', result)
    return used, result


def freeze():
    for folder in ('manifests', 'audit', 'metrics', 'logs', 'decoded', 'comparisons', 'scripts'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    path = SOURCE / 'runs/step_1000.pt'
    up = PS / 'ultraflash_downstream_aware_upsampler_v1/runs/beta_0_continue_8000/best.pt'
    sr = PS.parent / 'UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth'
    wan = PS / 'dependent_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
    for p in (path, up, sr, wan): assert p.is_file(), p
    prior = json.loads((SOURCE / 'audit/final_test_checkpoint.json').read_text())
    assert digest(path) == prior['checkpoint_sha256']
    state = {'method_name': SOURCE.name, 'selected_step': 1000,
        'selection_metric': 'val_macro_lpips', 'adapter_indices': [4,9,14,19,24,29],
        'checkpoint_path': str(path.resolve()), 'checkpoint_size_bytes': path.stat().st_size,
        'checkpoint_sha256': prior['checkpoint_sha256'],
        'upsampler_checkpoint_path': str(up.resolve()), 'upsampler_sha256': digest(up),
        'sr_checkpoint_path': str(sr.resolve()), 'sr_sha256': digest(sr),
        'wan_checkpoint_path': str(wan.resolve()), 'wan_sha256': digest(wan)}
    save_json(ROOT / 'audit/frozen_method_before_blind_test.json', state)


def build_split():
    assert (ROOT / 'audit/frozen_method_before_blind_test.json').is_file()
    used, inv = inventory()
    with POOL.open(newline='') as handle:
        pool = [r for r in csv.DictReader(handle) if r['valid'] == 'True']
    assert len(pool) == 628
    by_id = {path_id(r['absolute_path']): r for r in pool}
    assert len(by_id) == len(pool)
    candidates = sorted(set(by_id) - used)
    random.Random(SEED).shuffle(candidates)
    selected, clip_rows, skipped = [], [], []
    from importlib.util import spec_from_file_location, module_from_spec
    generator_path = PS / 'latent_sr_v2_multivideo/build_manifests.py'
    import sys
    sys.path.insert(0, str(generator_path.parent))
    spec = spec_from_file_location('original_manifest_builder', generator_path)
    module = module_from_spec(spec); spec.loader.exec_module(module)
    for ident in candidates:
        if len(selected) == 24: break
        video = by_id[ident]
        path = Path(video['absolute_path'])
        fps = float(video['fps']); duration = float(video['duration'])
        count = int(video['frame_count'])
        if not path.is_file():
            skipped.append({'video_id': ident, 'reason': 'source_missing'}); continue
        starts = module.clip_starts(duration)
        entries = []
        for index, start in enumerate(starts):
            timestamps = [start + k/17. for k in range(17)]
            indices = [int(round(t*fps)) for t in timestamps]
            if min(indices) < 0 or max(indices) >= count:
                entries = []; break
            entries.append({'video_id': ident, 'dataset': video['dataset'],
                'source_path': str(path), 'source_video': str(path),
                'clip_id': f'{ident}_clip_{index}', 'split': 'blind_final',
                'clip_index': index, 'start_time': f'{start:.9f}',
                'sample_timestamps': json.dumps(timestamps),
                'start_frame': indices[0], 'end_frame': indices[-1],
                'fps': fps, 'source_fps': fps, 'frame_count': 17,
                'source_frame_count': count, 'width': int(video['width']),
                'height': int(video['height'])})
        if len(entries) != 4:
            skipped.append({'video_id': ident, 'reason': 'invalid_frame_range'}); continue
        selected.append(ident); clip_rows.extend(entries)
    assert len(selected) == 24 and len(clip_rows) == 96
    assert not set(selected) & used
    video_manifest = {'seed': SEED, 'candidate_pool_size': len(candidates),
        'excluded_video_count': len(set(by_id) & used), 'selected_video_count': 24,
        'selected_video_ids': selected, 'skipped_invalid_videos': skipped}
    payload = json.dumps(video_manifest, sort_keys=True, separators=(',', ':')).encode()
    video_manifest['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
    video_manifest['sha256_scope'] = 'canonical JSON without manifest_sha256 or sha256_scope'
    save_json(ROOT / 'manifests/blind_final_video_manifest.json', video_manifest)
    clip_path = ROOT / 'manifests/blind_final_clip_manifest.csv'
    with clip_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clip_rows[0]))
        writer.writeheader(); writer.writerows(clip_rows)
    train100 = {r['video_id'] for r in csv.DictReader((PS / 'ultraflash_condition_projection_scale100_v1/data/train100_clips.csv').open()) if r['split']=='train'}
    val6 = {r['video_id'] for r in csv.DictReader((PS / 'ultraflash_condition_projection_scale100_v1/data/train100_clips.csv').open()) if r['split']=='val'}
    test6 = {r['video_id'] for r in csv.DictReader((PS / 'ultraflash_condition_projection_scale100_v1/data/train100_clips.csv').open()) if r['split']=='test'}
    assert not (set(selected)&train100 or set(selected)&val6 or set(selected)&test6 or set(selected)&used)
    counts = Counter(r['video_id'] for r in clip_rows)
    assert all(v==4 for v in counts.values()) and len({r['clip_id'] for r in clip_rows})==96
    audit = {'blind_videos':24, 'blind_clips':96,
        'blind_vs_train100_overlap':0, 'blind_vs_val6_overlap':0,
        'blind_vs_old_test6_overlap':0, 'blind_vs_any_previous_manifest_overlap':0,
        'previous_data_overlap':0, 'clips_per_video':dict(sorted(counts.items())),
        'missing_clips':0, 'duplicate_clips':0,
        'video_manifest_sha256':video_manifest['manifest_sha256'],
        'clip_manifest_sha256':digest(clip_path)}
    save_json(ROOT / 'audit/blindness_audit.json', audit)


if __name__ == '__main__':
    import sys
    assert len(sys.argv)==2 and sys.argv[1] in ('freeze','split')
    globals()[{'freeze':'freeze','split':'build_split'}[sys.argv[1]]]()
