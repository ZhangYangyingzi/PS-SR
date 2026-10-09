"""Frozen methods and shared 96-clip evaluation on the blind split."""
import collections
import csv
import gc
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT.parent
SOURCE = PS / 'ultraflash_state_aware_adapter_6depth_v1'
sys.path.insert(0, str(SOURCE / 'scripts'))
from common import BASE, INDICES, SEED, frozen_identity, load_bundle, restore_checkpoint, seed_all
from evaluate import metric_class

spec = importlib.util.spec_from_file_location('old_video_common', PS / 'latent_sr_v2_multivideo/common.py')
old_video = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_video)

METHODS = ('stateaware_step1000','train100_projection','step8000_baseline','pixel_pssr','bicubic')
KEYS = ('psnr','ssim','lpips','dists','latent_mse','latent_mae')


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def write_csv(path, values):
    values = list(values)
    assert values
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)
    os.replace(tmp, path)


def rows():
    with (ROOT / 'manifests/blind_final_clip_manifest.csv').open(newline='') as handle:
        value = list(csv.DictReader(handle))
    assert len(value)==96 and len({r['clip_id'] for r in value})==96
    assert len({r['video_id'] for r in value})==24
    assert json.loads((ROOT / 'audit/blindness_audit.json').read_text())['previous_data_overlap']==0
    return value


def cache_path(row): return ROOT / 'cache' / f"{row['clip_id']}.pt"
def gt_path(row): return ROOT / 'decoded/gt' / f"{row['clip_id']}.mp4"
def rgb_path(method,row): return ROOT / 'decoded' / method / f"{row['clip_id']}.mp4"


def prepare_cache():
    expected = rows()
    vae = BASE.load_vae()
    for index, row in enumerate(expected):
        cache = cache_path(row)
        bicubic_path = rgb_path('bicubic', row)
        reference = ROOT / 'decoded/gt_tensor' / f"{row['clip_id']}.pt"
        if cache.exists() and bicubic_path.exists() and reference.exists() and gt_path(row).exists():
            continue
        timestamps = json.loads(row['sample_timestamps'])
        source = old_video.sample_nearest_frames(row['source_path'], timestamps)
        hr = [old_video.resize_center_crop(frame) for frame in source]
        lr = [cv2.resize(frame, (288,160), interpolation=cv2.INTER_AREA) for frame in hr]
        bicubic = [cv2.resize(frame, (864,480), interpolation=cv2.INTER_CUBIC) for frame in lr]
        assert len(hr)==len(lr)==len(bicubic)==17
        assert all(x.shape==(480,864,3) for x in hr+bicubic)
        assert all(x.shape==(160,288,3) for x in lr)
        hr_tensor = torch.from_numpy(np.stack(hr)).permute(3,0,1,2).float().div(127.5).sub(1)
        lr_tensor = torch.from_numpy(np.stack(lr)).permute(3,0,1,2).float().div(127.5).sub(1)
        with torch.inference_mode():
            z_hr = vae.encode(hr_tensor.unsqueeze(0).bfloat16(), device='cuda', tiled=False)[0].half().cpu()
            z_lr = vae.encode(lr_tensor.unsqueeze(0).bfloat16(), device='cuda', tiled=False)[0].half().cpu()
        assert tuple(z_hr.shape)==(16,5,60,108) and tuple(z_lr.shape)==(16,5,20,36)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'clip_id':row['clip_id'],'video_id':row['video_id'],
                    'z_hr':z_hr,'z_lr':z_lr}, cache)
        reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(np.stack(hr)), reference)
        old_video.write_video(hr, gt_path(row), fps=17.)
        old_video.write_video(bicubic, bicubic_path, fps=17.)
        print(f'cache {index+1}/96', file=sys.stderr, flush=True)
        del source, hr, lr, bicubic, hr_tensor, lr_tensor, z_hr, z_lr
        gc.collect(); torch.cuda.empty_cache()
    assert all(cache_path(r).is_file() and gt_path(r).is_file() and rgb_path('bicubic',r).is_file() for r in expected)
    save_json(ROOT / 'audit/cache_integrity.json', {'clips':96,'frames_per_clip':17,
        'hr_resolution':[480,864], 'lr_resolution':[160,288],
        'lr_latent_shape':[16,5,20,36], 'hr_latent_shape':[16,5,60,108],
        'all_cached':True,'preprocessing':'same as latent_sr_v2_multivideo/cache_latents.py'})


def current_bundle():
    from final_test import module_hashes
    up, sr, adapter, runner, _, _ = load_bundle(False)
    vae = BASE.load_vae()
    prior = json.loads((SOURCE/'audit/final_integrity.json').read_text())['after']
    before = frozen_identity(up,sr,vae)
    before_modules = module_hashes(up,sr,vae)
    assert before==prior
    state = restore_checkpoint(SOURCE/'runs/step_1000.pt',sr,adapter)
    assert state['step']==1000 and tuple(INDICES)==(4,9,14,19,24,29)
    assert torch.equal(sr.patch_embedding.condition_weight.detach().cpu(),state['condition_weight'])
    assert all(torch.equal(v.detach().cpu(),state['adapter'][k]) for k,v in adapter.state_dict().items())
    after = frozen_identity(up,sr,vae)
    assert after==before==prior and module_hashes(up,sr,vae)==before_modules
    assert all(not p.requires_grad for p in up.parameters())
    assert all(not p.requires_grad for p in vae.parameters())
    assert [n for n,p in sr.named_parameters() if p.requires_grad]==['patch_embedding.condition_weight']
    integrity = {'upsampler_unchanged':True,'latent_side_projection_unchanged':True,
        'self_attention_unchanged':True,'cross_attention_unchanged':True,'ffn_unchanged':True,
        'norm_unchanged':True,'modulation_unchanged':True,'head_unchanged':True,
        'sr_dit_backbone_unchanged':True,'wan_unchanged':True,
        'condition_projection_restored':True,'adapter_restored':True,
        'adapter_indices':list(INDICES),'frozen_identity':after}
    save_json(ROOT/'audit/model_integrity.json',integrity)
    return up,sr,adapter,runner,vae


def run_latent(method):
    assert method in METHODS[:3]
    assert (ROOT/'audit/cache_integrity.json').is_file()
    seed_all()
    if method=='stateaware_step1000':
        up,sr,adapter,runner,vae = current_bundle()
    elif method=='train100_projection':
        from common import previous
        up,sr,_,_ = previous.load_models(False)
        state = previous.restore(sr,1500)
        assert state['step']==1500
        runner = None; vae = BASE.load_vae()
    else:
        up,sr,vae = BASE.load_up(train=False), BASE.load_sr(), BASE.load_vae()
        runner = None
    timing = []
    latent_path = ROOT/'metrics/latent_per_clip.csv'
    latent_records = list(csv.DictReader(latent_path.open(newline=''))) if latent_path.is_file() else []
    for index,row in enumerate(rows()):
        output = rgb_path(method,row)
        if output.exists():
            assert len(BASE.read_video(output))==17
            continue
        cache = torch.load(cache_path(row),map_location='cpu',weights_only=False)
        x=cache['z_lr'].unsqueeze(0).cuda().bfloat16()
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode():
            c=BASE.run_up(up,x)
            torch.cuda.synchronize(); t_up=time.perf_counter()
            pred=runner.refine(c,SEED+index) if runner is not None else BASE.refine(sr,c,SEED+index)
            torch.cuda.synchronize(); t_sr=time.perf_counter()
            rgb=vae.decode(pred,device='cuda',tiled=False)
            torch.cuda.synchronize(); t_decode=time.perf_counter()
        assert tuple(pred.shape)==(1,16,5,60,108) and bool(torch.isfinite(pred).all())
        assert tuple(rgb.shape)==(1,3,17,480,864) and bool(torch.isfinite(rgb).all())
        diff = pred.float().cpu()-cache['z_hr'].unsqueeze(0).float()
        latent_records.append({'method':method,'video_id':row['video_id'],'clip_id':row['clip_id'],
            'latent_mse':float(diff.square().mean()),'latent_mae':float(diff.abs().mean())})
        write_csv(latent_path,latent_records)
        BASE.write_video(BASE.video_frames(rgb),output)
        if method=='stateaware_step1000':
            timing.append({'video_id':row['video_id'],'clip_id':row['clip_id'],
                'upsampler_ms':(t_up-start)*1000,'sr_dit_adapter_ms':(t_sr-t_up)*1000,
                'wan_decode_ms':(t_decode-t_sr)*1000,'total_ms':(t_decode-start)*1000})
            write_csv(ROOT/'metrics/runtime_per_clip.csv',timing)
        print(f'{method} {index+1}/96',file=sys.stderr,flush=True)
        del cache,x,c,pred,rgb
        gc.collect();torch.cuda.empty_cache()
    assert all(rgb_path(method,r).is_file() for r in rows())


def run_pixel():
    expected=rows()
    pixel_python=str(Path('/Huang_group/zyyz/home_dir/.conda/envs/pssr-feature-prior-runtime/bin/python'))
    assert Path(pixel_python).is_file()
    input_dir=ROOT/'decoded/pixel_input'; step1=ROOT/'decoded/pixel_step1'
    final=ROOT/'decoded/pixel_pssr'
    input_dir.mkdir(parents=True,exist_ok=True)
    final.mkdir(parents=True,exist_ok=True)
    for row in expected:
        target=input_dir/f"{row['clip_id']}.mp4"
        if not target.exists(): target.symlink_to(rgb_path('bicubic',row))
    command1=[pixel_python,str(PS/'inference_step_1.py'),'--input_dir',str(input_dir),
        '--output_dir',str(step1),'--lora_base_path',str(PS/'checkpoints/pretrained_models/base.safetensors'),
        '--lora_draft_path',str(PS/'checkpoints/pretrained_models/draft.safetensors'),
        '--wan_model_dir',str(PS/'dependent_models/Wan2.1-T2V-1.3B'),'--sort_files','--skip_existing']
    command2=[pixel_python,str(PS/'inference_step_2.py'),'--consistent_dir',str(step1/'base'),
        '--sharp_dir',str(step1/'base+draft_3'),'--output_dir',str(final),'--sort_files','--skip_existing']
    try:
        subprocess.run(command1,cwd=PS,check=True)
        subprocess.run(command2,cwd=PS,check=True)
        assert all(rgb_path('pixel_pssr',r).is_file() for r in expected)
        for row in expected:
            assert len(BASE.read_video(rgb_path('pixel_pssr',row)))==17
        status={'available':True,'clips':96,'error':None}
    except Exception as exc:
        status={'available':False,'clips':sum(rgb_path('pixel_pssr',r).is_file() for r in expected),
            'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()}
    save_json(ROOT/'audit/pixel_pssr_status.json',status)


def score():
    expected=rows()
    pixel=json.loads((ROOT/'audit/pixel_pssr_status.json').read_text())['available']
    methods=[m for m in METHODS if m!='pixel_pssr' or pixel]
    metric=metric_class()()
    scored_path=ROOT/'metrics/blind_final_per_clip.csv'
    records=list(csv.DictReader(scored_path.open(newline=''))) if scored_path.is_file() else []
    seen={(r['method'],r['clip_id']) for r in records}
    assert len(seen)==len(records)
    for method in methods:
        for index,row in enumerate(expected):
            if (method,row['clip_id']) in seen: continue
            path=rgb_path(method,row)
            assert path.is_file(),path
            gt=torch.load(ROOT/'decoded/gt_tensor'/f"{row['clip_id']}.pt",map_location='cpu',weights_only=True).numpy()
            pred=BASE.read_video(path)
            assert len(gt)==len(pred)==17
            assert all(a.shape==b.shape==(480,864,3) for a,b in zip(gt,pred))
            values=metric(gt,pred)
            latent_mse=latent_mae=None
            if method in METHODS[:3]:
                # Latent output is reproduced by a single deterministic inference only;
                # those values are recorded during generation in a companion CSV.
                with (ROOT/'metrics/latent_per_clip.csv').open(newline='') as handle:
                    latent_map={(r['method'],r['clip_id']):r for r in csv.DictReader(handle)}
                latent=latent_map[(method,row['clip_id'])]
                latent_mse=float(latent['latent_mse']);latent_mae=float(latent['latent_mae'])
            record={'video_id':row['video_id'],'clip_id':row['clip_id'],'dataset':row['dataset'],
                'method':method,**values,'latent_mse':latent_mse,'latent_mae':latent_mae}
            assert all(math.isfinite(float(record[k])) for k in KEYS[:4])
            records.append(record)
            seen.add((method,row['clip_id']))
            write_csv(scored_path,records)
            print(f'score {method} {index+1}/96',file=sys.stderr,flush=True)
    assert len(records)==96*len(methods)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('cache','stateaware_step1000','train100_projection','step8000_baseline','pixel','score')
    stage=sys.argv[1]
    if stage=='cache': prepare_cache()
    elif stage=='pixel': run_pixel()
    elif stage=='score': score()
    else: run_latent(stage)
