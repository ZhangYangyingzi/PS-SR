"""Blind split aggregation, factual comparisons, visuals, and terminal report."""
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import cv2
import imageio
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=('stateaware_step1000','train100_projection','step8000_baseline','pixel_pssr','bicubic')
KEYS=('psnr','ssim','lpips','dists','latent_mse','latent_mae')


def load(path): return json.loads(path.read_text())
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_rows(path):
    with path.open(newline='') as handle: return list(csv.DictReader(handle))
def write_csv(path,items):
    assert items
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(items[0]))
        writer.writeheader();writer.writerows(items)


def aggregate():
    clips=csv_rows(ROOT/'metrics/blind_final_per_clip.csv')
    methods=[m for m in METHODS if any(r['method']==m for r in clips)]
    expected={r['clip_id'] for r in csv_rows(ROOT/'manifests/blind_final_clip_manifest.csv')}
    assert len(expected)==96
    for method in methods:
        ids=[r['clip_id'] for r in clips if r['method']==method]
        assert len(ids)==len(set(ids))==96 and set(ids)==expected
    assert len({r['clip_id'] for r in clips})==96
    grouped=collections.defaultdict(list)
    for r in clips:
        assert all(math.isfinite(float(r[k])) for k in KEYS[:4])
        grouped[(r['method'],r['video_id'])].append(r)
    assert all(len(v)==4 for v in grouped.values())
    videos=[]
    for (method,video),items in sorted(grouped.items()):
        record={'video_id':video,'dataset':items[0]['dataset'],'method':method}
        for k in KEYS:
            record[k]=sum(float(r[k]) for r in items)/4 if items[0][k] not in ('',None) else None
        videos.append(record)
    write_csv(ROOT/'metrics/blind_final_per_video.csv',videos)
    macro={}
    for method in methods:
        group=[r for r in videos if r['method']==method]
        assert len(group)==24
        macro[method]={k:sum(float(r[k]) for r in group)/24 if group[0][k] is not None else None for k in KEYS}
    if 'pixel_pssr' not in macro: macro['pixel_pssr']=None
    save(ROOT/'metrics/blind_final_macro.json',macro)
    by_dataset={}
    for method in methods:
        groups=collections.defaultdict(list)
        for r in videos:
            if r['method']==method: groups[r['dataset']].append(r)
        by_dataset[method]={ds:{'videos':len(g),**{k:sum(float(r[k]) for r in g)/len(g) for k in KEYS[:4]}}
                            for ds,g in sorted(groups.items())}
    save(ROOT/'metrics/blind_final_by_dataset.json',by_dataset)
    return clips,videos,macro


def comparison_rows(items,ident):
    current={r[ident]:r for r in items if r['method']=='stateaware_step1000'}
    prior={r[ident]:r for r in items if r['method']=='train100_projection'}
    assert len(current)==len(prior) and current.keys()==prior.keys()
    result=[]
    for key in sorted(current):
        item={ident:key}
        for k in KEYS:
            a,b=float(prior[key][k]),float(current[key][k])
            item['projection_'+k]=a;item['stateaware_'+k]=b;item['delta_'+k]=b-a
        result.append(item)
    return result


def counts(rows):
    return {k+'_improved':sum(r['delta_'+k]>0 if k in ('psnr','ssim') else r['delta_'+k]<0 for r in rows)
            for k in KEYS}


def runtime():
    items=csv_rows(ROOT/'metrics/runtime_per_clip.csv')
    assert len(items)==96 and len({r['clip_id'] for r in items})==96
    result={}
    for key in ('upsampler_ms','sr_dit_adapter_ms','wan_decode_ms','total_ms'):
        values=np.asarray([float(r[key]) for r in items],dtype=np.float64)
        assert np.isfinite(values).all()
        result[key]={'mean':float(values.mean()),'median':float(np.median(values)),
                     'std':float(values.std(ddof=0)),'p90':float(np.percentile(values,90))}
    save(ROOT/'metrics/runtime_summary.json',result)
    return result


def summarize():
    clips,videos,macro=aggregate()
    video_rows=comparison_rows(videos,'video_id')
    clip_rows=comparison_rows(clips,'clip_id')
    assert len(video_rows)==24 and len(clip_rows)==96
    vcounts,ccounts=counts(video_rows),counts(clip_rows)
    write_csv(ROOT/'metrics/stateaware_vs_projection_per_video.csv',video_rows)
    write_csv(ROOT/'metrics/stateaware_vs_projection_per_clip.csv',clip_rows)
    save(ROOT/'metrics/stateaware_vs_projection_video_counts.json',vcounts)
    save(ROOT/'metrics/stateaware_vs_projection_clip_counts.json',ccounts)
    deltas={}
    for method in METHODS[1:]:
        baseline=macro.get(method)
        deltas['stateaware_minus_'+method]={
            'delta_'+k:macro['stateaware_step1000'][k]-baseline[k] if baseline is not None and baseline[k] is not None else None
            for k in KEYS}
    save(ROOT/'metrics/blind_final_comparisons.json',deltas)
    timing=runtime()
    blind=load(ROOT/'audit/blindness_audit.json')
    frozen=load(ROOT/'audit/frozen_method_before_blind_test.json')
    integrity=load(ROOT/'audit/model_integrity.json')
    video_manifest=load(ROOT/'manifests/blind_final_video_manifest.json')
    video_manifest_file_sha256=hashlib.sha256(
        (ROOT/'manifests/blind_final_video_manifest.json').read_bytes()).hexdigest()
    with (ROOT/'manifests/blind_final_clip_manifest.csv').open(newline='') as handle:
        manifest=list(csv.DictReader(handle))
    completion={'blind_videos':24,'blind_clips':96,'previous_data_overlap':blind['previous_data_overlap'],
        'nan_metrics':0,'inf_metrics':0,'missing_clips':0,'duplicate_clips':0,
        'clips_per_video':4,'frames_per_clip':17,
        'aggregation':'clip -> 4-clip video mean -> 24-video macro'}
    summary={'blind_manifest_sha256':video_manifest_file_sha256,
        'canonical_selection_sha256':video_manifest['manifest_sha256'],
        'clip_manifest_sha256':blind['clip_manifest_sha256'],
        'model_checkpoint_sha256':frozen['checkpoint_sha256'],
        'selected_video_ids':video_manifest['selected_video_ids'],
        'clip_ids':[r['clip_id'] for r in manifest],
        'blindness_audit':blind,'method_macros':macro,'macro_deltas':deltas,
        'per_video_improvement_counts':vcounts,'per_clip_improvement_counts':ccounts,
        'runtime_summary':timing,'integrity':integrity,'completion':completion,
        'pixel_pssr_status':load(ROOT/'audit/pixel_pssr_status.json')}
    save(ROOT/'metrics/final_raw_summary.json',summary)
    return summary


def frames(path):
    cap=cv2.VideoCapture(str(path)); result=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        result.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(result)==17 and all(f.shape==(480,864,3) for f in result),path
    return result


def visuals():
    manifest=csv_rows(ROOT/'manifests/blind_final_clip_manifest.csv')
    pixel=load(ROOT/'audit/pixel_pssr_status.json')['available']
    first=[r for r in manifest if int(r['clip_index'])==0]
    assert len(first)==24
    for row in first:
        ident=row['clip_id'];video=row['video_id']
        methods=['gt','bicubic']+(['pixel_pssr'] if pixel else [])+['train100_projection','stateaware_step1000']
        sources=[frames(ROOT/'decoded'/method/f'{ident}.mp4') for method in methods]
        output=ROOT/'comparisons'/f'{video}.mp4'
        writer=imageio.get_writer(str(output),fps=17,quality=8,macro_block_size=1)
        try:
            for index in range(17): writer.append_data(np.concatenate([f[index] for f in sources],axis=1))
        finally: writer.close()
    assert len(list((ROOT/'comparisons').glob('*.mp4')))==24


def print_final(summary):
    inv=load(ROOT/'audit/previously_used_video_inventory.json')
    videos=load(ROOT/'manifests/blind_final_video_manifest.json')
    macro=summary['method_macros'];delta=summary['macro_deltas']['stateaware_minus_train100_projection']
    integrity=summary['integrity']; completion=summary['completion']
    def metrics(value,latent=False):
        if value is None:value={k:None for k in KEYS}
        for key in KEYS if latent else KEYS[:4]:
            name={'latent_mse':'latent MSE','latent_mae':'latent MAE'}.get(key,key.upper())
            print(f'{name}: {value[key]}')
    print('BLIND FINAL SPLIT')
    print('Previously used unique videos:',inv['total_unique_previously_used_videos'])
    print('Unseen candidate videos:',videos['candidate_pool_size'])
    print('Selected videos:',videos['selected_video_count'])
    print('Selected clips:',completion['blind_clips'])
    print('Manifest SHA256:',summary['blind_manifest_sha256'])
    print('Previous-data overlap:',completion['previous_data_overlap'])
    print('\nCURRENT METHOD')
    frozen=load(ROOT/'audit/frozen_method_before_blind_test.json')
    print('Checkpoint:',frozen['checkpoint_path'])
    print('Checkpoint SHA256:',frozen['checkpoint_sha256'])
    print('\nBLIND FINAL MACRO')
    for title,method,latent in (('STATE-AWARE STEP1000','stateaware_step1000',True),
        ('TRAIN100 PROJECTION','train100_projection',True),('STEP8000 BASELINE','step8000_baseline',False),
        ('PIXEL PS-SR','pixel_pssr',False),('BICUBIC','bicubic',False)):
        print('\n'+title);metrics(macro[method],latent)
    print('\nSTATE-AWARE - PROJECTION')
    for key in KEYS:
        name={'latent_mse':'latent MSE','latent_mae':'latent MAE'}.get(key,key.upper())
        print(f'delta {name}: {delta["delta_"+key]}')
    for title,key in (('PER-VIDEO STATE-AWARE VS PROJECTION','per_video_improvement_counts'),
                      ('PER-CLIP STATE-AWARE VS PROJECTION','per_clip_improvement_counts')):
        print('\n'+title)
        for metric in KEYS:
            name={'latent_mse':'latent MSE','latent_mae':'latent MAE'}.get(metric,metric.upper())
            print(f'{name} improved: {summary[key][metric+"_improved"]}')
    timing=summary['runtime_summary']['total_ms']
    print('\nRUNTIME CURRENT METHOD')
    print('mean total ms:',timing['mean'])
    print('median total ms:',timing['median'])
    print('p90 total ms:',timing['p90'])
    print('\nINTEGRITY')
    for label,key in (('Upsampler unchanged','upsampler_unchanged'),
        ('SR DiT backbone unchanged','sr_dit_backbone_unchanged'),('Wan unchanged','wan_unchanged'),
        ('Condition projection restored','condition_projection_restored'),('Adapter restored','adapter_restored'),
        ('Adapter indices','adapter_indices')):print(label+':',integrity[key])
    print('\nNaN:',completion['nan_metrics'])
    print('Inf:',completion['inf_metrics'])
    print('Missing clips:',completion['missing_clips'])
    print('Duplicate clips:',completion['duplicate_clips'])
    print('\nResults directory:',ROOT)


if __name__=='__main__':
    import sys
    assert len(sys.argv)==2 and sys.argv[1] in ('summarize','visuals','print')
    if sys.argv[1]=='summarize':summarize()
    elif sys.argv[1]=='visuals':visuals()
    else:print_final(load(ROOT/'metrics/final_raw_summary.json'))
