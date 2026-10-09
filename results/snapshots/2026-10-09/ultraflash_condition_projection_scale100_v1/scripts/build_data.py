import csv,json,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PS=ROOT.parent
OLD=PS/'latent_sr_v2_multivideo'; OUT=ROOT/'data'; SEED=2026
def vid(path): return Path(path).stem+'_'+hashlib.sha1(str(path).encode()).hexdigest()[:8]
def main():
    old=list(csv.DictReader((OLD/'manifests/clips.csv').open()))
    old_by={s:[r for r in old if r['split']==s] for s in ('train','val','test')}
    old_train_vid=sorted({r['video_id'] for r in old_by['train']}); val_vid=sorted({r['video_id'] for r in old_by['val']}); test_vid=sorted({r['video_id'] for r in old_by['test']})
    pool=list(csv.DictReader((OLD/'data_audit/all_videos.csv').open()))
    excluded=set(val_vid+test_vid+old_train_vid)
    candidates=[r for r in pool if str(r.get('valid')).lower()=='true' and vid(r['absolute_path']) not in excluded]
    rng=random.Random(SEED); candidates.sort(key=lambda r:r['absolute_path']); rng.shuffle(candidates); add=candidates[:76]
    assert len(add)==76
    def clip_rows(video,split):
        duration=float(video['duration']); fps=17.; last=duration-16/fps; margin=min(.5,max(0.,last/20)); low,high=margin,max(margin,last-margin)
        starts=[low+(high-low)*i/3 for i in range(4)] if high>low else [low]*4
        v=video['video_id']
        for i,start in enumerate(starts):
            yield {'clip_id':f'{v}_clip_{i}','video_id':v,'source_video':video['absolute_path'],'dataset':video['dataset'],'split':split,'clip_index':str(i),'start_time':f'{start:.9f}','sample_timestamps':json.dumps([start+k/17 for k in range(17)]),'source_fps':video['fps']}
    rows=[]
    # retain exact original rows for all original splits and Train24
    rows.extend(old_by['val']+old_by['test']+old_by['train'])
    for x in add:
        x=dict(x);x['video_id']=vid(x['absolute_path']);rows.extend(clip_rows(x,'train'))
    assert len({r['video_id'] for r in rows if r['split']=='train'})==100
    fields=list(rows[0])
    with (OUT/'train100_clips.csv').open('w',newline='') as f: csv.DictWriter(f,fieldnames=fields).writeheader();csv.DictWriter(f,fieldnames=fields).writerows(rows)
    for split in ('train','val','test'):
        selected=sorted({r['video_id'] for r in rows if r['split']==split}); (OUT/f'{split}_manifest.json').write_text(json.dumps({'videos':selected,'clips':[r for r in rows if r['split']==split]},indent=2)+'\n')
    tv={r['video_id'] for r in rows if r['split']=='train'}; vv=set(val_vid); xv=set(test_vid)
    audit={'train_videos':len(tv),'train_clips':sum(r['split']=='train' for r in rows),'val_videos':len(vv),'val_clips':sum(r['split']=='val' for r in rows),'test_videos':len(xv),'test_clips':sum(r['split']=='test' for r in rows),'train_val_overlap':len(tv&vv),'train_test_overlap':len(tv&xv),'val_test_overlap':len(vv&xv),'original_train24_included':len(set(old_train_vid)&tv),'added_video_ids':[vid(x['absolute_path']) for x in add],'seed':SEED}
    (OUT/'split_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(audit)
if __name__=='__main__':main()
