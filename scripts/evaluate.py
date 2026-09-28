import argparse
import fcntl
from common import *

spec=importlib.util.spec_from_file_location('balanced_metrics',PREVIOUS/'scripts/evaluate.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
metrics.ROOT=ROOT;metrics.RUN=RUN;metrics.STEPS=STEPS;metrics.checkpoint=checkpoint

@torch.inference_mode()
def generate(split,step):
    up,sr,_,_=load_models();restore(sr,step);vae=load_vae()
    for index,row in enumerate(rows(split)):
        output=ROOT/'outputs'/split/f'step_{step:06d}'/row['video_id']/f"clip_{row['clip_index']}.mp4"
        path=output.with_suffix('.pt')
        if output.exists() and path.exists():continue
        x,gt=latent_pair(row);pred=refine(sr,run_up(up,x),SEED+index)
        assert tuple(pred.shape)==(1,16,5,60,108) and torch.isfinite(pred).all()
        cpu=pred.float().cpu();target=gt.float().cpu()
        rgb=vae.decode(pred,device='cuda',tiled=False)
        assert tuple(rgb.shape)==(1,3,17,480,864) and torch.isfinite(rgb).all()
        write_video(video_frames(rgb),output)
        atomic_save(path,{'z_hr_hat':pred.half().cpu(),'latent_mse':float((cpu-target).square().mean()),
                         'latent_mae':float((cpu-target).abs().mean())})
        print(f'{split} generate step{step}: {index+1}/24',flush=True)
        del x,gt,pred,rgb,cpu,target;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('generate','score','select'))
    p.add_argument('--split',choices=('val','test'));p.add_argument('--step',type=int,choices=STEPS)
    a=p.parse_args();ensure();seed_all()
    if a.stage=='select':
        metrics.select();return
    if a.split=='test':
        selection=json.loads((ROOT/'metrics/selection.json').read_text())
        assert selection['selected']['step']==a.step
        lock=ROOT/'metrics/test_selection_frozen.json'
        if lock.exists():assert json.loads(lock.read_text())==selection
        else:write_json(lock,selection)
    with (RUN/f'eval_{a.split}_{a.step}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.stage=='generate':generate(a.split,a.step)
        else:metrics.score(a.split,a.step)

if __name__=='__main__':main()
