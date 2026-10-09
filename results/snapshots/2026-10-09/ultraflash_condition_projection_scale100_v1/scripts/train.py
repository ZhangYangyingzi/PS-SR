import argparse
import gc
import math
from common import *

def save_state(path,step,sr,opt):
    atomic_save(path,{'step':step,'condition_weight':parameters(sr)[0].detach().cpu(),
        'optimizer':opt.state_dict(),'config':settings(),
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
               'cuda':torch.cuda.get_rng_state_all()}})

def main():
    p=argparse.ArgumentParser();p.add_argument('--until',type=int,choices=STEPS[1:],required=True)
    a=p.parse_args();ensure();seed_all()
    assert json.loads((AUDIT/'gradient_balance_initial.json').read_text())['passed']
    expected={'psnr':25.691580,'ssim':.601324,'lpips':.254717,'dists':.196543}
    bundle=load_models(with_losses=True);up,sr,vae,lp=bundle
    before=identity(up,sr,vae)
    initial=parameters(sr)[0].detach().clone()
    opt=torch.optim.AdamW(parameters(sr),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
    latest=RUN/'latest.pt'
    state=torch.load(latest if latest.exists() else checkpoint(0),map_location='cpu',weights_only=False)
    assert state['config']==settings()
    assert state['step']<=a.until, 'Refusing to label a newer checkpoint with an older step'
    with torch.no_grad():parameters(sr)[0].copy_(state['condition_weight'])
    if 'optimizer' in state:
        opt.load_state_dict(state['optimizer']);r=state['rng']
        random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
    history_path=RUN/'train.csv'
    history=list(csv.DictReader(history_path.open())) if history_path.exists() else []
    history=[r for r in history if int(r['step'])<=state['step']]
    train_rows=rows('train'); assert len(train_rows)==400
    for step in range(state['step']+1,a.until+1):
        print(f'train step {step}/4000 start',flush=True)
        opt.zero_grad(set_to_none=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            raw,weighted,total=forward_losses(train_rows[(step-1)%400],*bundle,SEED+step)
        assert torch.isfinite(total)
        total.backward()
        grad=torch.nn.utils.clip_grad_norm_(parameters(sr),1.,error_if_nonfinite=True)
        opt.step()
        if step==1:
            save_state(latest,step,sr,opt)
            first_after=identity(up,sr,vae)
            assert first_after==before
            update=float((parameters(sr)[0]-initial).abs().max())
            assert update>0
            write_json(AUDIT/'first_training_step.json',{'step':1,'loss':float(total.detach()),
                'grad_norm':float(grad),'all_frozen_checksums_identical':True,
                'condition_max_abs_diff':update,'lpips_frames':FRAMES})
            print(f'train step 1 complete; loss={float(total.detach())}; grad_norm={float(grad)}',flush=True)
        if step%50==0:
            history.append({'step':step,'total_loss':float(total.detach()),
                **{f'{k}_loss':float(v.detach()) for k,v in raw.items()},'grad_norm':float(grad),'lr':1e-4})
            save_state(latest,step,sr,opt);write_csv(history_path,history)
        del raw,weighted,total;gc.collect();torch.cuda.empty_cache()
    save_state(checkpoint(a.until),a.until,sr,opt)
    after=identity(up,sr,vae)
    diff=float((parameters(sr)[0]-initial).abs().max())
    effective=float((parameters(sr)[0].bfloat16()-initial.bfloat16()).abs().max())
    result={'step':a.until,'before':before,'after':after,
        'upsampler_unchanged':before['upsampler']==after['upsampler'],
        'latent_side_projection_unchanged':before['latent_projection']==after['latent_projection'],
        'sr_dit_backbone_unchanged':before['sr_frozen']==after['sr_frozen'],
        'wan_unchanged':before['wan']==after['wan'],'condition_side_projection_changed':diff>0,
        'condition_max_abs_diff':diff,'condition_effective_bf16_max_abs_diff':effective,
        'only_trainable_parameters':['patch_embedding.condition_weight']}
    write_json(AUDIT/f'integrity_step{a.until:04d}.json',result);write_json(AUDIT/'integrity.json',result)
    assert all(result[k] for k in ('upsampler_unchanged','latent_side_projection_unchanged','sr_dit_backbone_unchanged','wan_unchanged','condition_side_projection_changed'))

if __name__=='__main__':main()
