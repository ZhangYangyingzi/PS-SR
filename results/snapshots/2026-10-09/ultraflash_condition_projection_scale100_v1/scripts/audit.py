import gc
import json
from common import *

def equivalence():
    ensure();seed_all()
    write_json(AUDIT/'configuration.json',settings())
    up=base.load_up(train=False);sr=base.load_sr()
    original=sr.patch_embedding
    split=ConditionProjection(original)
    assert torch.equal(split.weight,original.weight)
    records=[]
    for index,row in enumerate(rows('val')[:4]):
        x,_=latent_pair(row)
        with torch.no_grad():
            c=run_up(up,x)
            sr.patch_embedding=original
            before=refine(sr,c,SEED+index)
            sr.patch_embedding=split
            after=refine(sr,c,SEED+index)
        diff=float((before.float()-after.float()).abs().max())
        assert diff==0
        records.append({'clip_id':row['clip_id'],'max_abs_diff':diff})
        print(f'equivalence {index+1}/4: {diff}',flush=True)
    write_json(AUDIT/'initial_equivalence.json',{'clips':records,'max_abs_diff':max(x['max_abs_diff'] for x in records),'passed':True})
    write_json(AUDIT/'condition_projection_structure.json',{'module':'patch_embedding',
        'source':str(INF/'sr/stream_forward.py'),'source_concat':'torch.cat([u,v],dim=0) for u,v in zip(x_list,y_list)',
        'weight_shape':list(original.weight.shape),'latent_channel_range':[0,16],
        'condition_channel_range':[16,32],'range_convention':'half-open',
        'bias_handling':'frozen shared bias','trainable_parameter_count':split.condition_weight.numel(),
        'initial_weight_equal':True,'parameterization':'frozen latent buffer + FP32 condition Parameter cast to BF16; single unchanged Conv3d',
        'sr_checkpoint':str(base.SR_CKPT.resolve())})
    atomic_save(checkpoint(0),{'step':0,'condition_weight':split.condition_weight.detach().cpu(),'config':settings()})

def gradient_audit(clip_index=None):
    ensure();seed_all()
    manifest=json.loads((PREVIOUS/'audit/gradient_audit_samples.json').read_text())
    if clip_index in (None,0):write_json(AUDIT/'gradient_audit_samples.json',manifest)
    byid={r['clip_id']:r for r in rows('train')}
    selected=[byid[r['clip_id']] for r in manifest['samples']]
    bundle=load_models(with_losses=True);up,sr,vae,lp=bundle
    restore(sr,0)
    if clip_index in (None,0):write_json(AUDIT/'integrity_before.json',identity(up,sr,vae))
    names=[*WEIGHTS,'total']; vectors={name:[] for name in names};perclip=[]
    for index,row in enumerate(selected):
        if clip_index is not None and index!=clip_index:continue
        entry={'clip_id':row['clip_id']}
        for name in names:
            part=AUDIT/'gradient_parts'/f'{name}_{index}.pt'
            if part.exists():
                record=torch.load(part,map_location='cpu',weights_only=True)
            else:
                sr.zero_grad(set_to_none=True)
                with torch.autograd.graph.save_on_cpu(pin_memory=False):
                    raw,weighted,total=forward_losses(row,*bundle,SEED+index,only=name)
                objective=total if name=='total' else weighted[name]
                objective.backward()
                v=parameters(sr)[0].grad.detach().float().cpu().flatten()
                assert torch.isfinite(v).all() and torch.isfinite(objective)
                if name in ('latent','rgb','lpips','hf'): assert float(v.norm())>0
                record={'clip_id':row['clip_id'],'weighted_gradient':v,'weighted_loss':float(objective.detach())}
                atomic_save(part,record)
                del raw,weighted,total,objective,v
                sr.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            assert record['clip_id']==row['clip_id']
            vectors[name].append(record['weighted_gradient'])
            entry[name]=float(record['weighted_gradient'].norm())
            print(f'gradient {name} {index+1}/4: {entry[name]}',flush=True)
        perclip.append(entry)
    if clip_index is not None:return
    means={k:torch.stack(v).mean(0).double() for k,v in vectors.items()}
    norms={k:float(v.norm()) for k,v in means.items()}
    def cosine(a,b):
        denom=means[a].norm()*means[b].norm()
        return float(torch.dot(means[a],means[b])/denom) if denom else None
    cos=[{'loss':a,**{b:cosine(a,b) for b in names}} for a in names]
    write_csv(AUDIT/'gradient_cosine_initial.csv',cos)
    assert all(norms[k]>0 for k in ('latent','rgb','lpips','hf'))
    write_json(AUDIT/'gradient_balance_initial.json',{'weighted_grad_norms':norms,'per_clip':perclip,
        'weights':WEIGHTS,'lpips_frames':FRAMES,'aggregation':'norm of mean gradient across same four Train clips',
        'sum_vs_total_l2':float((sum(means[n] for n in WEIGHTS)-means['total']).norm()),
        'gate':'finite gradients; primary gradients nonzero; no balance-ratio gate','passed':True})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('equivalence','gradients'))
    p.add_argument('--clip-index',type=int,choices=range(4))
    a=p.parse_args()
    equivalence() if a.stage=='equivalence' else gradient_audit(a.clip_index)
