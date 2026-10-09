from common import *

def main():
    structure=json.loads((AUDIT/'condition_projection_structure.json').read_text())
    equivalence=json.loads((AUDIT/'initial_equivalence.json').read_text())
    gradients=json.loads((AUDIT/'gradient_balance_initial.json').read_text())
    history=list(csv.DictReader((ROOT/'metrics/val_history.csv').open()))
    selection=json.loads((ROOT/'metrics/selection.json').read_text())
    result=json.loads((ROOT/'metrics/test_macro.json').read_text())
    integrity=json.loads((AUDIT/'integrity.json').read_text())
    baseline={'psnr':23.780474,'ssim':.647606,'lpips':.201375,'dists':.153097}
    delta={k:result[k]-v for k,v in baseline.items()}
    selected=torch.load(RUN/'best.pt',map_location='cpu',weights_only=False)
    initial=torch.load(checkpoint(0),map_location='cpu',weights_only=False)
    integrity['selected_step']=selected['step']
    integrity['selected_condition_max_abs_diff']=float((selected['condition_weight']-initial['condition_weight']).abs().max())
    write_json(ROOT/'metrics/final_raw_summary.json',{'structure':structure,'equivalence':equivalence,
        'gradients':gradients,'val_history':history,'selection':selection,'baseline_test':baseline,
        'adapted_test':result,'delta':delta,'integrity':integrity})
    print('CONDITION PROJECTION AUDIT')
    for label,key in [('Module','module'),('Weight shape','weight_shape'),('Latent channel range','latent_channel_range'),
                      ('Condition channel range','condition_channel_range'),('Trainable params','trainable_parameter_count')]:print(label+':',structure[key])
    print('Initial equivalence max diff:',equivalence['max_abs_diff'])
    print('INITIAL GRADIENT NORMS')
    for k,v in gradients['weighted_grad_norms'].items():print(k+':',v)
    print('VAL HISTORY\nstep / PSNR / SSIM / LPIPS / DISTS / latent_MSE')
    for r in history:print(' / '.join(str(r[k]) for k in ('step','psnr','ssim','lpips','dists','latent_mse')))
    print('Best step:',selection['selected']['step'])
    for title,values in [('STEP8000 BASELINE TEST',baseline),('CONDITION-PROJECTION ADAPTED TEST',result),('DELTA',delta)]:
        print(title)
        for k,v in values.items():print({'latent_mse':'latent MSE','latent_mae':'latent MAE'}.get(k,k.upper())+':',v)
    print('INTEGRITY')
    for label,key in [('Upsampler unchanged','upsampler_unchanged'),
        ('Latent-side projection unchanged','latent_side_projection_unchanged'),
        ('Condition-side projection changed','condition_side_projection_changed'),
        ('SR DiT backbone unchanged','sr_dit_backbone_unchanged'),('Wan unchanged','wan_unchanged')]:
        print(label+':',integrity[key])
    print('Results directory:',ROOT)

if __name__=='__main__':main()
