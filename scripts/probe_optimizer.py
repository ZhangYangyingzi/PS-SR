"""CPU optimizer isolation test on an independent copy of the projection."""
from common import *

def main():
    state=torch.load(base.SR_CKPT,map_location='cpu',weights_only=True,mmap=True)['model']
    conv=torch.nn.Conv3d(32,1536,(1,2,2),stride=(1,2,2)).bfloat16()
    conv.load_state_dict({'weight':state['patch_embedding.weight'],'bias':state['patch_embedding.bias']})
    projection=ConditionProjection(conv)
    latent=projection.latent_weight.clone();bias=projection.bias.clone();condition=projection.condition_weight.detach().clone()
    gradient=torch.load(AUDIT/'gradient_parts/latent_0.pt',map_location='cpu',weights_only=True)['weighted_gradient']
    projection.condition_weight.grad=gradient.reshape_as(projection.condition_weight).clone()
    opt=torch.optim.AdamW([projection.condition_weight],lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_([projection.condition_weight],1.,error_if_nonfinite=True)
    opt.step()
    result={'isolated_cpu_copy_only':True,'actual_training_checkpoint_modified':False,
        'latent_max_abs_diff':float((projection.latent_weight-latent).abs().max()),
        'bias_max_abs_diff':float((projection.bias-bias).abs().max()),
        'condition_max_abs_diff':float((projection.condition_weight.detach()-condition).abs().max()),
        'condition_effective_bf16_max_abs_diff':float((projection.condition_weight.detach().bfloat16()-condition.bfloat16()).abs().max())}
    assert result['latent_max_abs_diff']==0 and result['bias_max_abs_diff']==0
    assert result['condition_max_abs_diff']>0 and result['condition_effective_bf16_max_abs_diff']>0
    result['passed']=True
    write_json(AUDIT/'optimizer_isolation_probe.json',result)
    print(result)

if __name__=='__main__':main()
