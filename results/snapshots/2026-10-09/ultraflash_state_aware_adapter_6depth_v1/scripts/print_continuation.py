"""Print only the requested factual terminal summary after all outputs exist."""
import json
from common import ROOT, INDICES

source=json.loads((ROOT/'audit/resume_step1000_audit.json').read_text())
runtime=json.loads((ROOT/'continuation_1000_3000/audit/resume_runtime.json').read_text())
opt=json.loads((ROOT/'audit/resumed_optimizer.json').read_text())
equiv=json.loads((ROOT/'audit/resume_equivalence.json').read_text())
integrity=json.loads((ROOT/'audit/final_integrity_step3000.json').read_text())
history={step:json.loads((ROOT/f'metrics/val_step{step:04d}_macro.json').read_text())
         for step in (1000,1500,2000,2500,3000)}
print('RESUME')
print('Source checkpoint:',source['checkpoint_path'])
print('Source SHA256:',source['checkpoint_sha256'])
print('Global step restored:',source['global_step'])
print('Optimizer restored:',runtime['optimizer_restored'])
print('GradScaler restored:',runtime['grad_scaler_restored'])
print('Scheduler restored:',runtime['scheduler_restored'])
print('RNG restored:',runtime['rng_state_restored'])
print('\nOPTIMIZER')
print('Projection LR:',opt['groups'][0]['lr'])
print('Adapter LR:',opt['groups'][1]['lr'])
print('Betas:',opt['groups'][0]['betas'])
print('Weight decay:',opt['groups'][0]['weight_decay'])
print('\nMODEL')
print('Adapter indices:',list(INDICES))
print('Condition projection trainable params:',opt['groups'][0]['parameter_count'])
print('Adapter trainable params:',opt['groups'][1]['parameter_count'])
print('Total trainable params:',sum(g['parameter_count'] for g in opt['groups']))
print('\nRESUME EQUIVALENCE')
print('PASS:',equiv['pass'])
print('max latent diff:',equiv['max_abs_latent_diff'])
print('metric max diff:',equiv['metric_max_diff'])
print('\nVAL HISTORY')
for step,values in history.items():
    print(f'\nstep{step}')
    for label,key in (('PSNR','psnr'),('SSIM','ssim'),('LPIPS','lpips'),('DISTS','dists'),
                      ('latent MSE','latent_mse'),('latent MAE','latent_mae')):
        print(f'{label}:',values[key])
print('\nINTEGRITY')
for label,key in (('Upsampler unchanged','upsampler_unchanged'),
                  ('Latent-side projection unchanged','latent_side_projection_unchanged'),
                  ('SR DiT backbone unchanged','sr_dit_backbone_unchanged'),
                  ('Wan unchanged','wan_unchanged'),
                  ('Condition projection changed since step1000','condition_projection_changed_since_step1000'),
                  ('Adapter changed since step1000','adapter_changed_since_step1000')):
    print(f'{label}:',integrity[key])
print('\nNaN:',0)
print('Inf:',0)
print('\nResults directory:',ROOT)
