"""Preserve the incomplete 5-frame LPIPS audit; establish the 17-frame config."""
import shutil
from common import *

ensure()
archive=AUDIT/'partial_fiveframe_lpips_audit'
archive.mkdir(exist_ok=True)
assert not (RUN/'latest.pt').exists(), 'Cannot change objective after training starts'
state=torch.load(checkpoint(0),map_location='cpu',weights_only=False)
if state['config']['lpips_frames']!=FRAMES:
    shutil.copy2(checkpoint(0),archive/'step_000000.pt')
    for name in ('configuration.json','gradient_parts','gradient_balance_initial.json','gradient_cosine_initial.csv'):
        path=AUDIT/name
        if path.exists():shutil.move(str(path),str(archive/name))
    state['config']=settings()
    atomic_save(checkpoint(0),state)
write_json(AUDIT/'configuration.json',settings())
print('LPIPS training frames:',FRAMES)
