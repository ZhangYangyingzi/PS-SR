import fcntl
import subprocess
import sys
import traceback
from common import *

def run(script,*args):
    command=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
    write_json(RUN/'status.json',{'completed':False,'running_stage':script,'arguments':list(map(str,args))})
    print('RUN',command,flush=True)
    subprocess.run(command,cwd=ROOT,check=True)

def main():
    ensure()
    with (RUN/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (AUDIT/'initial_equivalence.json').exists():run('audit.py','equivalence')
            assert json.loads((AUDIT/'initial_equivalence.json').read_text())['passed']
            if not (AUDIT/'gradient_balance_initial.json').exists():run('parallel_gradient_audit.py')
            assert json.loads((AUDIT/'gradient_balance_initial.json').read_text())['passed']
            for step in STEPS:
                if step and not checkpoint(step).exists():run('train.py','--until',step)
                if not (ROOT/f'metrics/val_per_video_step{step:04d}.csv').exists():
                    run('evaluate.py','generate','--split','val','--step',step)
                    run('evaluate.py','score','--split','val','--step',step)
            if not (ROOT/'metrics/selection.json').exists():run('evaluate.py','select')
            best=json.loads((ROOT/'metrics/selection.json').read_text())['selected']['step']
            if not (ROOT/'metrics/test_macro.json').exists():
                run('evaluate.py','generate','--split','test','--step',best)
                run('evaluate.py','score','--split','test','--step',best)
            run('finalize.py')
            write_json(RUN/'status.json',{'completed':True,'best_step':best})
        except BaseException:
            write_json(RUN/'status.json',{'completed':False,'exception':traceback.format_exc()})
            raise

if __name__=='__main__':main()
