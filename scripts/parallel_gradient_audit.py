import os
import subprocess
from common import *

def main():
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    free={int(line.split(',')[0]) for line in usage.splitlines() if int(line.split(',')[1])<100}
    if not {7,1,2,3}.issubset(free):
        if 7 not in free:raise RuntimeError('GPU7 occupied; refusing to interfere with existing work')
        subprocess.run([sys.executable,str(ROOT/'scripts/audit.py'),'gradients'],cwd=ROOT,check=True)
        return
    workers=[]
    handles=[]
    for index,gpu in enumerate((7,1,2,3)):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        log=(ROOT/'logs'/f'gradient_clip{index}.log').open('a')
        handles.append(log)
        worker=subprocess.Popen([sys.executable,str(ROOT/'scripts/audit.py'),'gradients','--clip-index',str(index)],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append(worker)
    codes=[w.wait() for w in workers]
    for h in handles:h.close()
    assert codes==[0,0,0,0],codes
    subprocess.run([sys.executable,str(ROOT/'scripts/audit.py'),'gradients'],cwd=ROOT,check=True)

if __name__=='__main__':main()
