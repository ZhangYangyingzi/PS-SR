"""Start the resumable pipeline independently of the interactive shell."""
import fcntl
import subprocess
from datetime import datetime,timezone
from common import *

ensure()
with (RUN/'pipeline.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(lock,fcntl.LOCK_UN)
with (ROOT/'logs/pipeline.log').open('a') as log:
    process=subprocess.Popen(['bash',str(ROOT/'run.sh')],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(RUN/'launch.json',{'pid':process.pid,'started_utc':datetime.now(timezone.utc).isoformat(),
    'log':str(ROOT/'logs/pipeline.log'),'resume_command':f'bash {ROOT}/run.sh'})
print('Pipeline PID:',process.pid)
