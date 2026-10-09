"""Detach the workflow continuation from the interactive tool session."""
import subprocess
import sys

from common import ROOT, write_json

TRAINING_PID = 639754
with (ROOT / 'logs/continuation.log').open('a') as handle:
    process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/after_250.py'),
                                '--training-pid', str(TRAINING_PID)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
        stderr=subprocess.STDOUT, start_new_session=True)
write_json(ROOT / 'runs/continuation_launch.json',
           {'pid':process.pid, 'training_pid':TRAINING_PID,
            'workflow':str(ROOT / 'scripts/after_250.py'),
            'log':str(ROOT / 'logs/workflow.log')})
print(process.pid)
