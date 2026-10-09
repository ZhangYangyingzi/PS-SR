"""Detach the fixed training/Val workflow from an interactive session."""
import subprocess
import sys

from common import ROOT, write_json

with (ROOT / 'logs/workflow.log').open('a') as handle:
    process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_all.py')],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
        stderr=subprocess.STDOUT, start_new_session=True)
write_json(ROOT / 'runs/workflow_launch.json',
           {'pid': process.pid, 'script': str(ROOT / 'scripts/run_all.py'),
            'log': str(ROOT / 'logs/workflow.log')})
print(process.pid)
