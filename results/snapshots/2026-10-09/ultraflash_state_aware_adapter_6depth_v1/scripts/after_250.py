"""Continue the fixed workflow only after the active step250 process finishes normally."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import ROOT, checkpoint_path, write_json


def alive(pid):
    status = Path('/proc') / str(pid) / 'stat'
    if status.exists():
        fields = status.read_text().split()
        if len(fields) > 2 and fields[2] == 'Z':
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-pid', type=int, required=True)
    args = parser.parse_args()
    write_json(ROOT / 'runs/workflow_status.json',
        {'status':'waiting_for_step250_process', 'training_pid':args.training_pid})
    while alive(args.training_pid):
        time.sleep(30)
    if not checkpoint_path(250).is_file():
        write_json(ROOT / 'runs/workflow_status.json',
            {'status':'stopped', 'reason':'step250 training process ended without step250 checkpoint'})
        return 1
    write_json(ROOT / 'runs/workflow_status.json', {'status':'continuing_from_step250'})
    try:
        with (ROOT / 'logs/workflow.log').open('a') as handle:
            subprocess.run([sys.executable, str(ROOT / 'scripts/run_all.py')],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        write_json(ROOT / 'runs/workflow_status.json', {'status':'complete', 'test_executed':False})
        return 0
    except Exception:
        write_json(ROOT / 'runs/workflow_status.json',
            {'status':'stopped', 'reason':'fixed workflow stage failed', 'traceback':traceback.format_exc()})
        return 1


if __name__ == '__main__': sys.exit(main())
