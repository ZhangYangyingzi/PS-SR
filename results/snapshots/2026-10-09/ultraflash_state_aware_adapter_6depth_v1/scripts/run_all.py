"""Complete fixed 250-step training/Val stages without Test or checkpoint selection."""
import subprocess
import sys
import traceback
from pathlib import Path

from common import ROOT, STEPS, checkpoint_path, write_json


def run(label, *args):
    log_path = ROOT / 'logs' / f'{label}.log'
    with log_path.open('a') as handle:
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    assert (ROOT / 'audit/pretrain_pass.json').is_file()
    assert checkpoint_path(0).is_file()
    for step in STEPS:
        if step and not checkpoint_path(step).is_file():
            write_json(ROOT / 'runs/workflow_status.json', {'status':'training', 'target_step':step})
            run(f'train_to_{step:04d}', ROOT / 'scripts/train.py', '--until', step)
        macro = ROOT / 'metrics' / f'val_step{step:04d}_macro.json'
        if not macro.is_file():
            write_json(ROOT / 'runs/workflow_status.json', {'status':'validating', 'step':step})
            run(f'val_{step:04d}', ROOT / 'scripts/evaluate.py', '--step', step)
    with (ROOT / 'logs/final_terminal_output.log').open('w') as handle:
        subprocess.run([sys.executable, str(ROOT / 'scripts/finalize.py')], cwd=ROOT,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    write_json(ROOT / 'runs/workflow_status.json', {'status':'complete', 'test_executed':False})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write_json(ROOT / 'runs/workflow_status.json',
                   {'status':'stopped', 'traceback':traceback.format_exc()})
        raise
