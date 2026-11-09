"""Portable local launch adapter, derived from recovered personal shell commands.
Does not reproduce unavailable assets/environment. No shutdown or server changes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pretrain', 'sft', 'grpo'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs' / f'{args.stage}.json').read_text())
    flags = {key: os.path.expandvars(value) for key, value in config['local_flags'].items()}
    flags['from_resume'] = '1' if args.resume else '0'
    entry = {'pretrain': 'train_pretrain.py', 'sft': 'train_full_sft.py', 'grpo': 'train_grpo.py'}[args.stage]
    command = [sys.executable, '-u', entry]
    for key, value in flags.items():
        command.extend(['--' + key, value])
    if args.dry_run:
        print(json.dumps({'cwd': str(ROOT / 'trainer'), 'argv': command}, indent=2))
        return
    if '$' in flags.get('reward_model_path', ''):
        parser.error('Set MINIMIND_REWARD_MODEL to a licensed local reward-model directory.')
    for key in ['data_path', 'reward_model_path']:
        if key in flags and not (ROOT / 'trainer' / flags[key]).exists():
            parser.error(f'Missing {key}; obtain and verify the required local asset first.')
    if not args.resume and flags['from_weight'] != 'none':
        seed = ROOT / 'out' / f"{flags['from_weight']}_{flags['hidden_size']}.pth"
        if not seed.is_file():
            parser.error('Missing parent weight in repository/out; verify lineage before running.')
    if args.resume:
        resume = ROOT / 'checkpoints' / f"{flags['save_weight']}_{flags['hidden_size']}_resume.pth"
        if not resume.is_file():
            parser.error('Missing matching resume checkpoint; weights alone do not restore optimizer state.')
    output = ROOT / 'out' / f"{flags['save_weight']}_{flags['hidden_size']}.pth"
    if output.exists() and not args.resume:
        parser.error('Output already exists; use an isolated run directory rather than overwrite it.')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run(command, cwd=ROOT / 'trainer', env=env, check=True)

if __name__ == '__main__':
    main()
