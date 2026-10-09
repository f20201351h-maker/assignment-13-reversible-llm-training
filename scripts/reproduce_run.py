"""Run a frozen recipe against locally regenerated data without editing it."""
import argparse
from dataclasses import replace
from pathlib import Path

from revllm.cli import _run_config
from revllm.train import train

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()
matches = list((ROOT/'configs').glob(f'*/{args.run}.json'))
if len(matches) != 1:
    raise ValueError(f'Expected one frozen recipe for {args.run}, got {matches}')
cfg = replace(_run_config(matches[0]), data_dir=args.data_dir.resolve(), out_dir=args.out.resolve())
print(train(cfg, resume=args.resume))
