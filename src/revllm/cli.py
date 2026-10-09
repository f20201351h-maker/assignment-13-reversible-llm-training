"""Stable command interface used by notebooks and Kaggle jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare
from .model import ModelConfig
from .profile import benchmark, probe_once
from .evaluate import evaluate_checkpoint
from .train import RunConfig, train


def _run_config(path: Path) -> RunConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model"] = ModelConfig(**raw["model"])
    raw["data_dir"] = Path(raw["data_dir"])
    raw["out_dir"] = Path(raw["out_dir"])
    if "adam_betas" in raw:
        raw["adam_betas"] = tuple(raw["adam_betas"])
    return RunConfig(**raw)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="revllm")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build pinned TinyStories tokenizer and token tapes")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-targets", type=int, default=50_000_000)
    t = sub.add_parser("train", help="run or resume an experiment")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--max-updates", type=int)
    probe = sub.add_parser("probe-once", help="fresh-process GPU batch-capacity trial")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--batch", type=int, required=True)
    probe.add_argument("--updates", type=int, default=20)
    bench = sub.add_parser("benchmark", help="isolated steady-state benchmark")
    bench.add_argument("--config", type=Path, required=True)
    ev = sub.add_parser("evaluate", help="re-evaluate a checkpoint using locally regenerated tapes")
    ev.add_argument("--config", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--data-dir", type=Path, required=True)
    ev.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.out, args.train_targets)
    elif args.command == "train":
        result = train(_run_config(args.config), resume=args.resume, max_updates=args.max_updates)
    elif args.command == "probe-once":
        result = probe_once(_run_config(args.config), args.batch, args.updates)
    elif args.command == "benchmark":
        result = benchmark(_run_config(args.config))
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.config, args.checkpoint, args.data_dir)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
