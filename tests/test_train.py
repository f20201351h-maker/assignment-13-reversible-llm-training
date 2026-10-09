import json
import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import pytest

from revllm.data import digest_file
from revllm.evaluate import evaluate_checkpoint
from revllm.model import ModelConfig
from revllm.train import RunConfig, learning_rate, train


def make_fake_data(root: Path) -> None:
    root.mkdir()
    (root / "tokenizer.json").write_text("test tokenizer", encoding="utf-8")
    specs = {"train": 29, "dev": 8, "holdout": 8}
    tapes = {}
    for name, targets in specs.items():
        (np.arange(targets + 1, dtype=np.uint16) % 31).tofile(root / f"{name}.bin")
        tapes[name] = {"target_count": targets, "sha256": digest_file(root / f"{name}.bin")}
    manifest = {
        "pad_id": 0, "eos_id": 1, "tokenizer_sha256": digest_file(root / "tokenizer.json"),
        "tapes": tapes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _config(tmp_path: Path, out_name: str) -> RunConfig:
    return RunConfig(
        run_id=out_name,
        model=ModelConfig(vocab_size=32, block_size=4, n_layer=2, n_head=2, n_embd=16,
                          integrator="midpoint", backward_mode="reconstructed"),
        data_dir=tmp_path / "data", out_dir=tmp_path / out_name,
        physical_batch=2, effective_batch=3, target_tokens=29,
        warmup_tokens=4, eval_every_tokens=12, checkpoint_every_tokens=12,
        use_amp=False,
    )


def test_exact_accounting_and_checkpoint_resume(tmp_path: Path):
    make_fake_data(tmp_path / "data")
    direct = train(_config(tmp_path, "direct"))
    interrupted_cfg = _config(tmp_path, "interrupted")
    partial = train(interrupted_cfg, max_updates=1)
    assert partial["status"] == "INCOMPLETE"
    assert partial["committed_targets"] == 12
    resumed = train(interrupted_cfg, resume=True)
    assert direct["committed_targets"] == resumed["committed_targets"] == 29
    assert direct["attempted_targets"] == resumed["attempted_targets"] == 29
    assert direct["updates"] == resumed["updates"] == 3
    assert direct["last_1m_training_loss_nats"] == resumed["last_1m_training_loss_nats"]
    direct_state = torch.load(tmp_path / "direct" / "latest.pt", weights_only=False)
    resumed_state = torch.load(tmp_path / "interrupted" / "latest.pt", weights_only=False)
    for key in direct_state["model"]:
        torch.testing.assert_close(direct_state["model"][key], resumed_state["model"][key], rtol=0, atol=0)
    rechecked = evaluate_checkpoint(interrupted_cfg.out_dir / "run_config.json",
                                   interrupted_cfg.out_dir / "latest.pt", tmp_path / "data")
    assert rechecked["committed_targets_at_checkpoint"] == 29
    assert rechecked["evaluation"]["holdout"]["targets"] == 8
    records = [json.loads(line) for line in (tmp_path / "interrupted" / "metrics.jsonl").read_text().splitlines()]
    assert [r["valid_targets"] for r in records] == [12, 12, 5]
    assert records[-1]["microbatches"] == 1


def test_token_based_lr_schedule(tmp_path: Path):
    cfg = _config(tmp_path, "lr")
    assert 0 < learning_rate(1, cfg) < cfg.learning_rate
    assert learning_rate(4, cfg) == cfg.learning_rate
    assert learning_rate(29, cfg) == cfg.min_learning_rate


def test_resume_rolls_back_uncheckpointed_log_without_double_counting(tmp_path: Path, monkeypatch):
    make_fake_data(tmp_path / "data")
    cfg = _config(tmp_path, "interrupted_tail")
    partial = train(cfg, max_updates=1)
    assert partial["committed_targets"] == 12
    metrics = cfg.out_dir / "metrics.jsonl"
    with metrics.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"status": "COMMITTED", "committed_targets": 24,
                               "attempted_targets": 24, "valid_targets": 12,
                               "step_seconds": 0.2}) + "\n")
    with (cfg.out_dir / "eval.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"committed_targets": 24, "dev_loss_nats": 99}) + "\n")
    train_module = importlib.import_module("revllm.train")
    real_schedule = train_module.learning_rate
    def interrupt_before_update(*_args):
        raise RuntimeError("simulated second interruption")
    monkeypatch.setattr(train_module, "learning_rate", interrupt_before_update)
    with pytest.raises(RuntimeError, match="simulated second interruption"):
        train(cfg, resume=True)
    monkeypatch.setattr(train_module, "learning_rate", real_schedule)
    resumed = train(cfg, resume=True)
    assert resumed["committed_targets"] == 29
    assert resumed["attempted_targets"] == 41
    live = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [r["committed_targets"] for r in live] == [12, 24, 29]
    assert sum(r["valid_targets"] for r in live) == 29
    assert (cfg.out_dir / "recovery_wal.json").exists()
    assert len(list((cfg.out_dir / "rollbacks").glob("metrics-*.jsonl"))) == 1
    assert len(list((cfg.out_dir / "rollbacks").glob("eval-*.jsonl"))) == 1
    assert not (cfg.out_dir / "partial.json").exists()


def test_unequal_microbatch_accumulation_matches_physical_batch(tmp_path: Path):
    make_fake_data(tmp_path / "data")
    base = _config(tmp_path, "physical")
    direct_cfg = replace(base, physical_batch=3, effective_batch=3, target_tokens=12,
                         eval_every_tokens=100, checkpoint_every_tokens=100)
    accumulated_cfg = replace(direct_cfg, run_id="accumulated", out_dir=tmp_path / "accumulated",
                              physical_batch=2)
    direct = train(direct_cfg)
    accumulated = train(accumulated_cfg)
    assert direct["committed_targets"] == accumulated["committed_targets"] == 12
    assert direct["updates"] == accumulated["updates"] == 1
    a = torch.load(direct_cfg.out_dir / "latest.pt", weights_only=False)["model"]
    b = torch.load(accumulated_cfg.out_dir / "latest.pt", weights_only=False)["model"]
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-6, msg=key)
