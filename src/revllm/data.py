"""Deterministic TinyStories preparation and exact target-token tape."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER_DOCS = 100_000
TRAIN_TARGETS = 50_000_000
DEV_TARGETS = 262_144
HOLDOUT_TARGETS = 1_000_000
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|eos|>"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_unique_text(split: str, banned_hashes: set[str] | None = None):
    """Yield source-order documents with exact normalized-text deduplication."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split=split, revision=DATASET_REVISION, streaming=True)
    seen: set[str] = set()
    banned_hashes = banned_hashes or set()
    for row in dataset:
        text = row["text"].replace("\r\n", "\n").strip()
        if not text:
            continue
        digest = digest_bytes(text.encode("utf-8"))
        if digest in seen or digest in banned_hashes:
            continue
        seen.add(digest)
        yield text, digest


def _validation_texts():
    """Materialize the modest official validation split for leakage filtering."""
    return list(iter_unique_text("validation"))


def _tokenizer_corpus(banned: set[str]):
    yielded = 0
    for text, _ in iter_unique_text("train", banned):
        yield text
        yielded += 1
        if yielded == TOKENIZER_DOCS:
            return
    raise RuntimeError("Training split ended before tokenizer corpus was complete")


def _encode_tape(documents, tokenizer, target_count: int) -> tuple[np.ndarray, int, int]:
    eos = tokenizer.token_to_id(EOS_TOKEN)
    tape = np.empty(target_count + 1, dtype=np.uint16)
    cursor = 0
    source_docs = 0
    separator_count = 0
    for text in documents:
        ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
        available = min(len(ids), len(tape) - cursor)
        tape[cursor:cursor + available] = ids[:available]
        separator_count += ids[:available].count(eos)
        cursor += available
        source_docs += 1
        if cursor == len(tape):
            return tape, source_docs, separator_count
    raise RuntimeError(f"Only {cursor - 1:,} loss-bearing targets available; need {target_count:,}")


def prepare(out_dir: Path, train_targets: int = TRAIN_TARGETS) -> dict:
    """Download once from a pinned revision and save reproducible uint16 tapes."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from huggingface_hub import HfApi

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Reading pinned TinyStories validation documents", flush=True)
    validation = _validation_texts()
    val_hashes = {digest for _, digest in validation}
    print(f"Training 10,000-token BPE from {TOKENIZER_DOCS:,} training documents", flush=True)
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        _tokenizer_corpus(val_hashes),
        trainer=trainers.BpeTrainer(
            vocab_size=10_000,
            special_tokens=[PAD_TOKEN, EOS_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    if tokenizer.get_vocab_size() != 10_000:
        raise RuntimeError(f"Tokenizer has {tokenizer.get_vocab_size()} entries, expected 10,000")
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    dev_docs = (text for text, digest in validation if int(digest[:8], 16) % 2 == 0)
    holdout_docs = (text for text, digest in validation if int(digest[:8], 16) % 2 == 1)
    train_docs = (text for text, _ in iter_unique_text("train", val_hashes))
    counts = {}
    for name, documents, target_count in (
        ("dev", dev_docs, DEV_TARGETS),
        ("holdout", holdout_docs, HOLDOUT_TARGETS),
        ("train", train_docs, train_targets),
    ):
        print(f"Preparing {name} tape: {target_count:,} targets", flush=True)
        tape, source_docs, separators = _encode_tape(documents, tokenizer, target_count)
        path = out_dir / f"{name}.bin"
        tape.tofile(path)
        counts[name] = {
            "target_count": target_count,
            "source_documents": source_docs,
            "separator_input_tokens": separators,
            "sha256": digest_file(path),
        }
        print(f"Prepared {name}: {source_docs:,} source documents", flush=True)
    manifest = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "license": "cdla-sharing-1.0",
        "tokenizer_sha256": digest_file(tokenizer_path),
        "tokenizer_training_documents": TOKENIZER_DOCS,
        "normalization": "CRLF to LF; strip surrounding whitespace; exact SHA256 document dedup",
        "split_rule": "official train for train/tokenizer; official validation parity of first 32 hash bits for dev/holdout",
        "pad_id": tokenizer.token_to_id(PAD_TOKEN),
        "eos_id": tokenizer.token_to_id(EOS_TOKEN),
        "tapes": counts,
    }
    info = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION, files_metadata=True)
    manifest["source_files"] = {
        sibling.rfilename: {"bytes": sibling.size, "upstream_lfs_sha256": sibling.lfs["sha256"]}
        for sibling in info.siblings
        if sibling.rfilename.startswith("data/") and sibling.rfilename.endswith(".parquet") and sibling.lfs
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass
class TokenTape:
    path: Path
    target_count: int
    block_size: int = 512
    pad_id: int = 0

    def __post_init__(self) -> None:
        self.array = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.array) < self.target_count + 1:
            raise ValueError("Tape is shorter than the requested target budget plus the final input")

    def batch(self, cursor: int, physical_batch: int, remaining: int | None = None):
        if cursor < 0 or cursor >= self.target_count or physical_batch < 1:
            raise ValueError("Invalid cursor or batch size")
        count = min(physical_batch * self.block_size, self.target_count - cursor)
        if remaining is not None:
            count = min(count, remaining)
        if count < 1:
            raise ValueError("No targets requested")
        x = np.full(physical_batch * self.block_size, self.pad_id, dtype=np.int64)
        y = np.full(physical_batch * self.block_size, -100, dtype=np.int64)
        x[:count] = self.array[cursor:cursor + count]
        y[:count] = self.array[cursor + 1:cursor + count + 1]
        shape = (physical_batch, self.block_size)
        return torch.from_numpy(x.reshape(shape)), torch.from_numpy(y.reshape(shape)), count


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, item in manifest["tapes"].items():
        tape_path = path.parent / f"{name}.bin"
        if not tape_path.exists() or digest_file(tape_path) != item["sha256"]:
            raise RuntimeError(f"Missing or corrupted {name} tape")
    if digest_file(path.parent / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer hash mismatch")
    return manifest
