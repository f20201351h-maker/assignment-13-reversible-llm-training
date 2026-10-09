# Source register

## Experimental and mathematical sources

- Gal et al., *Reversing Large Language Models for Efficient Training and Fine-Tuning* (2025), [arXiv:2512.02056](https://arxiv.org/abs/2512.02056). Used to check the reversible formulations and their memory argument. A literature result is not counted as a measured Session 13 result.
- The School of AI, [released LLM repository](https://github.com/The-School-of-AI/LLM). Consulted for the course-related midpoint and blended formulation, without copying its large-model infrastructure.
- Session 13 GMeet transcript around 90:48–91:53 and the supplied lecture: local assignment context only. These files are **not** distributed in this repository.
- [TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories), revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The upstream LFS SHA-256 values for the four training and one validation parquet shards are recorded in `data/manifest.json`; they are upstream metadata hashes, while the token-tape and tokenizer hashes are locally computed.

## Readable implementation references

The following already-local references were inspected at fixed Git commits. This project does not vendor or execute their training frameworks.

| Reference | Local commit | Upstream |
|---|---|---|
| nanoGPT | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | https://github.com/karpathy/nanoGPT |
| build-nanogpt | `6104ab1b53920f6e2159749676073ff7d815c1fa` | https://github.com/karpathy/build-nanogpt |
| nanochat | `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` | https://github.com/karpathy/nanochat |

Their role was design review of the small decoder and training loop; all experiment code here is written in `src/revllm/`.
