# Third-party sources and data

- TinyStories: `roneneldan/TinyStories`, pinned revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; dataset card identifies its license as CDLA-Sharing-1.0. The repository keeps a tokenizer and manifest, but excludes the large token tapes from Git. Anyone using the data must observe the dataset license.
- PyTorch, Hugging Face datasets/tokenizers, NumPy, and Matplotlib remain under their own upstream licenses; no source code from these packages is copied here.
- Architectural and experimental concepts are attributed to Gal et al., *Reversing Large Language Models for Efficient Training and Fine-Tuning*, arXiv:2512.02056, and the Session 13 course lecture. No course transcript or lecture file is distributed here.
- nanoGPT, build-nanogpt, nanochat, and the course LightningLM implementation informed design review. This repository's code is independently written; it does not vendor their source.

## ORX figure style

`scripts/orx_figstyle.py` is derived from the `orx-figures` style asset from
alphaXiv's openresearch-cli and is used under the MIT License. The complete
license text is preserved in `scripts/ORX_STYLE_LICENSE`.
