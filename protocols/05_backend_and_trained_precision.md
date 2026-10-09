# Attention provenance and trained mixed-precision diagnostics

Exploratory diagnostic, committed before execution. This does not alter any frozen training configuration or retrospectively repair missing historical telemetry.

The original principal campaign used PyTorch 2.10.0+cu128 automatic scaled-dot-product-attention (SDPA) selection on T4. It did not explicitly force or record the selected operator. Consequently the original plan's explicit-backend-lock requirement is not historically verified. Report this deviation, even if later profiling strongly suggests the same implementation was used.

On the same Kaggle T4 software stack, profile the repository's CausalAttention forward and backward at context 512, 6 heads, head dimension 64, FP16 autocast, FP32 parameters/residual inputs, and physical batches 1, 110, 181 and 192. Record actual aten SDPA operator names, flags, device UUID, package versions and CUDA capability. This diagnostic runs outside performance timing. Do not infer historical execution from a later profile without stating that it is an inference.

At each principal reversible 50M checkpoint, compare stored and reconstructed gradients on the same deterministic 1x512 random-token input and targets with FP16 autocast and identical loss scaling. Check every parameter, global relative gradient error <=1e-2, finite gradients, forward loss equality, and the existing inverse probe <=1e-3. Retain per-parameter relative and absolute errors plus reference norms to make near-zero cases visible. Checkpoint hashes must match their original final summaries. Preserve failures and investigate; never relax thresholds silently. This complements the already-passing trained CPU FP32 probes; it is not a representative-data generalization test.

These diagnostics consume no committed training targets. Supplemental trained checkpoints can be examined under the same protocol once available. All output remains private/local.
