# Batch-capacity decision before 50M-target runs

The selected formulation is coupled Euler at h=0.5. On Kaggle Tesla T4 workers, the fresh-process search used forward, backward, optimizer-state allocation, and real optimizer updates at every trial. Every reported maximum passed three additional fresh 20-update trials and one fresh sustained 200-update trial. The next integer batch failed with observed CUDA out-of-memory. Raw trial records and failure phases: `experiments/kaggle-batch-v1/batch-search-*/search/trials.jsonl`; summaries: sibling `summary.json` files. The source kernel is private Kaggle `KAGGLE_USER/reversible-llm-session-13-batch-search`, version 1.

| Execution | Largest stable physical batch | Smallest observed OOM batch |
|---|---:|---:|
| Conventional, stored autograd | 110 | 111 |
| Conventional, activation checkpointed | 181 | 182 |
| Coupled Euler, stored autograd | 110 | 111 |
| Coupled Euler, whole-stack reconstruction | 192 | 193 |

Thus the largest common physical batch for the conventional baseline and selected reversible formulation is **B0=110**, while the selected reconstructed formulation's maximum stable batch is **BR=192**. All are per one GPU with `world_size=1`, sequence length 512, FP16 autocast, FP32 parameters/optimizer/residual states, and compilation disabled. The measured maximum is specific to this GPU/software/allocation environment; it is not a hardware-independent architectural constant.

The search probe used a summed cross-entropy loss, while principal training requests an unreduced per-token loss for exact logging. A principal-code preflight at B0 and BR is required before the full job begins. If it fails, the boundary must be remeasured under the exact training path and this decision amended before counting full results.
