# Pilot decision (recorded before batch search and full training)

All eight preregistered pilots committed exactly 2,000,000 targets, used physical and effective batch 4, seed 1337, the same training tape prefix, and the same optimizer/schedule. Raw paths: `experiments/kaggle-pilots-v3/runs/pilot-*/{run_config.json,metrics.jsonl,eval.jsonl,final.json}`. Mixed-precision initial-weight gates: `experiments/kaggle-pilots-v3/gpu_correctness.json`. The pilots ran as two independent single-T4 processes, so their campaign throughput includes shared-host effects and is not an isolated benchmark.

| Candidate | Development loss (nats/target) | Initial global relative gradient error |
|---|---:|---:|
| Coupled Euler, h=0.5 | 3.915341 | 4.04e-8 |
| Coupled Euler, h=1.0 | 3.927409 | 4.82e-5 |
| Coupled Euler, h=0.25 | 3.940300 | 4.17e-8 |
| Conventional | 3.954306 | N/A |
| Midpoint, h=0.5 | 3.963086 | 1.72e-4 |
| Midpoint, h=0.25 | 3.963239 | 4.71e-5 |
| Blended midpoint, a=0.5, h=0.25 | 3.966583 | 3.54e-5 |
| Midpoint, h=1.0 | 3.973230 | 5.22e-6 |

The selected formulation is **coupled Euler at h=0.5**. Its development loss is lowest. The h=1.0 result is within the preregistered 0.02-nat tie window, but its measured gradient disagreement is greater. The h=0.25 result is farther than 0.02 nats. Post-training FP32 stored-versus-reconstructed gradient and explicit inverse-drift gates passed for all seven reversible checkpoints; raw measurements are in `experiments/kaggle-pilots-v3/trained_checkpoint_probes.json`. No pilot targets transfer to full runs.
