# Supplemental 50M-target protocol

Fixed before launching supplemental experiments. The principal A–E and seed-2027 runs remain the basis of the causal-control analysis; these runs add formulation and seed checks. Every row restarts from its own initialization and commits exactly 50,000,000 next-token targets on the same fixed training tape, with the same optimizer, precision, schedule, evaluation, and checkpoint rules as the principal runs.

| Run | Model/backward | Physical/effective batch | Purpose |
|---|---|---:|---|
| F-midpoint-1337 | midpoint h=0.5 / reconstructed | 110/110 | Other passing reversible formulation at the common batch |
| G-checkpointed-1337 | conventional / activation checkpointed | 110/110 | Conventional compute-for-memory control at the common batch |
| A-3141 | conventional / stored | 110/110 | Third seed for the architecture/practical comparison |
| B-3141 | coupled Euler h=0.5 / reconstructed | 110/110 | Third seed for the fixed-batch comparison |
| C-3141 | coupled Euler h=0.5 / reconstructed | 192/192 | Third seed for the larger-batch practical outcome |

The third seed is 3141, fixed here rather than chosen after seeing its outcome. The supplemental kernel first performs a one-update exact-training-path preflight for every row. A failed preflight or run is retained and reported; no lower batch is silently substituted. These rows are launched only after the principal job ends and available quota is checked. If quota or platform behavior prevents completion, report the missing rows as unverified rather than merging partial results into the principal evidence.

During paired runs, sample both T4 identifiers, utilization, memory use, and compute-app process mappings every ten seconds. Keep the raw monitor log alongside each worker's independent update log. Monitoring failure must not terminate training; it lowers the strength of the device-mapping claim.
