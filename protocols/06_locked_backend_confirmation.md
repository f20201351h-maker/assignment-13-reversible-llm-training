# Explicit-backend confirmation campaign

Preregistered after the original eight principal runs, before these new runs. Protocol 05 diagnosed a provenance deviation: the original runs used automatic SDPA dispatch instead of an explicitly locked backend. The diagnostic at batches 1, 110, 181 and 192 on Tesla T4 / PyTorch 2.10.0+cu128 observed `aten::_scaled_dot_product_efficient_attention` and its backward operator. All six trained original reversible checkpoints passed the intended mixed-precision gates; global gradient errors were 0.000307–0.000587, below 0.01. This establishes a working backend, not historical telemetry.

## Confirmation design

Run A-L/B-L/C-L/D-L/E-L with seed 1337 and A-L/B-L/C-L with seed 2027, each from initialization for exactly 50,000,000 successful loss-bearing targets. Clone all architecture, data, optimizer, precision, initialization seed, token schedule, physical and effective batch choices from the corresponding original frozen run. The only intended computational change is explicitly enabling the efficient SDPA backend and disabling math, flash and cuDNN alternatives. Any unsupported configuration must fail rather than silently fall back. Distinct `-L-` run IDs preserve the original evidence.

Before training, repeat GPU gates under the explicit backend, record actual forward/backward operator names and runtime flags, and preflight each run configuration with real training updates. Preserve RNG state around profiling so diagnostics cannot change model initialization. Record GPU UUIDs and process telemetry. New backend configuration is serialized in each checkpoint and final summary, and checkpoint evaluation must honor it. CPU evaluation, if permitted, must explicitly identify its diagnostic math fallback rather than claim identical GPU arithmetic.

No method reselection or retuning is permitted from these results. B0=110 and BR=192 remain the proposed batches; a preflight failure would invalidate the affected capacity claim and require a new documented search. The original observed OOM boundaries remain evidence for the automatic-dispatch environment, and any transfer to the forced environment is qualified unless retested.

The completed locked campaign becomes the primary confirmatory table; original results remain a separately identified replication/provenance comparison. Do not pool the original supplemental third seed into locked-backend seed uncertainty. Compare paired original/locked final losses and counters, without using checkpoint byte equality (checkpoint metadata necessarily changes) as a numerical-equivalence test.

## Isolated benchmark amendment

Protocol 03 has not yet executed. Its A–E and checkpointed benchmark conditions will now explicitly select efficient SDPA, with the same 20 warmup / 100 measured successful updates, three fresh processes, assignments 0/1/0, and exact effective batch accumulation. Record runtime flags and actual operator checks outside timed regions. Begin only after active training campaigns finish, with one benchmark worker at a time. Measure original automatic-dispatch and locked capacity boundaries separately if a discrepancy appears; do not silently equate them.

## Decision and evidence

Keep every result, including negative or differing outcomes. Successful confirmation requires all eight exact target budgets, explicit backend policy/telemetry, numerical gates, checkpoint integrity, independently reproduced evaluations, and no unexpected loss or update-counter discrepancy with the corresponding original recipe. This repair uses approved Kaggle quota without reducing training budgets. It is not a claim that the original campaign followed the explicit-backend instruction.
