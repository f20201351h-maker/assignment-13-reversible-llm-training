# Capacity-informed confirmation repair

Preregistered 2026-09-27 before the repaired training and benchmark campaigns.

Protocol 07 measured locked efficient-SDPA stable/OOM pairs of 110/111 conventional stored, 182/183 conventional checkpointed, 110/111 coupled stored, and 191/192 coupled reconstructed. Each stable limit passed three fresh 20-update confirmations and a sustained 200-update trial. These differ from automatic-dispatch capacity by one sequence for checkpointing and reconstruction. Allocator/shape sensitivity is plausible, not established as the mechanism.

Protocol 06 C-L-1337 at batch 192 committed 196,608 targets in two updates, then failed in backward with CUDA OOM on the third. It has no durable training checkpoint. Its raw log is retained and its targets are excluded from completed-run totals. A-L-1337, B-L-1337 and D-L-1337 completed; retain them unchanged. No changes to equations, precision, optimizer, allocator or numerical gates are authorized by this repair.

Use the measured locked BR=191, B0=110. New IDs C-L191-1337 and C-L191-2027 use physical/effective batch 191. E-L191-1337 uses physical 110 followed by 81, effective 191. All restart at initialization and commit exactly 50M targets. The unstarted A-L-2027 and B-L-2027 retain their frozen configs. These five runs plus the three completed runs form the eight-run locked campaign. Failed C-L-1337 and unexecuted old 192-batch configs remain historical, not main results.

Before the five runs, each replacement C/E receives a fresh 200-update exact-training-path preflight (separate data/output identity and excluded training totals). Failure stops promotion and requires a further recorded investigation; do not silently lower batch size.

The already launched isolated benchmark v1 tests the old 192/181 choices and is retained as diagnostic evidence. After it ends, benchmark v2 uses C/E effective 191 and checkpointed-max 182; all other conditions unchanged. Run all seven conditions with 20 warmup and 100 measured successful updates, three fresh processes on GPUs 0,1,0 and one worker at a time. No concurrent campaign training during either isolated benchmark job. Do not combine v1/v2 repetitions.

Independent evaluation and mixed-precision checkpoint checks must include the repaired IDs and the original three completed locked IDs. Reports must show both historical and locked capacity measurements and avoid attributing their difference solely to the backend.
