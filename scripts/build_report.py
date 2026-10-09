"""Write the research report (REPORT.md) from verified summaries and raw evidence."""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(rows: list[dict]) -> str:
    lines = ['| Run | Physical / effective batch | Holdout loss | Perplexity | Updates | Training targets/s | Allocated GiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda r: (r['seed'], r['condition'])):
        lines.append(f"| [{row['run_id']}]({row['source'].replace(chr(92), '/')}) | "
                     f"{row['physical_batch']} / {row['effective_batch']} | {row['holdout_loss_nats']:.4f} | "
                     f"{row['holdout_perplexity']:.3f} | {row['updates']} | "
                     f"{row['training_tokens_per_second']:,.0f} | {row['peak_allocated_bytes']/2**30:.3f} |")
    return '\n'.join(lines)


def main() -> None:
    rows = json.loads((ROOT/'results/runs.json').read_text())
    full = [row for row in rows if row['committed_targets'] == 50_000_000]
    expected = {(letter, seed) for letter in 'ABC' for seed in (1337, 2027)} | {('D',1337),('E',1337)}
    locked = [row for row in full if row['campaign'] == 'locked']
    locked_complete = expected <= {(r['condition'],r['seed']) for r in locked}
    selected = locked if locked_complete else [r for r in full if r['campaign']=='original']
    if not expected <= {(r['condition'],r['seed']) for r in selected}:
        raise RuntimeError('A complete eight-run campaign is required for this report')
    if not all(r['checkpoint_locally_verified'] for r in full):
        raise RuntimeError('Finish checkpoint download and hash verification before reporting')
    campaign = 'explicitly locked efficient SDPA' if locked_complete else 'original automatic SDPA'
    by = {(r['condition'],r['seed']):r for r in selected}
    a,b,c,d,e = [by[(key,1337)] for key in 'ABCDE']
    br = c['effective_batch']
    memory_saved = 100*(1-b['peak_allocated_bytes']/d['peak_allocated_bytes'])
    throughput_cost = 100*(1-b['training_tokens_per_second']/d['training_tokens_per_second'])
    architecture_delta = d['holdout_loss_nats']-a['holdout_loss_nats']
    reconstruction_delta = b['holdout_loss_nats']-d['holdout_loss_nats']
    batch_delta = e['holdout_loss_nats']-b['holdout_loss_nats']
    physical_delta = c['holdout_loss_nats']-e['holdout_loss_nats']
    ledger = json.loads((ROOT/'audit/ledger.json').read_text()) if (ROOT/'audit/ledger.json').exists() else []
    pending = [r['requirement'] for r in ledger if r['status']!='PASS']
    benchmark_path=ROOT/'experiments/kaggle-benchmark-v2/benchmark/summary.json'
    bench=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else {}
    finalized = locked_complete and len(bench)==7 and ledger and not pending
    status = ('Every check in the evidence ledger ([audit/ledger.md](audit/ledger.md)) passes.' if finalized else
              '**Final verification pending.** See the evidence ledger for the remaining checks. '
              'The table below reports verified measurements, not a completion claim.')
    headline = (f'Reconstruction saved **{memory_saved:.1f}%** of peak allocated memory at the common batch, '
                f'with **{throughput_cost:.1f}% lower campaign throughput** than stored autograd using the same reversible equations '
                f'(seed 1337, {campaign}). Increasing effective batch from 110 to {br} worsened holdout loss by '
                f'**{batch_delta:.3f} nats/target** under the fixed recipe. More capacity did not automatically improve training.')
    controls = f'''| Condition | Architecture / backward | Physical batch | Effective batch | Interpretation |
|---|---|---:|---:|---|
| A | Conventional / stored | 110 | 110 | Baseline |
| D | Coupled Euler / stored | 110 | 110 | A to D changes architecture |
| B | Coupled Euler / reconstruction | 110 | 110 | D to B changes backward implementation |
| E | Coupled Euler / reconstruction | 110, then {br-110} | {br} | B to E changes effective batch |
| C | Coupled Euler / reconstruction | {br} | {br} | E to C changes physical batching |
'''
    methods = '''All main models have **19,969,152 unique trainable parameters**: 9 blocks, width 384, 6 heads of width 64, GELU FFN width 1,536, context 512 and vocabulary 10,000. They use bias-free pre-LayerNorm, learned positions, tied token/output weights, no linear biases and no dropout. Duplicating a residual state adds no learned weights.

The conventional block is `x + F(x)`, where `F(x) = A(x) + M(x + A(x))`. Midpoint uses an Euler bootstrap `p1 = p0 + 0.5*h*F0(p0)` and eight steps `p[l+1] = p[l-1] + 2*h*F[l](p[l])`. Its inverse subtracts the recomputed increment. Blended midpoint adds coefficients `a` and `1-a` to the two predecessor states; the course-related diagnostic uses `a=0.5, h=0.25`. Coupled Euler uses `u' = u + h*A(v)` followed by `v' = v + h*M(u')`, then reverses these operations in the opposite order. Its two initial streams equal the embedding; their final mean feeds the shared normalization and output head. Arbitrary attention/MLP functions are not assumed to conserve Hamiltonian energy.

Whole-stack custom autograd retains boundary states and reconstructs hidden activations one block at a time. Embeddings, midpoint bootstrap, final normalization/head and transient workspace remain in memory accounting. Stored and reconstructed implementations use identical forward equations; ordinary residual Euler is not given an invalid subtraction-at-output inverse.

AdamW uses learning rate 3e-4, betas (0.9,0.95), epsilon 1e-8, and weight decay 0.1 on linear matrices, excluding embeddings and norm scales. After token-normalized accumulation, gradients are unscaled and clipped to norm 1. The schedule warms up for 1M targets and decays by cosine to 3e-5 at 50M. Parameters, optimizer state, residual states and reconstruction arithmetic stay FP32; suitable operations use FP16 autocast and GradScaler on T4. Compilation is disabled. Each worker is a separate single-GPU model with world_size=1; two workers are not distributed training.
'''
    data = '''TinyStories is pinned at revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. A byte-level BPE tokenizer is trained on the first 100,000 unique training documents only. Exact normalized-text duplicates and train/evaluation overlaps are removed. Official validation documents are deterministically hash-partitioned into development (262,144 targets) and final holdout (1,000,000 targets). Training consumes the identical ordered 50M-target stream across runs; seeds vary initialization. End-of-document tokens are counted as targets, and causal attention continues across packed story boundaries.

Every completed run commits exactly 50,000,000 non-padding next-token targets. Padding is masked; the final partial batch is normalized by its actual target count. AMP-skipped updates replay the same data without advancing the committed cursor or schedule. Attempted targets, successful targets, retries, padding and separators are logged separately. Pilots, validation, demonstrations and benchmarks do not contribute to assignment training totals. The first 131,072 training targets form the fixed training-subset evaluation; this subset and the held-out split need not have identical difficulty.
'''
    reproduction = '''```powershell
python -m pip install -e '.[test,report]'
python -m pytest -q tests
python -m revllm.cli prepare --out data
python scripts/reproduce_run.py --run A-L-1337 --data-dir data --out experiments/reproduction/A-L-1337
python -m revllm.cli evaluate --config experiments/reproduction/A-L-1337/run_config.json --checkpoint experiments/reproduction/A-L-1337/latest.pt --data-dir data --out results/A-L-1337-evaluation.json
python scripts/analyze.py
python scripts/analyze_benchmark.py
python scripts/report_figures.py
python scripts/build_notebooks.py
python scripts/audit.py
python scripts/build_report.py
python scripts/export_report.py
```

Run the frozen GPU recipes on CUDA hardware; they require efficient SDPA and do not silently fall back during training. CPU correctness tests and notebook demonstrations use smaller configurations. Independent CPU checkpoint evaluation explicitly records its math-backend fallback and is not claimed to reproduce FP16 GPU arithmetic. To resume, repeat `reproduce_run.py` with `--resume` and the same output directory. Frozen recipes live in `configs/locked/`; original and supplemental recipes remain separate.

Public reproduction needs no private Kaggle account: regenerate data from its public source, verify `data/manifest.json` and train the frozen recipes. Large tapes and checkpoints are excluded from Git. Their hashes are retained in raw final summaries; authenticated private retrieval commands and source versions are in `audit/artifact-index.md`. The GPU environment is PyTorch 2.10.0+cu128 on Tesla T4; the Windows CPU dependency lock is separate from recorded GPU package metadata.
'''
    references = '''- Gal et al., *Reversing Large Language Models for Efficient Training and Fine-Tuning* (2025), [arXiv:2512.02056](https://arxiv.org/abs/2512.02056): reversible formulations and recomputation/storage rationale. Its reported performance is not substituted for our measurements.
- Schaipp, *How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size* (2026), [arXiv:2607.01487](https://arxiv.org/abs/2607.01487): relates token budget, steps and batch. It motivates an optimization interpretation, not a causal proof for this case study.
- [TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories): dataset and CDLA-Sharing-1.0 license.
- [PyTorch 2.10 backend documentation](https://docs.pytorch.org/docs/2.10/backends.html): explicit SDPA selection controls.
- [Source register](literature/sources.md) records the fixed nanoGPT, build-nanogpt and nanochat commits and the released course implementation. Course transcript/lecture files are not redistributed.
'''
    capacity = ('Original automatic-dispatch batch searches observed stable/OOM pairs 110/111 (conventional stored), '
                '181/182 (conventional checkpointed), 110/111 (coupled stored), and 192/193 (coupled reconstructed). '
                'Each boundary passed three fresh 20-update confirmations and a sustained 200-update test. '
                'The reconstructed capacity is 74.5% higher than conventional stored under that search environment. '
                'These original OOM measurements are not silently relabeled as new locked-backend searches.')
    capacity += (' Locked efficient-SDPA searches measured 110/111, 182/183, 110/111 and 191/192 respectively. '
                 'The 192-batch locked training attempt failed on its third update; protocol 08 retains this failure and '
                 'uses new 191-batch C/E identities. The capacity gain under the locked search is 73.6%. '
                 'The one-sequence differences are not attributed solely to the backend.')
    benchmark_text='Isolated measurements are pending. Campaign-average throughput includes actual update and replay time, excludes evaluation/checkpoint time, and comes from two simultaneous independent workers. It is not an isolated throughput claim.'
    if len(bench)==7:
        lines=['| Condition | Median targets/s | Observed min–max | Allocated / reserved GiB |','|---|---:|---:|---:|']
        for name,r in sorted(bench.items()):
            lines.append(f"| {name} | {r['median_targets_per_second']:,.0f} | {r['min_targets_per_second']:,.0f}–{r['max_targets_per_second']:,.0f} | {r['peak_allocated_bytes']/2**30:.3f} / {r['peak_reserved_bytes']/2**30:.3f} |")
        benchmark_text='\n'.join(lines)+'\n\nEach condition uses 20 successful warmup updates and 100 measured updates in each of three fresh processes. Device assignments are 0,1,0 and one worker runs at a time. Rates divide committed measured targets by synchronized elapsed time; bars show the complete observed repetition range. Synthetic-token optimization is a performance probe, not a language-quality result.'
        comparison_path=ROOT/'results/benchmark_comparisons.json'
        comparisons=json.loads(comparison_path.read_text())['comparisons']
        slowdown=[100*(1-r['throughput_ratio']) for r in comparisons if r['contrast']=='reconstruction']
        physical=[r['throughput_ratio'] for r in comparisons if r['contrast']=='physical_batch']
        benchmark_text += (f'\n\nThe GPU-1 repetition is faster than both GPU-0 repetitions for every condition. '
                           f'Device identity and repetition order are confounded; these observations do not prove hardware throttling. '
                           f'Comparing matching assignments/repetition indices, reconstruction reduces throughput by '
                           f'{min(slowdown):.1f}–{max(slowdown):.1f}% (D to B). The physical/accumulated throughput ratio '
                           f'is {min(physical):.3f}–{max(physical):.3f} (E to C), so larger physical batching provides no '
                           f'observed speedup here. [Generated comparisons](results/benchmark_comparisons.csv) retain each ratio.'
                           '\n\n![Benchmark device and repetition variation](results/figures/benchmark_device_variation.png)')
    limitations = '''The original campaign omitted an explicit backend lock. Later profiling found efficient attention at the tested shapes, but cannot establish historical operator execution. Protocol 06 preserves that deviation and adds separate locked confirmations. No original result is overwritten or relabeled.

The architectural contrast is bundled: coupling, state duplication, normalization inputs and final stream averaging change together. The fixed recipe does not retune learning rate or schedule for the larger batch. Larger effective batch reduces the number of optimizer updates, but gradient noise and optimization mechanisms were not directly measured. Two main seeds support descriptive variation, not reliable population-level confidence intervals or significance claims. A third automatic-backend seed is shown separately, not pooled with locked confirmations.

The whole output logits/loss workspace remains ordinary and identical across main conditions, so constant-depth stack storage does not imply constant total memory or unlimited batch capacity. No chunked-head optimization is included in the principal results. Capacity is specific to the recorded GPU/software/allocator environment. GPU quota hours and physical GPU-hours are not assumed equal. The `job_seconds` field covers the training loop and subsequent evaluations/checkpointing; kernel startup/download overhead is not included in that per-run field and must not be presented as whole-session time.

Findings are limited to this small decoder, context 512, TinyStories, and the tested training recipe. No novelty claim, large-model generalization, or required positive result is made.
'''
    report = f'''# Reversible LLM training: memory saved, compute spent

{status}

## What the experiment answers

{headline}

This report is generated by `scripts/build_report.py` from raw-log-verified summaries. The selected table uses **{campaign}**, with {len(selected)} runs, each at 50M targets. Across retained full-budget campaigns, {len(full)} runs account for {sum(r['committed_targets'] for r in full):,} committed targets; this cumulative total is not one model's training budget.

## Controls

{controls}

## Model, equations and recipe

{methods}

## Data and exact accounting

{data}

## Selection before confirmation

Eight equal 2M-target pilots investigated conventional, midpoint at h=0.25/0.5/1.0, coupled Euler at the same steps, and blended midpoint at a=0.5,h=0.25. Initial mixed-precision and trained FP32 numerical gates passed. Coupled Euler h=0.5 was selected by the frozen rule: development loss 3.915341 versus midpoint h=0.5 at 3.963086. Full runs restarted from initialization; pilot targets do not count toward them. All candidate results and failures are retained.

## Full-budget results

{table(selected)}

All losses are nats per next-token target; perplexity is exp(loss). Throughput here is campaign throughput, not the isolated benchmark. Allocated memory includes optimizer state. Exact values, reserved memory, final-update loss, last-1M target-weighted loss, fixed-training-subset loss, padding, separators and hashes are in [runs.csv](results/runs.csv).

![Development loss versus targets and update time](results/figures/development_loss_curves.png)

Individual seed curves are shown without smoothing of development evaluations. Time is accumulated training-update time. The training-only graph below overlays a 25-update centered rolling mean on faint raw updates; endpoints are padded with their boundary values. It is a visualization, not additional observations.

![Training loss trajectories](results/figures/training_loss_curves.png)

## What changes caused which result?

For seed 1337, A to D changes holdout loss by **{architecture_delta:+.6f}** nats/target. The stored reversible architecture already obtains the loss improvement; it cannot be attributed to reconstruction. D to B changes loss by **{reconstruction_delta:+.6f}**, saves **{memory_saved:.2f}%** of allocated memory, and reduces campaign throughput by **{throughput_cost:.2f}%**. This is the direct memory/recomputation trade-off.

B to E raises loss by **{batch_delta:+.6f}** nats/target when effective batch increases. E to C changes loss by only **{physical_delta:+.6f}** at matched effective batch. The large quality difference therefore follows the effective-batch change in this comparison rather than the choice between physical and accumulated batching. Updates decrease from **{b['updates']}** to **{c['updates']}** at the same token budget; this is consistent with an optimization explanation, but does not establish its detailed mechanism.

![Causal-control contrasts](results/figures/causal_loss_chain_seed1337.png)

The contrast panel uses one seed for D/E; its small differences are not significance tests. A versus C changes architecture, backward implementation and batch together and is interpreted only as a combined practical outcome.

## Memory capacity and performance

{capacity}

![Measured capacity boundaries](results/figures/batch_capacity_boundaries.png)

![Memory and throughput against physical batch](results/figures/batch_memory_and_throughput.png)

These capacity-search curves show successful 20-update probes. Throughput ranges reflect repeated trials where available; points with one trial do not estimate variability. They are not the isolated benchmark.

{benchmark_text}

![Memory and throughput trade-off](results/figures/memory_throughput.png)

## Seed variation and additional conditions

![Individual seed results](results/figures/seed_variation_holdout.png)

Points are individual seeds; spans show observed minima/maxima, not confidence intervals. Supplemental automatic-dispatch midpoint, checkpointing and seed-3141 experiments remain separate sensitivity evidence:

{table([r for r in full if r['campaign']=='supplemental'])}

## Numerical evidence

Tiny CPU FP64/FP32 tests check inverses, custom gradients, finite differences, duplicated inputs, bootstrap, tied embeddings, causality, masking and accumulation. Depth probes measure saved activation storage excluding parameters. Checkpoint/resume tests exercise next-batch identity and updates, including interrupted logs and overflow replay. Trained checkpoint checks retain per-parameter error information and global gates rather than relying only on matching losses.

![Unique retained activation storage](results/figures/activation_storage_vs_depth.png)

The storage diagnostic counts unique tensor-storage allocations retained for the stack backward, excluding parameter storage and its aliases. It includes the squared-output reduction, uses a tiny CPU FP32 model and does not measure temporary workspace or full-model GPU peak memory. At depths 2, 4, 9 and 18, reconstructed storage remains constant while stored autograd grows with depth. This establishes the intended stack-storage property separately from the whole-model GPU measurement.

![Numerical errors across depth and trained checkpoints](results/figures/numerical_error_depth_and_progress.png)

The training-progress panel compares checkpoints from different runs, not a longitudinal reconstruction-error trajectory. FP64 gates are 1e-9 inverse / 1e-7 gradient; FP32 1e-5 / 1e-4; intended FP16 autocast 1e-3 / 1e-2. Thresholds were not relaxed. Exact gate records and independent checkpoint-evaluation files are linked by the [evidence ledger](audit/ledger.md).

## Limitations and negative findings

{limitations}

## Reproduction

{reproduction}

## References and research trail

{references}

'''
    (ROOT/'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps({'main_campaign':campaign,'full_runs':len(full),'audit_complete':bool(finalized),
                      'outputs':['REPORT.md']}))


if __name__=='__main__':
    main()
