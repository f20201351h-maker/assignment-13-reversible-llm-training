"""Generate the final quantitative report figures from recorded experiment artifacts.

Regenerate from the project root with::

    python scripts/report_figures.py

The script discovers completed supplemental runs when they are present and switches
the memory/throughput figure to the isolated benchmark once its summary exists.
Every figure is written as vector PDF/SVG plus a 300 dpi PNG preview.  The adjacent
``figure_provenance.json`` records the exact inputs and their SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from orx_figstyle import BASELINE, MUTED, PALETTE, WIDE, figure, figure_grid, panel_labels, save, si_ticks, use_style


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUT = ROOT / "results" / "figures"
BENCHMARK = EXPERIMENTS / "kaggle-benchmark-v2" / "benchmark" / "summary.json"

CONDITION_COLORS = {
    "A": PALETTE["blue"],
    "B": PALETTE["orange"],
    "C": PALETTE["green"],
    "D": PALETTE["purple"],
    "E": PALETTE["red"],
    "F": PALETTE["cyan"],
    "G": BASELINE,
}
CONDITION_LABELS = {
    "A": "A  conventional, stored, batch 110",
    "B": "B  coupled, reconstructed, batch 110",
    "C": "C  coupled, reconstructed, batch 192",
    "D": "D  coupled, stored, batch 110",
    "E": "E  coupled, reconstructed, effective batch 192",
    "F": "F  midpoint, reconstructed, batch 110",
    "G": "G  conventional, checkpointed, batch 110",
}
SEED_STYLES = ["-", (0, (5, 2)), (0, (1.5, 1.5)), (0, (5, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "P", "X", "v"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def condition(run_id: str) -> str:
    first = run_id.split("-", 1)[0]
    return first if first in CONDITION_LABELS else run_id


def discover_runs() -> list[dict]:
    """Load all full-budget campaigns while keeping their identities separate."""
    roots = [
        ("original", EXPERIMENTS / "kaggle-full-v1" / "runs"),
        ("locked", EXPERIMENTS / "kaggle-locked-v1" / "runs"),
        ("locked", EXPERIMENTS / "kaggle-locked-repair-v1" / "runs"),
        ("supplemental", EXPERIMENTS / "kaggle-supplemental-v1" / "runs"),
    ]
    runs = []
    seen = set()
    for campaign, run_root in roots:
        if not run_root.exists():
            continue
        for final_path in sorted(run_root.glob("*/final.json")):
            final = read_json(final_path)
            run_id = final.get("run_id", final_path.parent.name)
            identity = (campaign, run_id)
            if identity in seen or final.get("status") != "COMPLETE":
                continue
            config_path = final_path.parent / "run_config.json"
            if not config_path.exists():
                continue
            config = read_json(config_path)
            metrics_path = final_path.parent / "metrics.jsonl"
            eval_path = final_path.parent / "eval.jsonl"
            metrics = [row for row in read_jsonl(metrics_path) if row.get("status") == "COMMITTED"]
            evaluations = read_jsonl(eval_path)
            if not metrics or not evaluations:
                continue
            target = int(config.get("target_tokens", final.get("committed_targets", 0)))
            if int(final.get("committed_targets", 0)) != target:
                continue
            runs.append({
                "run_id": run_id,
                "condition": condition(run_id),
                "campaign": campaign,
                "seed": int(config["seed"]),
                "config": config,
                "final": final,
                "metrics": metrics,
                "evaluations": evaluations,
                "paths": [final_path, config_path, metrics_path, eval_path],
            })
            seen.add(identity)
    return runs


def select_campaign(runs: list[dict]) -> tuple[list[dict], str, str]:
    """Prefer locked evidence only once the complete matched eight-run set exists."""
    expected = {(key, seed) for key in "ABC" for seed in (1337, 2027)}
    expected.update({("D", 1337), ("E", 1337)})
    locked = [run for run in runs if run["campaign"] == "locked"]
    locked_keys = {(run["condition"], run["seed"]) for run in locked}
    if expected <= locked_keys:
        selected = [run for run in locked if (run["condition"], run["seed"]) in expected]
        return selected, "locked", "protocols 06/08 (efficient SDPA locked)"
    original = [run for run in runs if run["campaign"] == "original" and
                (run["condition"], run["seed"]) in expected]
    return original, "original", "kaggle-full-v1 (original auto SDPA)"


def smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    """Centered 25-update rolling mean; endpoint values pad the boundaries."""
    if len(values) < 3:
        return values.copy()
    width = min(window, len(values) if len(values) % 2 else len(values) - 1)
    width = max(3, width)
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def save_all(fig, name: str) -> list[Path]:
    stem = OUTPUT / name
    save(fig, str(stem), formats=("pdf", "svg"), close=False)
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=300, format="png")
    plt.close(fig)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".svg"), png]


def common_y_limits(arrays: list[np.ndarray], *, floor_zero: bool = False) -> tuple[float, float]:
    values = np.concatenate([np.asarray(x, dtype=float) for x in arrays])
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    margin = max(0.04 * (high - low), 0.02)
    return (max(0.0, low - margin) if floor_zero else low - margin, high + margin)


def seed_legend(ax, runs: list[dict], *, ncol: int = 3) -> None:
    conditions = sorted({run["condition"] for run in runs})
    seeds = sorted({run["seed"] for run in runs})
    handles = [Line2D([0], [0], color=CONDITION_COLORS[key], lw=1.8, label=CONDITION_LABELS[key])
               for key in conditions]
    handles.extend(Line2D([0], [0], color="#333333", lw=1.3,
                          linestyle=SEED_STYLES[index % len(SEED_STYLES)], label=f"seed {seed}")
                   for index, seed in enumerate(seeds))
    ax.figure.legend(handles=handles, loc="outside lower center", ncol=2,
                     handlelength=2.4, columnspacing=1.2, fontsize=6.5)


def plot_training_curves(runs: list[dict], campaign_label: str) -> tuple[list[Path], list[Path]]:
    selected = [run for run in runs if run["condition"] in "ABC"]
    if not selected:
        return [], []
    seed_index = {seed: i for i, seed in enumerate(sorted({run["seed"] for run in selected}))}

    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.48, sharey=True)
    all_losses = []
    for run in selected:
        rows = run["metrics"]
        loss = np.asarray([row["loss_nats"] for row in rows], dtype=float)
        tokens = np.asarray([row["committed_targets"] for row in rows], dtype=float) / 1e6
        minutes = np.asarray([row["training_seconds"] for row in rows], dtype=float) / 60
        color = CONDITION_COLORS[run["condition"]]
        style = SEED_STYLES[seed_index[run["seed"]] % len(SEED_STYLES)]
        for ax, x in zip(axes, (tokens, minutes)):
            ax.plot(x, loss, color=color, alpha=0.10, linewidth=0.55)
            ax.plot(x, smooth(loss), color=color, linestyle=style, linewidth=1.25)
        all_losses.append(loss)
    axes[0].set_xlabel("Committed training targets (millions)")
    axes[1].set_xlabel("Elapsed training time (minutes)")
    axes[0].set_ylabel("Training loss (nats/target)")
    axes[0].set_ylim(*common_y_limits(all_losses))
    axes[0].set_xlim(left=0)
    axes[1].set_xlim(left=0)
    panel_labels(axes)
    seed_legend(axes[0], selected)
    axes[1].text(0.98, 0.96, "faint: raw update\nline: 25-update rolling mean",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=6.5, color="#555555")
    fig.suptitle(campaign_label, fontsize=7, color="#555555")
    paths = save_all(fig, "training_loss_curves")

    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.48, sharey=True)
    all_dev = []
    for run in selected:
        rows = run["evaluations"]
        targets = np.asarray([row["committed_targets"] for row in rows], dtype=float)
        loss = np.asarray([row["dev_loss_nats"] for row in rows], dtype=float)
        metric_targets = np.asarray([row["committed_targets"] for row in run["metrics"]], dtype=float)
        metric_minutes = np.asarray([row["training_seconds"] for row in run["metrics"]], dtype=float) / 60
        minutes = np.interp(targets, metric_targets, metric_minutes)
        color = CONDITION_COLORS[run["condition"]]
        style = SEED_STYLES[seed_index[run["seed"]] % len(SEED_STYLES)]
        for ax, x in zip(axes, (targets / 1e6, minutes)):
            ax.plot(x, loss, color=color, linestyle=style, marker="o", markersize=2.8,
                    markerfacecolor="white", markeredgewidth=0.7)
        all_dev.append(loss)
    axes[0].set_xlabel("Committed training targets (millions)")
    axes[1].set_xlabel("Elapsed training time (minutes)")
    axes[0].set_ylabel("Development loss (nats/target)")
    axes[0].set_ylim(*common_y_limits(all_dev))
    axes[0].set_xlim(left=0)
    axes[1].set_xlim(left=0)
    panel_labels(axes)
    seed_legend(axes[0], selected)
    fig.suptitle(campaign_label, fontsize=7, color="#555555")
    dev_paths = save_all(fig, "development_loss_curves")
    return paths, dev_paths


def plot_causal_chain(runs: list[dict], campaign_label: str) -> list[Path]:
    by_key = {(run["condition"], run["seed"]): run for run in runs}
    order = ["A", "D", "B", "E", "C"]
    chain = [by_key.get((key, 1337)) for key in order]
    if any(run is None for run in chain):
        return []
    losses = np.asarray([run["final"]["holdout_loss_nats"] for run in chain], dtype=float)
    deltas = np.diff(losses)
    contrasts = ["architecture", "reconstruction", "effective batch", "physical batch"]

    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.45, gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(order))
    axes[0].plot(x, losses, color=BASELINE, linewidth=0.9, zorder=1)
    for index, (key, value) in enumerate(zip(order, losses)):
        axes[0].scatter(index, value, color=CONDITION_COLORS[key], marker=MARKERS[index], s=38,
                        edgecolor="white", linewidth=0.6, zorder=3)
        axes[0].annotate(f"{value:.3f}", (index, value), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=6.5)
    axes[0].set_xticks(x, order)
    axes[0].set_xlabel("Matched condition chain (seed 1337)")
    axes[0].set_ylabel("Holdout loss (nats/target)")
    axes[0].set_ylim(*common_y_limits([losses]))
    fig.suptitle(campaign_label, fontsize=7, color="#555555")

    colors = [PALETTE["green"] if value < 0 else PALETTE["red"] for value in deltas]
    bars = axes[1].barh(np.arange(len(deltas)), deltas, color=colors, height=0.58,
                        edgecolor="white", linewidth=0.6)
    axes[1].axvline(0, color="#333333", linewidth=0.7)
    axes[1].set_yticks(np.arange(len(deltas)), contrasts)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Change in holdout loss (nats/target)")
    axes[1].grid(axis="x")
    axes[1].grid(axis="y", visible=False)
    for bar, value in zip(bars, deltas):
        axes[1].text(value + math.copysign(0.008, value), bar.get_y() + bar.get_height() / 2,
                     f"{value:+.3f}", ha="left" if value >= 0 else "right", va="center", fontsize=6.5)
    bound = max(abs(deltas)) * 1.22
    axes[1].set_xlim(-bound, bound)
    panel_labels(axes)
    return save_all(fig, "causal_loss_chain_seed1337")


def load_batch_evidence() -> list[dict]:
    evidence = []
    base = EXPERIMENTS / "kaggle-batch-v1"
    locked_base = EXPERIMENTS / "kaggle-locked-batch-v1"
    locked = list(locked_base.glob('batch-search-*/search/summary.json'))
    if len(locked) == 4 and all(read_json(path).get('status')=='VALIDATED' for path in locked):
        base = locked_base
    for path in sorted(base.glob("batch-search-*/search/summary.json")):
        summary = read_json(path)
        if summary.get("status") != "VALIDATED":
            continue
        evidence.append({
            "name": path.parent.parent.name.removeprefix("batch-search-"),
            "summary": summary,
            "summary_path": path,
            "trials_path": path.parent / "trials.jsonl",
        })
    return evidence


def plot_batch_capacity(evidence: list[dict]) -> list[Path]:
    order = ["conventional", "coupled_stored", "checkpointed", "coupled_reconstructed"]
    labels = {
        "conventional": "Conventional / stored",
        "coupled_stored": "Coupled / stored",
        "checkpointed": "Conventional / checkpointed",
        "coupled_reconstructed": "Coupled / reconstructed",
    }
    by_name = {item["name"]: item for item in evidence}
    selected = [by_name[name] for name in order if name in by_name]
    if not selected:
        return []
    stable = np.asarray([item["summary"]["largest_stable_batch"] for item in selected], dtype=float)
    oom = np.asarray([item["summary"].get("smallest_observed_oom_batch", np.nan) for item in selected], dtype=float)
    y = np.arange(len(selected))
    colors = [PALETTE["blue"], PALETTE["purple"], PALETTE["orange"], PALETTE["green"]]

    fig, ax = figure(width=WIDE, ratio=0.48)
    fig.suptitle('Efficient SDPA locked' if 'kaggle-locked-batch-v1' in str(selected[0]['summary_path'])
                 else 'Original automatic SDPA', fontsize=7, color='#555555')
    bars = ax.barh(y, stable, color=colors[:len(selected)], height=0.56, edgecolor="white", linewidth=0.6)
    ax.scatter(oom, y, marker="x", s=34, linewidth=1.4, color=PALETTE["red"], zorder=4,
               label="Smallest observed OOM batch")
    ax.set_yticks(y, [labels[item["name"]] for item in selected])
    ax.invert_yaxis()
    ax.set_xlabel("Physical sequences per GPU")
    ax.set_ylabel("Training method")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for bar, stable_value, oom_value in zip(bars, stable, oom):
        ax.text(stable_value - 3, bar.get_y() + bar.get_height() / 2, f"stable {stable_value:.0f}",
                color="white", ha="right", va="center", fontsize=7, fontweight="bold")
        if np.isfinite(oom_value):
            ax.annotate(f"OOM {oom_value:.0f}", (oom_value, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5)
    ax.set_xlim(0, max(np.nanmax(oom), np.max(stable)) * 1.14)
    ax.legend(loc="lower right")
    return save_all(fig, "batch_capacity_boundaries")


def plot_memory_throughput(runs: list[dict], campaign_label: str) -> tuple[list[Path], dict]:
    fig, ax = figure(width=WIDE, ratio=0.56)
    metadata = {}
    if BENCHMARK.exists() and len(read_json(BENCHMARK)) == 7:
        summary = read_json(BENCHMARK)
        items = sorted(summary.values(), key=lambda row: row["condition"])
        for index, row in enumerate(items):
            key = condition(row["condition"])
            x = row["peak_allocated_bytes"] / 2**30
            y = row["median_targets_per_second"]
            lo = y - row["min_targets_per_second"]
            hi = row["max_targets_per_second"] - y
            ax.errorbar(x, y, yerr=np.asarray([[lo], [hi]]), fmt=MARKERS[index % len(MARKERS)],
                        color=CONDITION_COLORS.get(key, BASELINE), markersize=5.5, capsize=2,
                        markeredgecolor="white", markeredgewidth=0.5)
            offset = {'B':(5,-13), 'E':(5,8), 'D':(5,-13),
                      'checkpointed-B0':(-65,0),'checkpointed-max':(-55,12)}.get(row['condition'],(5,5))
            label = {'checkpointed-B0':'Checkpointed\n110',
                     'checkpointed-max':'Checkpointed\n182'}.get(row['condition'],row['condition'])
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=6.5)
        ax.set_ylabel("Isolated benchmark throughput (targets/s)")
        ax.text(0.99, 0.03, "median and min–max, 3 isolated repetitions",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555555")
        metadata = {"mode": "isolated_benchmark", "benchmark_summary": BENCHMARK}
    else:
        points = [run for run in runs if run["final"].get("peak_allocated_bytes") is not None]
        grouped = defaultdict(list)
        for run in points:
            grouped[run["condition"]].append(run)
        for index, key in enumerate(sorted(grouped)):
            group = grouped[key]
            xs = np.asarray([run["final"]["peak_allocated_bytes"] / 2**30 for run in group])
            ys = np.asarray([run["final"]["training_tokens_per_second"] for run in group])
            jitter = np.linspace(-0.035, 0.035, len(group)) if len(group) > 1 else np.zeros(1)
            ax.scatter(xs, ys, color=CONDITION_COLORS.get(key, BASELINE),
                       marker=MARKERS[index % len(MARKERS)], s=35, edgecolor="white", linewidth=0.5,
                       label=f"{key} (n={len(group)})")
        ax.set_ylabel("Campaign-average training throughput (targets/s)")
        ax.text(0.99, 0.03, f"CAMPAIGN MEASUREMENTS — not isolated\n{campaign_label}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                color=PALETTE["red"], fontweight="bold")
        ax.legend(loc="upper left", ncol=2)
        metadata = {"mode": "campaign_fallback", "benchmark_summary": None}
    ax.set_xlabel("Peak allocated GPU memory (GiB)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    return save_all(fig, "memory_throughput"), metadata


def plot_benchmark_devices() -> list[Path]:
    names = ['A','B','C','D','E','checkpointed-B0','checkpointed-max']
    fig, ax = figure(width=WIDE, ratio=0.48)
    for rep, gpu in enumerate((0,1,0)):
        records = [read_json(BENCHMARK.parent/f'{name}-rep{rep}.json') for name in names]
        rates = [r['median_tokens_per_second'] for r in records]
        ax.scatter(np.arange(len(names))+(rep-1)*0.12,rates,
                   color=PALETTE['blue'] if gpu==0 else PALETTE['orange'],
                   marker=('o','s','^')[rep],s=32,label=f'GPU {gpu}, repetition {rep+1}')
    ax.set_xticks(np.arange(len(names)), ['A','B','C','D','E','Checkpointed\n110','Checkpointed\n182'])
    ax.set_ylabel('Isolated throughput (targets/s)')
    ax.set_xlabel('Condition; every point is 100 measured updates')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper center',ncol=3,fontsize=6.5)
    fig.suptitle('Device identity and repetition order are confounded; no causal hardware claim',fontsize=7)
    return save_all(fig,'benchmark_device_variation')


def plot_batch_curves(evidence: list[dict]) -> list[Path]:
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.50)
    colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['purple'], PALETTE['green']]
    for item, color in zip(evidence, colors):
        grouped = defaultdict(list)
        for trial in read_jsonl(item['trials_path']):
            if trial.get('status') == 'PASS' and trial.get('updates_requested') == 20:
                grouped[trial['batch']].append(trial['measurement'])
        batches = sorted(grouped)
        memory = [max(r['peak_allocated_bytes'] for r in grouped[b])/2**30 for b in batches]
        rates = [[r['tokens_per_second'] for r in grouped[b]] for b in batches]
        medians = np.asarray([np.median(r) for r in rates])
        label = item['name'].replace('_', ' ')
        axes[0].plot(batches, memory, '-o', color=color, markersize=3, label=label)
        axes[1].errorbar(batches, medians,
                        yerr=[[m-min(r) for m,r in zip(medians,rates)], [max(r)-m for m,r in zip(medians,rates)]],
                        fmt='-o', color=color, markersize=3, capsize=2)
    axes[0].set_ylabel('Peak allocated memory (GiB)')
    axes[1].set_ylabel('Capacity-probe throughput (targets/s)')
    for ax in axes:
        ax.set_xlabel('Physical sequences per GPU')
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=7)
    fig.suptitle('20-update capacity probes; concurrent workers; not isolated benchmarks', fontsize=7)
    return save_all(fig, 'batch_memory_and_throughput')


def plot_activation_storage(path: Path) -> list[Path]:
    records = read_json(path)
    fig, ax = figure(width=WIDE, ratio=0.48)
    for kind, color in [('midpoint', PALETTE['orange']), ('coupled_euler', PALETTE['blue'])]:
        for mode, style in [('stored', '-'), ('reconstructed', '--')]:
            rows = sorted([r for r in records if r['integrator']==kind and r['backward_mode']==mode], key=lambda r:r['depth'])
            ax.plot([r['depth'] for r in rows], [r['saved_unique_storage_bytes']/1024 for r in rows],
                    style, marker='o', color=color, label=f"{kind.replace('_',' ')} / {mode}")
    ax.set_xlabel('Stack depth (blocks)'); ax.set_ylabel('Unique saved storage (KiB)')
    ax.set_yscale('log'); ax.set_xticks([2,4,9,18])
    ax.legend(ncol=2, loc='upper left', fontsize=7)
    fig.suptitle('CPU FP32 stack diagnostic; parameter storage excluded; not GPU peak memory', fontsize=7)
    return save_all(fig, 'activation_storage_vs_depth')


def plot_seed_variation(runs: list[dict], campaign_label: str) -> list[Path]:
    selected = [run for run in runs if run["condition"] in "ABC" and
                run["final"].get("holdout_loss_nats") is not None]
    grouped = {key: sorted((run for run in selected if run["condition"] == key), key=lambda run: run["seed"])
               for key in "ABC"}
    grouped = {key: value for key, value in grouped.items() if value}
    if not grouped:
        return []
    fig, ax = figure(width=WIDE, ratio=0.48)
    all_values = []
    for index, (key, group) in enumerate(grouped.items()):
        values = np.asarray([run["final"]["holdout_loss_nats"] for run in group])
        all_values.append(values)
        if len(values) > 1:
            ax.vlines(index, values.min(), values.max(), color=CONDITION_COLORS[key], linewidth=2.0, alpha=0.45)
        offsets = np.linspace(-0.065, 0.065, len(group)) if len(group) > 1 else np.zeros(1)
        for offset, run, value in zip(offsets, group, values):
            ax.scatter(index + offset, value, color=CONDITION_COLORS[key], s=42,
                       marker=MARKERS[index], edgecolor="white", linewidth=0.6, zorder=3)
            ax.annotate(str(run["seed"]), (index + offset, value), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6.3)
        ax.hlines(np.median(values), index - 0.14, index + 0.14, color="#222222", linewidth=1.0, zorder=4)
        ax.text(index, max(values) + 0.035, f"n={len(values)}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(np.arange(len(grouped)), [f"{key}: batch {group[0]['final']['physical_batch']}" for key, group in grouped.items()])
    ax.set_ylabel("Holdout loss (nats/target)")
    ax.set_xlabel("Principal condition")
    low, high = common_y_limits(all_values)
    extra = max(0.10 * (high - low), 0.03)
    ax.set_ylim(low, high + extra)
    ax.text(0.99, 0.04, "points: seeds   vertical span: min–max   tick: median   no CI",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555555")
    fig.suptitle(campaign_label, fontsize=7, color="#555555")
    return save_all(fig, "seed_variation_holdout")


def plot_supplemental_sensitivity(runs: list[dict]) -> list[Path]:
    """Keep auto-policy seed 3141 and F/G checks out of the locked main estimates."""
    selected = [run for run in runs if run["campaign"] == "supplemental" and
                run["final"].get("holdout_loss_nats") is not None]
    if not selected:
        return []
    selected.sort(key=lambda run: (run["condition"], run["seed"], run["run_id"]))
    y = np.arange(len(selected))
    values = np.asarray([run["final"]["holdout_loss_nats"] for run in selected])
    fig, ax = figure(width=WIDE, ratio=max(0.40, 0.105 * len(selected) + 0.20))
    for index, (row, value) in enumerate(zip(selected, values)):
        key = row["condition"]
        ax.scatter(value, index, color=CONDITION_COLORS.get(key, BASELINE),
                   marker=MARKERS[index % len(MARKERS)], s=42, edgecolor="white",
                   linewidth=0.6, zorder=3)
        ax.annotate(f"{value:.3f}", (value, index), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=6.5)
    ax.set_yticks(y, [run["run_id"] for run in selected])
    ax.invert_yaxis()
    ax.set_xlabel("Holdout loss (nats/target)")
    ax.set_ylabel("Supplemental auto-policy run")
    ax.set_xlim(*common_y_limits([values]))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.04, "separate sensitivity evidence — not pooled with the main campaign",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555555")
    return save_all(fig, "supplemental_sensitivity_holdout")


def discover_probe_files() -> list[Path]:
    paths = [EXPERIMENTS / "depth-sweep" / "probes.json"]
    paths.extend(sorted(EXPERIMENTS.glob("kaggle-*/trained_checkpoint_probes.json")))
    return [path for path in paths if path.exists()]


def positive_for_log(value: float, floor: float = 1e-18) -> float:
    return max(float(value), floor)


def plot_numerical_error() -> list[Path]:
    depth_path = EXPERIMENTS / "depth-sweep" / "probes.json"
    if not depth_path.exists():
        return []
    depth = [row for row in read_json(depth_path) if row["dtype"] == "torch.float32"]
    probe_paths = sorted(EXPERIMENTS.glob("kaggle-*/trained_checkpoint_probes.json"))
    progress = []
    for path in probe_paths:
        progress.extend(row for row in read_json(path)
                        if row.get("status") in {"PASS", "FAIL"} and
                        row.get("committed_targets_at_probe") is not None and
                        math.isclose(float(row.get("step_size", 0.5)), 0.5))
    fig, axes = figure_grid(2, 2, width=WIDE, ratio=0.90)
    fig.suptitle('FP32 numerical probes', fontsize=8)
    integrators = sorted({row["integrator"] for row in depth})
    styles = {"coupled_euler": (PALETTE["blue"], "o", "coupled Euler"),
              "midpoint": (PALETTE["orange"], "s", "midpoint")}
    metrics = [
        ("relative_reconstruction_error", "Relative reconstruction error", 1e-5),
        ("relative_gradient_error", "Global relative gradient error", 1e-4),
    ]
    for ax, (metric, ylabel, threshold) in zip(axes[0], metrics):
        for name in integrators:
            points = sorted((row for row in depth if row["integrator"] == name), key=lambda row: row["depth"])
            color, marker, label = styles.get(name, (BASELINE, "D", name))
            ax.plot([row["depth"] for row in points], [positive_for_log(row[metric]) for row in points],
                    color=color, marker=marker, label=label)
        ax.axhline(threshold, color=BASELINE, linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_xlabel("Reversible stack depth (blocks)")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xticks(sorted({row["depth"] for row in depth}))
    axes[0, 0].legend(loc="lower right")

    for ax, (metric, ylabel, threshold) in zip(axes[1], metrics):
        for index, name in enumerate(sorted({row["integrator"] for row in progress})):
            selected = [row for row in progress if row["integrator"] == name]
            color, marker, label = styles.get(name, (BASELINE, MARKERS[index], name))
            x = np.asarray([row["committed_targets_at_probe"] / 1e6 for row in selected])
            y = np.asarray([positive_for_log(row["reconstruction"]["relative_reconstruction_error"]
                                             if metric == "relative_reconstruction_error"
                                             else row["global_relative_gradient_error_fp32"])
                            for row in selected])
            offsets = np.linspace(-0.28, 0.28, len(selected)) if len(selected) > 1 else np.zeros(1)
            ax.scatter(x, y, color=color, marker=marker, s=25, edgecolor="white",
                       linewidth=0.5, label=label)
        ax.axhline(threshold, color=BASELINE, linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_xlabel("Committed targets at probe (millions)")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
    axes[1, 0].legend(loc="lower right")
    axes[1, 1].text(0.98, 0.96, "cross-run checkpoints\n(not a longitudinal trace)",
                    transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=6.3,
                    color="#555555")
    panel_labels(axes)
    return save_all(fig, "numerical_error_depth_and_progress")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_provenance(figure_inputs: dict[str, set[Path]], figure_outputs: dict[str, list[Path]],
                     memory_metadata: dict, campaign: str, campaign_label: str) -> Path:
    inputs = sorted({path.resolve() for paths in figure_inputs.values() for path in paths if path.exists()})
    payload = {
        "schema_version": 1,
        "generator": "scripts/report_figures.py",
        "style": "scripts/orx_figstyle.py",
        "figure_width_inches": WIDE,
        "raster_dpi": 300,
        "uncertainty_policy": "Individual seeds and observed min-max ranges; no confidence interval from two seeds.",
        "main_campaign": campaign,
        "main_campaign_label": campaign_label,
        "campaign_selection_policy": (
            "Use kaggle-locked-v1 only after A/B/C seeds 1337 and 2027 plus D/E seed 1337 are all complete; "
            "otherwise use kaggle-full-v1. Supplemental auto-policy runs are never pooled into main estimates."
        ),
        "memory_throughput_source_mode": memory_metadata["mode"],
        "figures": {
            name: {
                "outputs": [relative(path) for path in outputs],
                "inputs": [relative(path) for path in sorted(figure_inputs.get(name, set())) if path.exists()],
            }
            for name, outputs in sorted(figure_outputs.items())
        },
        "inputs": [{"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                   for path in inputs],
        "gaps": [] if memory_metadata["mode"] == "isolated_benchmark" else [
            "experiments/kaggle-benchmark-v2/benchmark/summary.json was absent; memory_throughput uses explicitly labelled campaign measurements."
        ],
    }
    path = OUTPUT / "figure_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    use_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_runs = discover_runs()
    if not all_runs:
        raise RuntimeError("No complete full-budget runs with metrics and development evaluations were found")
    runs, campaign, campaign_label = select_campaign(all_runs)
    if runs:
        max_batch = next(run['final']['effective_batch'] for run in runs if run['condition']=='C')
        CONDITION_LABELS['C'] = f'C  coupled, reconstructed, batch {max_batch}'
        CONDITION_LABELS['E'] = f'E  coupled, reconstructed, effective batch {max_batch}'
    if not runs:
        raise RuntimeError("Neither the original nor locked principal campaign is complete enough to plot")
    batch = load_batch_evidence()

    outputs: dict[str, list[Path]] = {}
    inputs: dict[str, set[Path]] = defaultdict(set)
    training, development = plot_training_curves(runs, campaign_label)
    if training:
        outputs["training_loss_curves"] = training
        outputs["development_loss_curves"] = development
        selected_paths = {path for run in runs if run["condition"] in "ABC" for path in run["paths"]}
        inputs["training_loss_curves"] = selected_paths
        inputs["development_loss_curves"] = selected_paths

    causal = plot_causal_chain(runs, campaign_label)
    if causal:
        outputs["causal_loss_chain_seed1337"] = causal
        inputs["causal_loss_chain_seed1337"] = {
            path for run in runs if run["condition"] in "ADBEC" and run["seed"] == 1337
            for path in run["paths"]
        }

    capacity = plot_batch_capacity(batch)
    if capacity:
        outputs["batch_capacity_boundaries"] = capacity
        inputs["batch_capacity_boundaries"] = {
            path for item in batch for path in (item["summary_path"], item["trials_path"])
        }
        outputs['batch_memory_and_throughput'] = plot_batch_curves(batch)
        inputs['batch_memory_and_throughput'] = {item['trials_path'] for item in batch}
    storage_path = EXPERIMENTS/'storage-depth-v1/measurements.json'
    if storage_path.exists():
        outputs['activation_storage_vs_depth'] = plot_activation_storage(storage_path)
        inputs['activation_storage_vs_depth'] = {storage_path}

    tradeoff, memory_metadata = plot_memory_throughput(runs, campaign_label)
    outputs["memory_throughput"] = tradeoff
    if memory_metadata["mode"] == "isolated_benchmark":
        inputs["memory_throughput"] = {BENCHMARK}
        outputs['benchmark_device_variation'] = plot_benchmark_devices()
        inputs['benchmark_device_variation'] = set(BENCHMARK.parent.glob('*-rep[012].json'))
    else:
        inputs["memory_throughput"] = {path for run in runs for path in run["paths"]}

    variation = plot_seed_variation(runs, campaign_label)
    if variation:
        outputs["seed_variation_holdout"] = variation
        inputs["seed_variation_holdout"] = {
            path for run in runs if run["condition"] in "ABC" for path in run["paths"]
        }

    supplemental = plot_supplemental_sensitivity(all_runs)
    if supplemental:
        outputs["supplemental_sensitivity_holdout"] = supplemental
        inputs["supplemental_sensitivity_holdout"] = {
            path for run in all_runs if run["campaign"] == "supplemental" for path in run["paths"]
        }

    numerical = plot_numerical_error()
    if numerical:
        outputs["numerical_error_depth_and_progress"] = numerical
        inputs["numerical_error_depth_and_progress"] = set(discover_probe_files())

    provenance = write_provenance(inputs, outputs, memory_metadata, campaign, campaign_label)
    print(f"generated {len(outputs)} figures ({sum(len(paths) for paths in outputs.values())} files)")
    print(f"provenance: {relative(provenance)}")


if __name__ == "__main__":
    main()
