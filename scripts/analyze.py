"""Regenerate evidence summaries and quantitative plots solely from raw run files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from revllm.data import digest_file


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect() -> tuple[list[dict], dict[str, list[dict]]]:
    summary, traces = [], {}
    for final_path in sorted(EXPERIMENTS.rglob("final.json")):
        run_dir = final_path.parent
        final = json.loads(final_path.read_text(encoding="utf-8"))
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            raise RuntimeError(f"Missing frozen config for {run_dir}")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        committed = [r for r in metrics if r.get("status") == "COMMITTED"]
        if not committed or committed[-1]["committed_targets"] != final["committed_targets"]:
            raise RuntimeError(f"Raw committed-token log disagrees with final summary for {run_dir}")
        if sum(r["valid_targets"] for r in committed) != final["committed_targets"]:
            raise RuntimeError(f"Raw target counts do not add up for {run_dir}")
        previous = 0
        for index, record in enumerate(committed, 1):
            if record["update"] != index or record["committed_targets"] != previous + record["valid_targets"]:
                raise RuntimeError(f"Non-monotonic update or token cursor for {run_dir}")
            previous = record["committed_targets"]
        if final["status"] == "COMPLETE" and final["committed_targets"] != cfg["target_tokens"]:
            raise RuntimeError(f"Completed run is short of its frozen token budget: {run_dir}")
        if final["attempted_targets"] < final["committed_targets"]:
            raise RuntimeError(f"Attempted target counter is smaller than committed: {run_dir}")
        checkpoint = run_dir / "latest.pt"
        if checkpoint.exists() and digest_file(checkpoint) != final["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for {run_dir}")
        row = {
            "run_id": final["run_id"], "status": final["status"],
            "condition": final["run_id"].split("-")[0],
            "campaign": ("reproduction" if "reproduction" in final_path.parts else
                         "locked" if cfg.get('attention_backend') == 'efficient' else
                         "supplemental" if "kaggle-supplemental" in str(final_path) else
                         "pilot" if final["run_id"].startswith("pilot-") else "original"),
            "attention_backend": cfg.get("attention_backend", "auto"),
            "seed": cfg["seed"], "integrator": cfg["model"]["integrator"],
            "backward_mode": cfg["model"]["backward_mode"],
            "step_size": cfg["model"].get("step_size", 0.25),
            "blend": cfg["model"].get("blend", 1.0),
            "committed_targets": final["committed_targets"],
            "attempted_targets": final["attempted_targets"],
            "physical_batch": final["physical_batch"],
            "effective_batch": final["effective_batch"],
            "parameter_count": final["parameter_count"],
            "updates": final.get("updates"), "retries": final.get("retries"),
            "training_seconds": final.get("training_seconds"),
            "job_seconds": final.get("job_seconds"),
            "final_update_loss_nats": final.get("final_update_loss_nats"),
            "last_1m_training_loss_nats": final.get("last_1m_training_loss_nats"),
            "fixed_training_subset_loss_nats": final.get("fixed_training_subset_loss_nats"),
            "holdout_perplexity": final.get("holdout_perplexity"),
            "separator_targets": sum(r.get("separator_targets", 0) for r in committed),
            "padding_slots": sum(r.get("padding_slots", 0) for r in committed),
            "checkpoint_sha256": final.get("checkpoint_sha256"),
            "holdout_loss_nats": final.get("holdout_loss_nats"),
            "dev_loss_nats": read_jsonl(run_dir / "eval.jsonl")[-1]["dev_loss_nats"] if read_jsonl(run_dir / "eval.jsonl") else None,
            "training_tokens_per_second": final["training_tokens_per_second"],
            "peak_allocated_bytes": final.get("peak_allocated_bytes"),
            "peak_reserved_bytes": final.get("peak_reserved_bytes"),
            "checkpoint_locally_verified": checkpoint.exists(),
            "source": str(final_path.relative_to(ROOT)),
        }
        summary.append(row)
        trace_name = final["run_id"] if row['campaign'] != 'reproduction' else f"reproduction/{final['run_id']}"
        if trace_name in traces:
            raise RuntimeError(f'Duplicate run identity in analysis: {trace_name}')
        traces[trace_name] = committed
    return summary, traces


def plot_line(traces: dict[str, list[dict]], x_key: str, output: Path, xlabel: str) -> None:
    if not traces:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in traces.items():
        ax.plot([r[x_key] for r in rows], [r["loss_nats"] for r in rows],
                label=name, linewidth=1, alpha=0.8)
    ax.set(xlabel=xlabel, ylabel="Training loss (nats/target)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_tradeoff(rows: list[dict], output: Path) -> None:
    points = [r for r in rows if r["peak_allocated_bytes"] is not None]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in points:
        x = row["peak_allocated_bytes"] / 2**30
        y = row["training_tokens_per_second"]
        ax.scatter(x, y)
        ax.annotate(row["run_id"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Peak training allocation (GiB)", ylabel="Campaign training targets/s")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def batch_evidence() -> list[dict]:
    evidence = []
    for summary_path in sorted(EXPERIMENTS.rglob("batch-search*/search/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trials = read_jsonl(summary_path.parent / "trials.jsonl")
        campaign = summary_path.relative_to(EXPERIMENTS).parts[0]
        name = campaign + ': ' + summary_path.parent.parent.name.removeprefix("batch-search-")
        evidence.append({"name": name, "summary": summary, "trials": trials,
                         "source": str(summary_path.relative_to(ROOT))})
    return evidence


def plot_batch(evidence: list[dict], key: str, output: Path, ylabel: str) -> None:
    if not evidence:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for item in evidence:
        points = [r for r in item["trials"] if r.get("status") == "PASS" and "measurement" in r]
        points.sort(key=lambda r: r["batch"])
        if not points:
            continue
        y = [r["measurement"][key] / (2**30 if "bytes" in key else 1) for r in points]
        ax.plot([r["batch"] for r in points], y, marker=".", label=item["name"])
    ax.set(xlabel="Physical sequences per GPU", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_boundaries(evidence: list[dict], output: Path) -> None:
    valid = [x for x in evidence if x["summary"].get("status") == "VALIDATED"]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [x["name"] for x in valid]
    stable = [x["summary"]["largest_stable_batch"] for x in valid]
    failed = [x["summary"].get("smallest_observed_oom_batch") for x in valid]
    ax.barh(names, stable, label="Largest validated stable batch", color="#4477aa")
    for index, value in enumerate(failed):
        if value is not None:
            ax.scatter(value, index, marker="x", color="#cc3311", zorder=3)
            ax.annotate(f" OOM {value}", (value, index), va="center", fontsize=8)
    ax.set(xlabel="Physical sequences per GPU")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_depth_probes(path: Path, metric: str, output: Path, ylabel: str) -> None:
    if not path.exists():
        return
    records = json.loads(path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = sorted({(r["integrator"], r["dtype"]) for r in records})
    for kind, dtype in groups:
        points = sorted((r for r in records if (r["integrator"], r["dtype"]) == (kind, dtype)),
                        key=lambda r: r["depth"])
        ax.plot([p["depth"] for p in points], [max(p[metric], 1e-17) for p in points],
                marker="o", label=f"{kind}, {dtype.removeprefix('torch.')}")
    ax.set(xlabel="Reversible stack depth (blocks)", ylabel=ylabel, yscale="log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_seed_variation(rows: list[dict], output: Path) -> None:
    full = [r for r in rows if r["committed_targets"] == 50_000_000 and
            r["holdout_loss_nats"] is not None]
    if len({r["seed"] for r in full}) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for letter in "ABC":
        points = [r for r in full if r["run_id"].startswith(letter + "-")]
        if points:
            ax.scatter([letter] * len(points), [r["holdout_loss_nats"] for r in points], label=letter)
            for point in points:
                ax.annotate(str(point["seed"]), (letter, point["holdout_loss_nats"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Principal condition", ylabel="Holdout loss (nats/target)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_numerical_progress(output: Path) -> None:
    records = []
    for path in sorted(EXPERIMENTS.rglob("trained_checkpoint_probes.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    points = [row for row in records if row.get("status") in {"PASS", "FAIL"} and
              "committed_targets_at_probe" in row]
    if len({row["committed_targets_at_probe"] for row in points}) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for kind in sorted({row["integrator"] for row in points}):
        selected = [row for row in points if row["integrator"] == kind]
        xs = [row["committed_targets_at_probe"] / 1_000_000 for row in selected]
        axes[0].scatter(xs, [max(row["global_relative_gradient_error_fp32"], 1e-17)
                             for row in selected], label=kind, alpha=0.8)
        axes[1].scatter(xs, [max(row["reconstruction"]["relative_reconstruction_error"], 1e-17)
                             for row in selected], label=kind, alpha=0.8)
    for ax, title in zip(axes, ("Global relative gradient error", "Relative reconstruction error")):
        ax.set(xlabel="Committed training targets (millions)", ylabel=title, yscale="log")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_causal_comparisons(rows: list[dict], output: Path) -> None:
    """Keep each contrast explicit; only compare completed equal-token runs."""
    by_id = {row["run_id"]: row for row in rows if row["status"] == "COMPLETE" and
             row["committed_targets"] == 50_000_000 and row['campaign'] != 'reproduction'}
    contrasts = [("A", "D", "architecture"), ("D", "B", "reconstruction"),
                 ("B", "E", "effective_batch"), ("E", "C", "physical_batch"),
                 ("A", "C", "combined")]
    fields = ["seed", "contrast", "from_run", "to_run", "from_holdout_loss_nats",
              "to_holdout_loss_nats", "delta_holdout_loss_nats",
              "from_training_targets_per_second", "to_training_targets_per_second",
              "throughput_ratio", "from_peak_allocated_bytes", "to_peak_allocated_bytes",
              "memory_ratio"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        def emit(a: dict | None, b: dict | None, label: str) -> None:
            if a is None or b is None or a["holdout_loss_nats"] is None or b["holdout_loss_nats"] is None:
                return
            writer.writerow({
                "seed": a["seed"], "contrast": label, "from_run": a["run_id"], "to_run": b["run_id"],
                "from_holdout_loss_nats": a["holdout_loss_nats"],
                "to_holdout_loss_nats": b["holdout_loss_nats"],
                "delta_holdout_loss_nats": b["holdout_loss_nats"] - a["holdout_loss_nats"],
                "from_training_targets_per_second": a["training_tokens_per_second"],
                "to_training_targets_per_second": b["training_tokens_per_second"],
                "throughput_ratio": b["training_tokens_per_second"] / a["training_tokens_per_second"],
                "from_peak_allocated_bytes": a["peak_allocated_bytes"],
                "to_peak_allocated_bytes": b["peak_allocated_bytes"],
                "memory_ratio": b["peak_allocated_bytes"] / a["peak_allocated_bytes"],
            })

        for campaign_name in ('original', 'locked'):
            campaign_rows = {(row['condition'], row['seed']): row for row in by_id.values()
                             if row['campaign'] == campaign_name}
            for seed in sorted({key[1] for key in campaign_rows}):
                for left, right, label in contrasts:
                    a, b = campaign_rows.get((left,seed)), campaign_rows.get((right,seed))
                    emit(a, b, label + '_' + campaign_name)
        emit(by_id.get("B-1337"), by_id.get("F-midpoint-1337"), "reversible_formulation")
        emit(by_id.get("A-1337"), by_id.get("G-checkpointed-1337"), "activation_checkpointing")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for stale in RESULTS.glob("*.png"):
        stale.unlink()
    rows, traces = collect()
    fields = list(rows[0]) if rows else ["run_id", "status"]
    with (RESULTS / "runs.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_causal_comparisons(rows, RESULTS / "causal_comparisons.csv")
    expected_locked = set(json.loads((ROOT/'configs/locked_campaign.json').read_text())['run_ids'])
    selected_campaign = 'locked' if expected_locked <= set(traces) else 'original'
    main_rows = [row for row in rows if row['campaign']==selected_campaign and row['seed'] in (1337,2027)]
    principal = {row['run_id']: traces[row['run_id']] for row in main_rows}
    pilots = {name: trace for name, trace in traces.items() if name.startswith("pilot-")}
    plot_line(principal, "committed_targets", RESULTS / "loss_vs_tokens.png", "Committed training targets")
    plot_line(principal, "training_seconds", RESULTS / "loss_vs_time.png", "Training seconds")
    plot_line(pilots, "committed_targets", RESULTS / "pilot_loss_vs_tokens.png", "Committed pilot targets")
    plot_tradeoff(main_rows, RESULTS / "memory_throughput.png")
    batches = batch_evidence()
    plot_batch(batches, "peak_allocated_bytes", RESULTS / "memory_vs_batch.png", "Peak allocated memory (GiB)")
    plot_batch(batches, "tokens_per_second", RESULTS / "throughput_vs_batch.png", "Probe targets/s")
    plot_boundaries(batches, RESULTS / "max_batch_boundaries.png")
    depths = EXPERIMENTS / "depth-sweep" / "probes.json"
    plot_depth_probes(depths, "relative_reconstruction_error", RESULTS / "reconstruction_error_vs_depth.png",
                      "Relative reconstruction error")
    plot_depth_probes(depths, "relative_gradient_error", RESULTS / "gradient_error_vs_depth.png",
                      "Global relative gradient error")
    plot_seed_variation(main_rows, RESULTS / "seed_variation.png")
    plot_numerical_progress(RESULTS / "numerical_error_vs_training_progress.png")
    print(json.dumps({"runs_verified": len(rows), "output": str(RESULTS / "runs.csv")}, indent=2))


if __name__ == "__main__":
    main()
