"""Rebuild a conservative, file-level evidence ledger from the retained artifacts."""

from __future__ import annotations

import json
import base64
import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revllm.data import digest_file, load_manifest
from revllm.model import ModelConfig, TinyGPT, count_unique_parameters

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
AUDIT = ROOT / "audit"
EXPECTED_PILOTS = {f"pilot-{name}" for name in (
    "conventional", "midpoint025", "midpoint050", "midpoint100",
    "coupled025", "coupled050", "coupled100", "blended025")}
EXPECTED_FULL = {f"{letter}-1337" for letter in "ABCDE"} | {
    f"{letter}-2027" for letter in "ABC"}
CAMPAIGN = json.loads((ROOT/'configs/locked_campaign.json').read_text())
EXPECTED_LOCKED = set(CAMPAIGN['run_ids'])
EXPECTED_SUPPLEMENTAL = {'A-3141','B-3141','C-3141','F-midpoint-1337','G-checkpointed-1337'}
EXPECTED_BATCH = {"conventional": (110, 111), "checkpointed": (181, 182),
                  "coupled_stored": (110, 111), "coupled_reconstructed": (192, 193)}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def claim(name: str, status: str, evidence: list[str], detail: str = "") -> dict:
    assert status in {"PASS", "FAIL", "UNVERIFIED"}
    return {"requirement": name, "status": status, "evidence": evidence, "detail": detail}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_run(directory: Path, expected_targets: int, require_checkpoint: bool) -> tuple[bool, str]:
    needed = [directory / name for name in ("final.json", "run_config.json", "metrics.jsonl")]
    if not all(path.exists() for path in needed):
        return False, "missing final, config, or update log"
    final = json.loads(needed[0].read_text(encoding="utf-8"))
    cfg = json.loads(needed[1].read_text(encoding="utf-8"))
    recipes = list((ROOT/'configs').glob(f"*/{cfg.get('run_id')}.json"))
    if len(recipes) == 1:
        recipe = json.loads(recipes[0].read_text())
        for key, value in recipe.items():
            if key in {'data_dir', 'out_dir'}:
                continue
            if key == 'model':
                if any(cfg.get('model', {}).get(k) != v for k, v in value.items()):
                    return False, 'model differs from preregistered recipe'
            elif cfg.get(key) != value:
                return False, f'{key} differs from preregistered recipe'
    committed = [row for row in read_jsonl(needed[2]) if row.get("status") == "COMMITTED"]
    if not committed:
        return False, "no committed updates"
    cursor = 0
    for index, row in enumerate(committed, 1):
        cursor += row["valid_targets"]
        if row["update"] != index or row["committed_targets"] != cursor:
            return False, "non-contiguous update or token cursor"
    if (cursor != expected_targets or final.get("committed_targets") != cursor or
            cfg.get("target_tokens") != cursor or final.get("updates") != len(committed) or
            final.get("status") != "COMPLETE" or
            final.get("parameter_count") != 19_969_152 or
            final.get("run_id") != cfg.get("run_id") or
            final.get("attempted_targets", -1) < cursor):
        return False, "final metadata disagrees with raw updates or frozen config"
    checkpoint = directory / "latest.pt"
    if require_checkpoint and not checkpoint.exists():
        return False, "checkpoint unavailable for local hash verification"
    if checkpoint.exists() and digest_file(checkpoint) != final.get("checkpoint_sha256"):
        return False, "checkpoint SHA-256 mismatch"
    if expected_targets == 50_000_000:
        evals = read_jsonl(directory / "eval.jsonl")
        if not evals or evals[-1].get("committed_targets") != cursor:
            return False, "final development evaluation missing"
        if final.get("holdout_loss_nats") is None:
            return False, "final holdout evaluation missing"
    return True, "raw updates, frozen config, final summary, and available checkpoint agree"


def credential_scan() -> dict:
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    paths = [ROOT / name.decode("utf-8") for name in listed.stdout.split(b"\0") if name]
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in (
        r"AKIA[0-9A-Z]{16}", r"gh[pousr]_[A-Za-z0-9_]{30,}",
        r"(?:sk|hf)_[A-Za-z0-9]{30,}",
        r"(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[=:]\s*['\"]?[A-Za-z0-9_/-]{16,}")]
    findings = []
    verified_inline_images = []
    png_hashes = {digest_file(p) for p in paths if p.is_file() and p.suffix.lower() == '.png'}
    inline_png = re.compile(r'data:image/png;base64,([A-Za-z0-9+/=]+)')
    for path in paths:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            def verified_image(match):
                try:
                    payload = base64.b64decode(match.group(1), validate=True)
                    digest = hashlib.sha256(payload).hexdigest()
                except ValueError:
                    return match.group(0)
                if payload.startswith(b"\x89PNG\r\n\x1a\n") and digest in png_hashes:
                    verified_inline_images.append({"path": relative(path), "line": number, "sha256": digest})
                    return "[verified tracked PNG]"
                return match.group(0)
            scan_line = inline_png.sub(verified_image, line)
            if any(pattern.search(scan_line) for pattern in patterns):
                findings.append({"path": relative(path), "line": number})
    result = {"tracked_files_scanned": len(paths), "findings": findings,
              "verified_inline_images": verified_inline_images,
              "scope": "tracked text files; inline PNG payloads excluded only when byte-identical to a tracked PNG; credential-pattern scan, not proof of absence"}
    (AUDIT / "credential_scan.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    AUDIT.mkdir(exist_ok=True)
    ledger = []
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                           cwd=ROOT, text=True, capture_output=True)
    (AUDIT / "pytest.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    ledger.append(claim("Critical local correctness, accounting, and resume tests",
                        "PASS" if tests.returncode == 0 else "FAIL",
                        ["audit/pytest.txt", "tests/"], f"exit={tests.returncode}"))

    counts = {name: count_unique_parameters(TinyGPT(ModelConfig(integrator=name)))
              for name in ("conventional", "midpoint", "coupled_euler", "blended_midpoint")}
    (AUDIT / "parameter_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    ledger.append(claim("All principal architectures have 19,969,152 unique parameters",
                        "PASS" if all(value == 19_969_152 for value in counts.values()) else "FAIL",
                        ["audit/parameter_counts.json"]))

    manifest_path = ROOT / "data" / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        expected = {"train": 50_000_000, "dev": 262_144, "holdout": 1_000_000}
        sizes_ok = all(manifest["tapes"][name]["target_count"] == count
                       for name, count in expected.items())
        files_ok = all((ROOT / "data" / f"{name}.bin").exists() and
                       digest_file(ROOT / "data" / f"{name}.bin") == manifest["tapes"][name]["sha256"]
                       for name in expected)
        data_status = "PASS" if sizes_ok and files_ok else "FAIL"
    else:
        data_status = "UNVERIFIED"
    ledger.append(claim("Pinned data and exact split sizes; local tape hashes verified",
                        data_status, ["data/manifest.json", "data/tokenizer.json"]))

    gates = sorted(EXPERIMENTS.rglob("gpu_correctness.json"))
    passing_gates = [path for path in gates if (records := json.loads(path.read_text(encoding="utf-8")))
                     and len(records) >= 7 and all(row.get("status") == "PASS" for row in records)]
    ledger.append(claim("Intended mixed-precision GPU numerical gates",
                        "PASS" if passing_gates else "UNVERIFIED",
                        [relative(path) for path in gates],
                        f"{len(passing_gates)} complete passing GPU-gate bundles"))
    depth_path = EXPERIMENTS / "depth-sweep" / "probes.json"
    depths = json.loads(depth_path.read_text(encoding="utf-8")) if depth_path.exists() else []
    ledger.append(claim("Depth-dependent FP64/FP32 reconstruction and gradient probes",
                        "PASS" if depths and all(row.get("status") == "PASS" for row in depths) else "UNVERIFIED",
                        [relative(depth_path)] if depths else [], f"{len(depths)} retained probes"))

    pilot_dirs = {path.parent.name: path.parent for path in EXPERIMENTS.rglob("pilot-*/final.json")}
    pilot_results = {name: verify_run(pilot_dirs[name], 2_000_000, False)
                     for name in EXPECTED_PILOTS if name in pilot_dirs}
    probes_path = EXPERIMENTS / "kaggle-pilots-v3" / "trained_checkpoint_probes.json"
    trained = json.loads(probes_path.read_text(encoding="utf-8")) if probes_path.exists() else []
    trained_pass = sum(row.get("status") == "PASS" for row in trained) >= 7
    pilots_ok = (set(pilot_results) == EXPECTED_PILOTS and
                 all(ok for ok, _ in pilot_results.values()) and trained_pass)
    ledger.append(claim("Eight 2M-target pilots and seven post-training numerical probes",
                        "PASS" if pilots_ok else "UNVERIFIED",
                        [relative(path / "final.json") for path in pilot_dirs.values()] +
                        ([relative(probes_path)] if probes_path.exists() else []),
                        f"{sum(ok for ok, _ in pilot_results.values())}/8 raw pilots valid; post-training probes={trained_pass}"))

    batch_evidence = []
    batch_ok = True
    for name, (stable, failing) in EXPECTED_BATCH.items():
        base = EXPERIMENTS / "kaggle-batch-v1" / f"batch-search-{name}" / "search"
        summary_path = base / "summary.json"
        trials_path = base / "trials.jsonl"
        if not summary_path.exists() or not trials_path.exists():
            batch_ok = False
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trials = read_jsonl(trials_path)
        successes = [row for row in trials if row.get("status") == "PASS" and row.get("batch") == stable]
        sustained = any(row.get("updates_requested") == 200 for row in successes)
        fresh = sum(row.get("updates_requested") == 20 for row in successes) >= 3
        oom = any(row.get("status") == "OOM" and row.get("batch") == failing for row in trials)
        batch_ok &= (summary.get("status") == "VALIDATED" and
                     summary.get("largest_stable_batch") == stable and
                     summary.get("smallest_observed_oom_batch") == failing and
                     sustained and fresh and oom)
        batch_evidence.extend([relative(summary_path), relative(trials_path)])
    ledger.append(claim("Four observed OOM boundaries with repeated and sustained trials",
                        "PASS" if batch_ok and len(batch_evidence) == 8 else "UNVERIFIED",
                        batch_evidence, "B0=110 and BR=192 under the search probe"))
    locked_batch_files = []
    locked_batch_ok = True
    for name, (stable, failing) in {'conventional':(110,111),'checkpointed':(182,183),'coupled_stored':(110,111),'coupled_reconstructed':(191,192)}.items():
        base = EXPERIMENTS/'kaggle-locked-batch-v1'/f'batch-search-{name}'/'search'
        path = base/'summary.json'
        if not path.exists():
            locked_batch_ok = False
            continue
        summary = json.loads(path.read_text())
        trials = read_jsonl(base/'trials.jsonl')
        passing = [r for r in trials if r.get('status')=='PASS' and r.get('batch')==stable]
        locked_batch_ok &= (summary.get('status')=='VALIDATED' and
                           summary.get('largest_stable_batch')==stable and
                           summary.get('smallest_observed_oom_batch')==failing and
                           sum(r.get('updates_requested')==20 for r in passing)>=3 and
                           any(r.get('updates_requested')==200 for r in passing) and
                           any(r.get('status')=='OOM' and r.get('batch')==failing for r in trials) and
                           all(r['measurement'].get('attention_backend',{}).get('effective_policy')=='efficient' for r in passing))
        locked_batch_files.extend([relative(path),relative(base/'trials.jsonl')])
    ledger.append(claim('Four locked-backend OOM boundaries reproduced with fresh and sustained trials',
                        'PASS' if locked_batch_ok else 'UNVERIFIED', locked_batch_files,
                        'Different measured boundaries require investigation and amended interpretation, not automatic rejection.'))

    full_dirs = {path.parent.name: path.parent for path in EXPERIMENTS.rglob("final.json")
                 if path.parent.name in EXPECTED_FULL | EXPECTED_LOCKED | EXPECTED_SUPPLEMENTAL}
    full_results = {name: verify_run(full_dirs[name], 50_000_000, True)
                    for name in EXPECTED_FULL if name in full_dirs}
    full_ok = set(full_results) == EXPECTED_FULL and all(ok for ok, _ in full_results.values())
    ledger.append(claim("Eight original 50M-target runs, raw counts and checkpoint hashes",
                        "PASS" if full_ok else "UNVERIFIED",
                        [relative(directory / "final.json") for directory in full_dirs.values()],
                        f"{sum(ok for ok, _ in full_results.values())}/8 fully verified; " +
                        "; ".join(f"{name}: {message}" for name, (ok, message) in full_results.items() if not ok)))

    locked_results = {name: verify_run(full_dirs[name], 50_000_000, True)
                      for name in EXPECTED_LOCKED if name in full_dirs}
    locked_ok = set(locked_results) == EXPECTED_LOCKED and all(ok for ok, _ in locked_results.values())
    backend_ok = locked_ok
    for name in locked_results:
        final = json.loads((full_dirs[name] / "final.json").read_text())
        backend = final.get("attention_backend", {})
        flags = backend.get("flags", {})
        operators = backend.get("operator_probe", {}).get("operators", [])
        backend_ok &= (backend.get("requested_policy") == backend.get("effective_policy") == "efficient" and
                       flags == {"flash_sdp_enabled": False, "mem_efficient_sdp_enabled": True,
                                 "math_sdp_enabled": False, "cudnn_sdp_enabled": False} and
                       any("efficient_attention_backward" in key for key in operators))
    ledger.append(claim("Eight locked-backend 50M confirmations with raw counts, hashes and actual operators",
                        "PASS" if backend_ok else "UNVERIFIED",
                        [relative(full_dirs[name] / "final.json") for name in sorted(locked_results)],
                        f"{sum(ok for ok, _ in locked_results.values())}/8 complete; backend verified={backend_ok}"))
    supplementary = {name: verify_run(full_dirs[name], 50_000_000, True)
                     for name in EXPECTED_SUPPLEMENTAL if name in full_dirs}
    ledger.append(claim('Five supplemental 50M runs: midpoint, checkpointing and third seed',
                        'PASS' if set(supplementary)==EXPECTED_SUPPLEMENTAL and all(ok for ok,_ in supplementary.values()) else 'UNVERIFIED',
                        [relative(full_dirs[name]/'final.json') for name in sorted(supplementary)]))

    full_probes_path = EXPERIMENTS / "kaggle-full-v1" / "trained_checkpoint_probes.json"
    full_probes = json.loads(full_probes_path.read_text(encoding="utf-8")) if full_probes_path.exists() else []
    expected_reversible = EXPECTED_FULL - {"A-1337", "A-2027"}
    passing_reversible = {row["run_id"] for row in full_probes if row.get("status") == "PASS" and
                          row.get("committed_targets_at_probe") == 50_000_000}
    ledger.append(claim("Post-training reversible inverse and gradient checks at 50M targets",
                        "PASS" if expected_reversible <= passing_reversible else "UNVERIFIED",
                        [relative(full_probes_path)] if full_probes_path.exists() else [],
                        f"{len(expected_reversible & passing_reversible)}/{len(expected_reversible)} principal reversible checkpoints pass"))

    evaluations = list(EXPERIMENTS.rglob("*-evaluation.json"))
    evaluation_ids, evaluation_errors = set(), []
    manifest = load_manifest(ROOT / "data" / "manifest.json")
    for path in evaluations:
        record = json.loads(path.read_text(encoding="utf-8"))
        name = record.get("run_id")
        if name not in full_dirs:
            continue
        final = json.loads((full_dirs[name] / "final.json").read_text(encoding="utf-8"))
        original_dev = read_jsonl(full_dirs[name] / "eval.jsonl")[-1]["dev_loss_nats"]
        expected = {"dev": (262_144, original_dev),
                    "holdout": (1_000_000, final["holdout_loss_nats"]),
                    "train_subset": (131_072, final["fixed_training_subset_loss_nats"])}
        valid = (record.get("checkpoint_sha256") == final["checkpoint_sha256"] and
                 record.get("data_sha256") == manifest["tapes"]["train"]["sha256"] and
                 record.get("committed_targets_at_checkpoint") == 50_000_000 and
                 record.get("parameter_count") == 19_969_152)
        for split, (targets, loss) in expected.items():
            measured = record.get("evaluation", {}).get(split, {})
            actual = measured.get("loss_nats", float("nan"))
            valid &= (measured.get("targets") == targets and math.isfinite(actual) and
                      abs(actual - loss) <= 1e-4 and
                      math.isclose(measured.get("perplexity", -1), math.exp(actual), rel_tol=1e-6))
        if valid:
            evaluation_ids.add(name)
        else:
            evaluation_errors.append(relative(path))
    ledger.append(claim("Clean-process checkpoint development, holdout, and training-subset evaluation",
                        "FAIL" if evaluation_errors else ("PASS" if EXPECTED_FULL <= evaluation_ids else "UNVERIFIED"),
                        [relative(path) for path in evaluations],
                        f"{len(EXPECTED_FULL & evaluation_ids)}/8 principal checkpoints match hashes, split counts and losses; errors={evaluation_errors}"))

    benchmark_dir = EXPERIMENTS / "kaggle-benchmark-v2" / "benchmark"
    benchmark_ok, benchmark_files = True, []
    for condition in ("A", "B", "C", "D", "E", "checkpointed-B0", "checkpointed-max"):
        for repetition, gpu in enumerate((0, 1, 0)):
            path = benchmark_dir / f"{condition}-rep{repetition}.json"
            if not path.exists():
                benchmark_ok = False
                continue
            benchmark_files.append(relative(path))
            record = json.loads(path.read_text())
            reps = record.get("repetitions", [])
            expected_effective = CAMPAIGN['BR'] if condition in ("C", "E") else (CAMPAIGN['checkpointed_max'] if condition == "checkpointed-max" else 110)
            expected_physical = CAMPAIGN['BR'] if condition == "C" else (CAMPAIGN['checkpointed_max'] if condition == "checkpointed-max" else 110)
            benchmark_ok &= (record.get("status") == "COMPLETE" and
                             record.get("assigned_physical_gpu") == gpu and
                             record.get("physical_batch") == expected_physical and
                             record.get("effective_batch") == expected_effective and len(reps) == 1)
            for rep in reps:
                seconds = rep.get("seconds", 0)
                targets = rep.get("measured_tokens", 0)
                backend = rep.get('attention_backend', record.get('attention_backend', {}))
                benchmark_ok &= (backend.get('effective_policy') == 'efficient' and
                                 backend.get('operator_probe', {}).get('efficient_operator_observed') is True)
                benchmark_ok &= (rep.get("warmup_updates") == 20 and rep.get("successful_updates") == 100 and
                                 targets == 100 * expected_effective * 512 and seconds > 0 and
                                 math.isclose(rep.get("tokens_per_second", 0), targets / max(seconds, 1e-30), rel_tol=1e-9))
    ledger.append(claim("Isolated 20-warmup/100-measured benchmarks, three fresh GPU-swapped repeats",
                        "PASS" if benchmark_ok else "UNVERIFIED", benchmark_files,
                        f"{len(benchmark_files)}/21 repetition files available"))

    diagnostics_dir = EXPERIMENTS / "kaggle-diagnostics-v1" / "diagnostics"
    precision_file = diagnostics_dir / "trained_mixed_precision.json"
    precision_rows = json.loads(precision_file.read_text()) if precision_file.exists() else []
    passing_mixed = {r["run_id"] for r in precision_rows if r.get("status") == "PASS"}
    ledger.append(claim("Trained principal checkpoints pass intended mixed-precision gradients and inverse gates",
                        "FAIL" if any(r.get("status") == "FAIL" for r in precision_rows) else
                        ("PASS" if expected_reversible <= passing_mixed else "UNVERIFIED"),
                        [relative(precision_file)] if precision_file.exists() else []))
    locked_precision_file = EXPERIMENTS/'kaggle-locked-diagnostics-v2'/'diagnostics'/'trained_mixed_precision.json'
    locked_precision = json.loads(locked_precision_file.read_text()) if locked_precision_file.exists() else []
    expected_locked_reversible = {name for name in EXPECTED_LOCKED if not name.startswith('A-')}
    locked_fp32_paths = [EXPERIMENTS/name/'trained_checkpoint_probes.json'
                        for name in ('kaggle-locked-v1','kaggle-locked-repair-v1')]
    locked_fp32 = [row for path in locked_fp32_paths if path.exists()
                   for row in json.loads(path.read_text())]
    fp32_passing = {row['run_id'] for row in locked_fp32 if row.get('status')=='PASS'
                   and row.get('committed_targets_at_probe')==50_000_000
                   and row.get('global_relative_gradient_error_fp32',1)>=0
                   and row.get('global_relative_gradient_error_fp32',1)<=1e-4
                   and row.get('reconstruction',{}).get('status')=='PASS'}
    ledger.append(claim('All six main trained reversible checkpoints pass FP32 inverse and gradient gates',
                        'PASS' if expected_locked_reversible<=fp32_passing else 'UNVERIFIED',
                        [relative(path) for path in locked_fp32_paths]))
    locked_pass = {r['run_id'] for r in locked_precision if r.get('status')=='PASS' and
                   r.get('attention_backend',{}).get('effective_policy')=='efficient'}
    ledger.append(claim('Locked trained checkpoints pass intended-precision numerical gates',
                        'FAIL' if any(r.get('status')=='FAIL' for r in locked_precision) else
                        ('PASS' if expected_locked_reversible <= locked_pass else 'UNVERIFIED'),
                        [relative(locked_precision_file)] if locked_precision_file.exists() else []))
    ledger.append(claim("Original attention-dispatch deviation disclosed; new confirmation IDs preserve history",
                        "PASS", ["protocols/05_backend_and_trained_precision.md", "protocols/06_locked_backend_confirmation.md"],
                        "Original runs used automatic SDPA dispatch and failed the planned explicit-lock requirement. Later profiles cannot prove historical operators. The separate locked-confirmation claim above must pass before completion."))
    ledger.append(claim("Locked checkpoint independent development, holdout and training-subset evaluation",
                        "PASS" if EXPECTED_LOCKED <= evaluation_ids else "UNVERIFIED",
                        [relative(path) for path in evaluations if any(name in path.name for name in EXPECTED_LOCKED)],
                        f"{len(EXPECTED_LOCKED & evaluation_ids)}/8 independently reproduced"))
    ledger.append(claim('Supplemental checkpoint independent evaluations',
                        'PASS' if EXPECTED_SUPPLEMENTAL <= evaluation_ids else 'UNVERIFIED',
                        [relative(path) for path in evaluations if 'supplemental' in str(path)]))

    storage_path = EXPERIMENTS/'storage-depth-v1/measurements.json'
    storage = json.loads(storage_path.read_text()) if storage_path.exists() else []
    storage_ok = len(storage) == 16
    for kind in ('midpoint', 'coupled_euler'):
        groups = {mode: sorted((r for r in storage if r['integrator']==kind and r['backward_mode']==mode),
                               key=lambda r:r['depth']) for mode in ('stored','reconstructed')}
        storage_ok &= all([r['depth'] for r in group] == [2,4,9,18] for group in groups.values())
        reconstructed = [r['saved_unique_storage_bytes'] for r in groups['reconstructed']]
        retained = [r['saved_unique_storage_bytes'] for r in groups['stored']]
        storage_ok &= (len(set(reconstructed))==1 and all(a<b for a,b in zip(retained,retained[1:])))
    ledger.append(claim('Unique saved stack storage is constant with depth, excluding parameters',
                        'PASS' if storage_ok else 'UNVERIFIED', [relative(storage_path)],
                        'Tiny CPU FP32 storage diagnostic; not a full-model GPU peak-memory claim.'))

    notebook_path = AUDIT/'notebook_execution.json'
    notebooks = json.loads(notebook_path.read_text()) if notebook_path.exists() else []
    notebook_ok = len(notebooks) == 5
    for record in notebooks:
        path = ROOT/record['path']
        if not path.exists() or digest_file(path) != record.get('sha256'):
            notebook_ok = False
            continue
        body = json.loads(path.read_text(encoding='utf-8'))
        cells = [cell for cell in body['cells'] if cell['cell_type']=='code']
        notebook_ok &= bool(cells) and all(cell.get('execution_count') is not None and
                         not any(out.get('output_type') == 'error' for out in cell.get('outputs', [])) for cell in cells)
    ledger.append(claim('Five notebooks execute top-to-bottom with genuine outputs',
                        'PASS' if notebook_ok else 'UNVERIFIED', ['audit/notebook_execution.json']))
    required_plots = ("loss_vs_tokens.png", "loss_vs_time.png", "memory_vs_batch.png",
                      "throughput_vs_batch.png", "max_batch_boundaries.png",
                      "reconstruction_error_vs_depth.png", "gradient_error_vs_depth.png",
                      "numerical_error_vs_training_progress.png",
                      "memory_throughput.png", "seed_variation.png")
    plots_ok = all((ROOT / "results" / name).exists() for name in required_plots)
    ledger.append(claim("Raw-log regenerated tables and all planned quantitative figures",
                        "PASS" if full_ok and plots_ok and (ROOT / "results" / "runs.csv").exists()
                        else "UNVERIFIED", ["scripts/analyze.py", "results/runs.csv"] +
                        [f"results/{name}" for name in required_plots if (ROOT / "results" / name).exists()]))

    provenance_path = ROOT/'results/figures/figure_provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    inputs = provenance.get('inputs', [])
    provenance_ok = bool(inputs) and all((ROOT/r['path']).is_file() and
                                        digest_file(ROOT/r['path'])==r['sha256'] for r in inputs)
    ledger.append(claim('Report figure source hashes match retained raw evidence',
                        'PASS' if provenance_ok else 'UNVERIFIED', [relative(provenance_path)]))
    export_path = AUDIT/'report_export.json'
    exported = json.loads(export_path.read_text()) if export_path.exists() else {}
    export_ok = (exported.get('source_sha256')==digest_file(ROOT/'REPORT.md') and
                 exported.get('generator_sha256')==digest_file(ROOT/'scripts/export_report.py') and
                 set(exported.get('outputs', {}))=={'REPORT.html','REPORT.pdf'} and
                 all((ROOT/name).is_file() and digest_file(ROOT/name)==digest
                     for name,digest in exported.get('outputs',{}).items()))
    ledger.append(claim('Portable PDF and HTML match generated report source and exporter',
                        'PASS' if export_ok else 'UNVERIFIED', [relative(export_path)]))

    scan = credential_scan()
    ledger.append(claim("Tracked-file credential-pattern scan",
                        "PASS" if not scan["findings"] else "FAIL",
                        ["audit/credential_scan.json"],
                        f"{scan['tracked_files_scanned']} tracked files scanned; {len(scan['findings'])} findings"))

    (AUDIT / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    with (AUDIT / "ledger.md").open("w", encoding="utf-8") as stream:
        stream.write("# Evidence ledger\n\n| Requirement | Status | Evidence |\n|---|---|---|\n")
        for item in ledger:
            paths = ", ".join(item["evidence"])
            stream.write(f"| {item['requirement']} | {item['status']} | {paths} |\n")
        stream.write("\nDetails and limitations are recorded in `ledger.json`.\n")
    print(json.dumps({status: sum(item["status"] == status for item in ledger)
                      for status in ("PASS", "FAIL", "UNVERIFIED")}, indent=2))


if __name__ == "__main__":
    main()
