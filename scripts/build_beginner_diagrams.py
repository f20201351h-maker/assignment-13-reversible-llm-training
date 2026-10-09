"""Generate exact, source-backed technical diagrams (docs/diagrams/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#12324A"
TEAL = "#0D7C83"
AMBER = "#ECA72C"
CORAL = "#E76F51"
BLUE = "#DCEEF4"
PALE_TEAL = "#DDF2EF"
PALE_AMBER = "#FFF1CF"
PALE_CORAL = "#FADFD7"
INK = "#1F2933"
MUTED = "#5E6C76"
WHITE = "#FFFFFF"
GRID = "#C8D6DC"


def setup(figsize=(15, 8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, xy, wh, title, lines=(), *, face=BLUE, edge=NAVY, title_size=13,
        body_size=10, title_color=NAVY, linewidth=1.7, align="center"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if align == "center" else x + 0.02
    ha = "center" if align == "center" else "left"
    ax.text(tx, y + h - 0.034, title, ha=ha, va="top", fontsize=title_size,
            color=title_color, weight="bold")
    if isinstance(lines, str):
        lines = [lines]
    if lines:
        ax.text(tx, y + h - 0.087, "\n".join(lines), ha=ha, va="top",
                fontsize=body_size, color=INK, linespacing=1.35)
    return patch


def arrow(ax, start, end, *, color=TEAL, width=2.0, style="-|>", connection="arc3"):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=15,
                            linewidth=width, color=color, connectionstyle=connection)
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, size=10, color=MUTED, weight="normal", ha="center"):
    ax.text(x, y, text, fontsize=size, color=color, weight=weight, ha=ha, va="center")


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def model_architecture():
    fig, ax = setup((16, 8.7))
    ax.text(0.03, 0.94, "Exact model architecture and tensor shapes", fontsize=22,
            color=NAVY, weight="bold", va="top")
    ax.text(0.03, 0.895, "All principal conditions keep the same 19,969,152 unique trainable parameters",
            fontsize=11, color=MUTED, va="top")

    y, h = 0.60, 0.19
    components = [
        (0.03, 0.11, "Token IDs", ["shape [B, T]", "T <= 512"], PALE_AMBER, AMBER),
        (0.17, 0.14, "Embeddings", ["token + position", "[B, T, 384]"], BLUE, NAVY),
        (0.35, 0.28, "Nine Transformer blocks", ["pre-LayerNorm", "6 heads x 64 dimensions", "FFN: 384 -> 1536 -> 384"], PALE_TEAL, TEAL),
        (0.67, 0.11, "Final norm", ["scale only", "[B, T, 384]"], BLUE, NAVY),
        (0.82, 0.15, "Tied LM head", ["same weights as tokens", "logits [B, T, 10,000]"], PALE_CORAL, CORAL),
    ]
    for x, w, title, lines, face, edge in components:
        box(ax, (x, y), (w, h), title, lines, face=face, edge=edge, title_size=12, body_size=9.5)
    for x1, x2 in ((0.14, 0.17), (0.31, 0.35), (0.63, 0.67), (0.78, 0.82)):
        arrow(ax, (x1, y + h / 2), (x2, y + h / 2))

    # Exact block detail
    box(ax, (0.18, 0.15), (0.64, 0.27), "Inside one block", [], face="#F8FAFB", edge=GRID)
    detail = [
        (0.22, 0.11, "LayerNorm", ["x"], BLUE, NAVY),
        (0.36, 0.13, "Causal attention", ["Q, K, V", "future masked"], PALE_TEAL, TEAL),
        (0.53, 0.10, "Add", ["x + A(x)"], PALE_AMBER, AMBER),
        (0.66, 0.11, "LayerNorm", ["residual"], BLUE, NAVY),
        (0.80, 0.13, "GELU MLP", ["4x expansion"], PALE_CORAL, CORAL),
    ]
    for x, w, title, lines, face, edge in detail:
        box(ax, (x, 0.215), (w, 0.12), title, lines, face=face, edge=edge,
            title_size=10, body_size=8)
    for x1, x2 in ((0.33, 0.36), (0.49, 0.53), (0.63, 0.66), (0.77, 0.80)):
        arrow(ax, (x1, 0.275), (x2, 0.275), width=1.6)
    arrow(ax, (0.93, 0.275), (0.93, 0.18), color=CORAL, connection="arc3,rad=0.4")
    arrow(ax, (0.93, 0.18), (0.53, 0.18), color=CORAL)
    label(ax, 0.74, 0.155, "second residual add", size=9, color=CORAL, weight="bold")

    label(ax, 0.5, 0.52, "The reversible and stored versions change how backward is executed, not the parameter count.",
          size=11, color=NAVY, weight="bold")
    save(fig, "01_model_architecture")


def training_loop():
    fig, ax = setup((15, 9))
    ax.text(0.04, 0.94, "What one training update actually does", fontsize=22,
            color=NAVY, weight="bold", va="top")
    ax.text(0.04, 0.895, "Example shapes use physical batch B, sequence T=512, hidden width 384",
            fontsize=11, color=MUTED, va="top")
    steps = [
        (0.05, 0.62, 0.16, "1  Batch", ["input IDs [B,512]", "targets [B,512]"]),
        (0.27, 0.62, 0.16, "2  Forward", ["embeddings", "attention + MLP", "logits [B,512,10000]"]),
        (0.49, 0.62, 0.16, "3  Loss", ["cross entropy", "sum valid targets", "lower is better"]),
        (0.71, 0.62, 0.22, "4  Backward", ["gradient for every parameter", "stored states OR reconstructed states"]),
    ]
    for i, (x, y, w, title, lines) in enumerate(steps):
        face = [PALE_AMBER, BLUE, PALE_CORAL, PALE_TEAL][i]
        edge = [AMBER, NAVY, CORAL, TEAL][i]
        box(ax, (x, y), (w, 0.20), title, lines, face=face, edge=edge, title_size=12, body_size=9)
    for a, b in ((0.21, 0.27), (0.43, 0.49), (0.65, 0.71)):
        arrow(ax, (a, 0.72), (b, 0.72))

    box(ax, (0.25, 0.27), (0.50, 0.18), "5  AdamW optimizer step",
        ["use accumulated, token-normalized gradients", "clip norm to 1.0", "update 19,969,152 learned parameters"],
        face="#F8FAFB", edge=NAVY, title_size=13, body_size=10)
    arrow(ax, (0.82, 0.62), (0.75, 0.45), color=TEAL, connection="arc3,rad=0.25")
    arrow(ax, (0.25, 0.36), (0.13, 0.62), color=AMBER, connection="arc3,rad=-0.35")
    label(ax, 0.12, 0.46, "repeat with next committed batch", size=10, color=AMBER, weight="bold")

    box(ax, (0.04, 0.07), (0.27, 0.11), "Token budget",
        ["50,000,000 valid next-token targets per full run"], face=PALE_AMBER, edge=AMBER,
        title_size=11, body_size=8.5)
    box(ax, (0.36, 0.07), (0.27, 0.11), "Mixed precision",
        ["FP16 suitable ops; FP32 parameters and reconstruction"], face=BLUE, edge=NAVY,
        title_size=11, body_size=8.5)
    box(ax, (0.68, 0.07), (0.27, 0.11), "Skipped overflow",
        ["replay same batch; do not advance data or schedule"], face=PALE_CORAL, edge=CORAL,
        title_size=11, body_size=8.5)
    save(fig, "02_training_update")


def reversible_math():
    fig, ax = setup((16, 10))
    ax.text(0.03, 0.95, "Coupled reversible block: exact forward and inverse", fontsize=22,
            color=NAVY, weight="bold", va="top")
    ax.text(0.03, 0.91, "Selected after eight pilots: coupled Euler, step size h = 0.5",
            fontsize=11, color=MUTED, va="top")

    ax.text(0.06, 0.83, "FORWARD", fontsize=14, weight="bold", color=TEAL)
    box(ax, (0.06, 0.58), (0.16, 0.17), "Input states", ["u", "v"], face=BLUE, edge=NAVY)
    box(ax, (0.31, 0.58), (0.24, 0.17), "Attention update", ["u' = u + h A(v)", "v is unchanged"], face=PALE_TEAL, edge=TEAL)
    box(ax, (0.64, 0.58), (0.25, 0.17), "MLP update", ["v' = v + h M(u')", "u' is unchanged"], face=PALE_AMBER, edge=AMBER)
    arrow(ax, (0.22, 0.665), (0.31, 0.665))
    arrow(ax, (0.55, 0.665), (0.64, 0.665))

    ax.text(0.06, 0.48, "BACKWARD RECONSTRUCTION", fontsize=14, weight="bold", color=CORAL)
    box(ax, (0.06, 0.22), (0.22, 0.17), "Saved boundary", ["u'", "v'"], face=PALE_CORAL, edge=CORAL)
    box(ax, (0.38, 0.22), (0.24, 0.17), "Recover v first", ["v = v' - h M(u')"], face=PALE_AMBER, edge=AMBER)
    box(ax, (0.71, 0.22), (0.22, 0.17), "Recover u", ["u = u' - h A(v)"], face=PALE_TEAL, edge=TEAL)
    arrow(ax, (0.28, 0.305), (0.38, 0.305), color=CORAL)
    arrow(ax, (0.62, 0.305), (0.71, 0.305), color=CORAL)

    arrow(ax, (0.84, 0.58), (0.84, 0.39), color=CORAL, width=2.4)
    label(ax, 0.88, 0.485, "same parameters\nrecompute locally", size=9.5, color=CORAL, weight="bold", ha="left")

    box(ax, (0.05, 0.02), (0.90, 0.14), "Custom whole-stack backward",
        ["Keep only stack input + final u/v boundaries. For each block in reverse: reconstruct earlier states -> rerun that block with gradients -> accumulate parameter gradients -> continue."],
        face="#F8FAFB", edge=NAVY, title_size=11, body_size=8.5)
    save(fig, "03_reversible_forward_inverse")


def storage_comparison():
    fig, ax = setup((16, 8.8))
    ax.text(0.03, 0.94, "Why reversible backward uses less activation memory", fontsize=22,
            color=NAVY, weight="bold", va="top")
    ax.text(0.03, 0.895, "Parameters and optimizer state are present on both sides; the comparison below focuses on stack activations",
            fontsize=11, color=MUTED, va="top")

    ax.text(0.04, 0.80, "Stored autograd", fontsize=14, weight="bold", color=CORAL)
    for i in range(9):
        x = 0.05 + i * 0.095
        face = PALE_CORAL if i % 2 == 0 else "#FFF7F3"
        box(ax, (x, 0.61), (0.065, 0.09), f"L{i+1}", [], face=face, edge=CORAL, title_size=9)
        ax.add_patch(Rectangle((x + 0.018, 0.56), 0.03, 0.035, facecolor=CORAL, edgecolor=CORAL))
        label(ax, x + 0.033, 0.535, "saved", size=7.2, color=CORAL)
        if i < 8:
            arrow(ax, (x + 0.065, 0.655), (x + 0.095, 0.655), color=CORAL, width=1.2)
    label(ax, 0.50, 0.48, "Saved activation storage grows with depth", size=11, color=CORAL, weight="bold")

    ax.text(0.04, 0.39, "Reconstructed reversible stack", fontsize=14, weight="bold", color=TEAL)
    for i in range(9):
        x = 0.05 + i * 0.095
        box(ax, (x, 0.20), (0.065, 0.09), f"L{i+1}", [], face=PALE_TEAL if i % 2 == 0 else "#F3FBF9", edge=TEAL, title_size=9)
        if i < 8:
            arrow(ax, (x + 0.065, 0.245), (x + 0.095, 0.245), color=TEAL, width=1.2)
    ax.add_patch(Rectangle((0.068, 0.145), 0.03, 0.035, facecolor=AMBER, edgecolor=AMBER))
    ax.add_patch(Rectangle((0.828, 0.145), 0.03, 0.035, facecolor=AMBER, edgecolor=AMBER))
    label(ax, 0.083, 0.12, "input", size=7.2, color=AMBER, weight="bold")
    label(ax, 0.843, 0.12, "boundary", size=7.2, color=AMBER, weight="bold")
    arrow(ax, (0.84, 0.19), (0.08, 0.19), color=AMBER, width=2.2, connection="arc3,rad=-0.17")
    label(ax, 0.50, 0.075, "Reconstruct one block at a time while walking backward", size=11, color=TEAL, weight="bold")
    save(fig, "04_activation_storage")


def causal_chain():
    fig, ax = setup((16, 9.2))
    ax.text(0.03, 0.95, "Causal experiment chain: change one factor at a time", fontsize=22,
            color=NAVY, weight="bold", va="top")
    ax.text(0.03, 0.91, "Seed 1337 supplies the complete A -> D -> B -> E -> C chain",
            fontsize=11, color=MUTED, va="top")
    xs = [0.03, 0.23, 0.43, 0.63, 0.83]
    titles = ["A  Baseline", "D  Architecture", "B  Reconstruction", "E  Effective batch", "C  Physical batch"]
    lines = [
        ["conventional", "stored", "physical 110", "effective 110"],
        ["coupled Euler", "stored", "physical 110", "effective 110"],
        ["coupled Euler", "reconstructed", "physical 110", "effective 110"],
        ["coupled Euler", "reconstructed", "physical 110 + 81", "effective 191"],
        ["coupled Euler", "reconstructed", "physical 191", "effective 191"],
    ]
    faces = [BLUE, PALE_TEAL, PALE_CORAL, PALE_AMBER, "#E7F1FB"]
    edges = [NAVY, TEAL, CORAL, AMBER, NAVY]
    for x, title, body, face, edge in zip(xs, titles, lines, faces, edges):
        box(ax, (x, 0.50), (0.15, 0.25), title, body, face=face, edge=edge, title_size=10.5, body_size=8.5)
    changes = [
        (0.18, 0.23, "change\narchitecture"),
        (0.38, 0.43, "change\nbackward"),
        (0.58, 0.63, "change\neffective batch"),
        (0.78, 0.83, "change\nphysical batch"),
    ]
    for x1, x2, txt in changes:
        arrow(ax, (x1, 0.625), (x2, 0.625), width=2.2)
        label(ax, (x1 + x2) / 2, 0.80, txt, size=8.5, color=TEAL, weight="bold")

    comparisons = [
        (0.03, 0.26, 0.35, "A vs D", "Architecture effect", "holdout improves by 0.1105"),
        (0.35, 0.26, 0.29, "D vs B", "Reconstruction effect", "48% memory saved; loss nearly same"),
        (0.66, 0.26, 0.30, "E vs C", "Physical batching", "no observed throughput benefit"),
    ]
    for x, y, w, title, subtitle, note in comparisons:
        box(ax, (x, y), (w, 0.13), title, [subtitle, note], face="#F8FAFB", edge=GRID,
            title_size=10, body_size=8.5)
    box(ax, (0.30, 0.04), (0.40, 0.14), "B vs E",
        ["Effective-batch change: holdout worsens by 0.3297", "888 updates at batch 110 vs 512 at batch 191"],
        face=PALE_AMBER, edge=AMBER, title_size=10, body_size=8)
    save(fig, "05_causal_experiment_chain")


def batch_accumulation():
    fig, ax = setup((15, 8.7))
    ax.text(0.03, 0.94, "Physical batch versus effective batch", fontsize=22,
            color=NAVY, weight="bold", va="top")
    ax.text(0.03, 0.895, "Conditions C and E process the same 191 sequences before one optimizer step",
            fontsize=11, color=MUTED, va="top")

    box(ax, (0.05, 0.55), (0.40, 0.24), "E  Gradient accumulation",
        ["microbatch 1: 110 sequences", "+ microbatch 2: 81 sequences", "one optimizer step after 191"],
        face=PALE_AMBER, edge=AMBER, title_size=13, body_size=10)
    ax.add_patch(Rectangle((0.09, 0.42), 0.24, 0.06, facecolor=AMBER, alpha=0.8, edgecolor=AMBER))
    ax.add_patch(Rectangle((0.33, 0.42), 0.09, 0.06, facecolor="#F6CC71", edgecolor=AMBER))
    label(ax, 0.21, 0.45, "110", size=10, color=NAVY, weight="bold")
    label(ax, 0.375, 0.45, "81", size=10, color=NAVY, weight="bold")

    box(ax, (0.55, 0.55), (0.40, 0.24), "C  One physical batch",
        ["191 sequences resident together", "one forward/backward", "one optimizer step after 191"],
        face=PALE_TEAL, edge=TEAL, title_size=13, body_size=10)
    ax.add_patch(Rectangle((0.59, 0.42), 0.32, 0.06, facecolor=TEAL, alpha=0.8, edgecolor=TEAL))
    label(ax, 0.75, 0.45, "191", size=10, color=WHITE, weight="bold")

    arrow(ax, (0.25, 0.40), (0.25, 0.27), color=AMBER)
    arrow(ax, (0.75, 0.40), (0.75, 0.27), color=TEAL)
    box(ax, (0.08, 0.11), (0.34, 0.13), "Same effective batch", ["same target total per update", "same fixed learning-rate schedule"], face="#F8FAFB", edge=GRID, title_size=11, body_size=8.5)
    box(ax, (0.58, 0.11), (0.34, 0.13), "Different physical batching", ["isolates whether more simultaneous work", "improves speed or quality"], face="#F8FAFB", edge=GRID, title_size=11, body_size=8.5)
    save(fig, "06_batch_accumulation")


def evidence_pipeline():
    fig, ax = setup((16, 9.5))
    ax.text(0.03, 0.95, "Evidence pipeline: from idea to defensible conclusion", fontsize=22,
            color=NAVY, weight="bold", va="top")
    ax.text(0.03, 0.91, "This is the work beyond simply launching a training script",
            fontsize=11, color=MUTED, va="top")
    stages = [
        (0.04, 0.67, "1  Freeze protocol", ["question", "comparison rules", "error gates"]),
        (0.28, 0.67, "2  Prove math/code", ["inverse", "gradients", "resume"]),
        (0.52, 0.67, "3  Select candidate", ["8 pilots x 2M", "midpoint + coupled"]),
        (0.76, 0.67, "4  Find capacity", ["fresh OOM search", "stable + failing"]),
        (0.04, 0.35, "5  Full training", ["8 locked main runs", "50M each"]),
        (0.28, 0.35, "6  Benchmark", ["20 warm-up", "100 measured x 3"]),
        (0.52, 0.35, "7  Independent eval", ["reload checkpoints", "exact holdout replay"]),
        (0.76, 0.35, "8  Audit/report", ["hash provenance", "26/26 PASS"]),
    ]
    palette = [(BLUE, NAVY), (PALE_TEAL, TEAL), (PALE_AMBER, AMBER), (PALE_CORAL, CORAL)] * 2
    for (x, y, title, lines), (face, edge) in zip(stages, palette):
        box(ax, (x, y), (0.19, 0.17), title, lines, face=face, edge=edge, title_size=11, body_size=8.5)
    for a, b in [((0.23, 0.755), (0.28, 0.755)), ((0.47, 0.755), (0.52, 0.755)), ((0.71, 0.755), (0.76, 0.755)),
                 ((0.23, 0.435), (0.28, 0.435)), ((0.47, 0.435), (0.52, 0.435)), ((0.71, 0.435), (0.76, 0.435))]:
        arrow(ax, a, b, width=1.7)
    ax.plot([0.855, 0.96, 0.96, 0.135], [0.67, 0.61, 0.56, 0.56], color=CORAL, linewidth=1.8)
    arrow(ax, (0.135, 0.56), (0.135, 0.52), color=CORAL, width=1.8)
    label(ax, 0.53, 0.585, "Only after capacity is observed do the frozen full-run identities begin", size=9.5, color=CORAL, weight="bold")
    box(ax, (0.18, 0.08), (0.64, 0.14), "Final retained evidence",
        ["21 full 50M-target runs | 1.05B cumulative committed targets | 43 tests", "8 exact independent holdout rechecks | 6 trained reversible mixed-precision PASS | 21 isolated benchmark repetitions"],
        face="#F8FAFB", edge=NAVY, title_size=12, body_size=9)
    save(fig, "07_evidence_pipeline")


def main():
    model_architecture()
    training_loop()
    reversible_math()
    storage_comparison()
    causal_chain()
    batch_accumulation()
    evidence_pipeline()
    print(f"Generated technical diagrams in {OUT}")


if __name__ == "__main__":
    main()
