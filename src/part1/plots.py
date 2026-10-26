"""Comparison plots across the five Part 1 variants (for the report).

    python -m src.part1.plots --results results/part1 --wandb-path <entity>/anlp-a2-part1

Writes to <results>/plots/:
  val_ppl_curves.png   validation perplexity vs training tokens (from WandB), all variants
  test_metrics.png     test perplexity and BLEU per variant, split by source language
  expert_load.png      per-expert load over training, one panel per MoE variant (from WandB)
  summary.md           table of test metrics + params, ready to paste
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VARIANTS = ["1_dense", "2_moe_4e_top1", "3_moe_4e_top2", "4_moe_shared1_3e_top1", "5_moe_4e_top2_active"]
LABELS = {"1_dense": "dense", "2_moe_4e_top1": "4e top-1", "3_moe_4e_top2": "4e top-2",
          "4_moe_shared1_3e_top1": "1 shared + 3e top-1", "5_moe_4e_top2_active": "4e top-2 (active-matched)"}
# fixed categorical order (colour follows the variant, never its rank)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LANG_COLORS = {"vi": "#2a78d6", "ja": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def wandb_histories(path: str, keys: list[str]) -> dict[str, list[dict]]:
    """Latest finished run per variant name -> list of logged rows that have ALL `keys`.

    scan_history can also return rows where some keys are None (e.g. training-log rows have
    `tokens` but no `val/ppl`); those would put gaps between points and break the lines."""
    import wandb
    runs = wandb.Api().runs(path)
    out = {}
    for run in sorted(runs, key=lambda r: r.created_at):  # later runs overwrite earlier ones
        if run.name in VARIANTS and run.state == "finished":
            out[run.name] = [r for r in run.scan_history(keys=keys) if all(r.get(k) is not None for k in keys)]
    return out


def plot_val_curves(hist: dict, path: Path, zoom_from: float = 20.0) -> None:
    """Left: whole run, log scale. Right: second half, linear scale, where the curves are
    too close to label individually -- final values go in the legend instead."""
    fig, (ax, az) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for v, c in zip(VARIANTS, COLORS):
        rows = sorted(hist.get(v, []), key=lambda r: r["tokens"])
        if not rows:
            continue
        x = [r["tokens"] / 1e6 for r in rows]
        y = [r["val/ppl"] for r in rows]
        ax.plot(x, y, color=c, linewidth=2, marker="o", markersize=4)
        zx, zy = zip(*[(a, b) for a, b in zip(x, y) if a >= zoom_from])
        az.plot(zx, zy, color=c, linewidth=2, marker="o", markersize=4, label=f"{LABELS[v]}  ({y[-1]:.2f})")
    ax.set_yscale("log")
    ax.set_ylabel("validation perplexity (log scale)", color=INK)
    ax.set_title("Validation perplexity: whole run", fontsize=11, color=INK, loc="left")
    az.set_ylabel("validation perplexity", color=INK)
    az.set_title(f"Zoom: from {zoom_from:.0f}M tokens (final value in legend)", fontsize=11, color=INK, loc="left")
    for a in (ax, az):
        a.set_xlabel("training tokens (M)", color=INK)
        style(a)
    az.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_test_metrics(metrics: dict, path: Path) -> None:
    vs = [v for v in VARIANTS if v in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    w = 0.38
    for ax, key, title in ((axes[0], "test_ppl", "Test perplexity (lower is better)"),
                           (axes[1], "bleu", "Test BLEU (higher is better)")):
        for j, lang in enumerate(("vi", "ja")):
            xs = [i + (j - 0.5) * w for i in range(len(vs))]
            ys = [metrics[v][f"{key}_{lang}"] for v in vs]
            bars = ax.bar(xs, ys, width=w - 0.04, color=LANG_COLORS[lang], label=f"{lang} → en")
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}" if key == "test_ppl" else f"{y:.1f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(vs)), [LABELS[v] for v in vs], rotation=20, ha="right")
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=22)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)  # headroom for the value labels
        style(ax)
        # legend above the plot, right-aligned, so it never covers bars or labels
        ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_expert_load(hist: dict, path: Path, layer: int) -> None:
    moe = [v for v in VARIANTS[1:] if hist.get(v)]
    if not moe:
        return
    fig, axes = plt.subplots(1, len(moe), figsize=(3.6 * len(moe), 3.4), sharey=True, constrained_layout=True)
    axes = [axes] if len(moe) == 1 else axes
    for ax, v in zip(axes, moe):
        rows = sorted(hist[v], key=lambda r: r["tokens"])
        n_exp = 4 if f"load/layer{layer}_expert3" in rows[0] else 3
        x = [r["tokens"] / 1e6 for r in rows]
        for e in range(n_exp):
            ax.plot(x, [r[f"load/layer{layer}_expert{e}"] for r in rows], color=COLORS[e], linewidth=1.5,
                    label=f"expert {e}")
        ax.axhline(1 / n_exp, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(LABELS[v], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training tokens (M)", color=INK)
        style(ax)
    axes[0].set_ylabel(f"share of routing slots, layer {layer}", color=INK)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Expert load during training (dashed = perfectly balanced)", fontsize=11, color=INK)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/part1"))
    p.add_argument("--wandb-path", default=None, help="<entity>/<project>; omit to skip WandB plots")
    p.add_argument("--load-layer", type=int, default=3)
    args = p.parse_args()
    out = args.results / "plots"
    out.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for v in VARIANTS:
        f = args.results / v / "metrics.json"
        if f.exists():
            metrics[v] = json.loads(f.read_text())
    plot_test_metrics(metrics, out / "test_metrics.png")

    lines = ["| variant | test ppl | ppl vi | ppl ja | BLEU | BLEU vi | BLEU ja |", "|---|---|---|---|---|---|---|"]
    for v, m in metrics.items():
        lines.append(f"| {LABELS[v]} | {m['test_ppl']:.2f} | {m['test_ppl_vi']:.2f} | {m['test_ppl_ja']:.2f} | "
                     f"{m['bleu']:.2f} | {m['bleu_vi']:.2f} | {m['bleu_ja']:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    val_file = out / "val_history.json"  # cached so the curves can be redrawn without WandB access
    if args.wandb_path:
        val_file.write_text(json.dumps(wandb_histories(args.wandb_path, ["tokens", "val/ppl"]), indent=1))
    if val_file.exists():
        plot_val_curves(json.loads(val_file.read_text()), out / "val_ppl_curves.png")
    if args.wandb_path:
        # scan_history only returns rows with ALL keys, and variant 4 has 3 routed experts (others 4)
        keys = lambda n: ["tokens"] + [f"load/layer{args.load_layer}_expert{e}" for e in range(n)]  # noqa: E731
        h4, h3 = wandb_histories(args.wandb_path, keys(4)), wandb_histories(args.wandb_path, keys(3))
        load = {v: h4.get(v) or h3.get(v) for v in VARIANTS[1:]}
        plot_expert_load(load, out / "expert_load.png", args.load_layer)
    print(f"wrote plots to {out}")


if __name__ == "__main__":
    main()
