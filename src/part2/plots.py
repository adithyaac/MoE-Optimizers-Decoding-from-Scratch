"""Part 2 plots from the history.json files written by src.part2.train.

    python -m src.part2.plots --root checkpoints/part2 --out results/part2

  curves.png   validation loss and test BLEU vs dataset fraction (0.1x steps), one line per optimizer
  sweep.png    (if sweep runs exist) val loss at 0.1x dataset vs learning rate, per optimizer
  summary.md   final numbers per optimizer
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OPTIMIZERS = ["adamw", "mars", "lion", "muon"]
LABELS = {"adamw": "AdamW", "mars": "MARS", "lion": "Lion", "muon": "Muon"}
COLORS = {"adamw": "#2a78d6", "mars": "#eb6834", "lion": "#1baf7a", "muon": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(root: Path) -> tuple[dict, list[dict]]:
    """Full runs keyed by optimizer, plus the list of short sweep runs (run names 'sweep_*')."""
    main, sweep = {}, []
    for f in sorted(root.glob("*/history.json")):
        h = json.loads(f.read_text())
        if h["run_name"].startswith("sweep_"):
            sweep.append(h)
        else:
            main[h["optimizer"]] = h
    return main, sweep


def plot_curves(main: dict, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for o in OPTIMIZERS:
        if o not in main:
            continue
        hist = main[o]["history"]
        x = [r["dataset_frac"] for r in hist]
        a1.plot(x, [r["val_loss"] for r in hist], color=COLORS[o], linewidth=2, marker="o", markersize=4,
                label=f"{LABELS[o]}  ({hist[-1]['val_loss']:.3f})")
        a2.plot(x, [r["test_bleu"] for r in hist], color=COLORS[o], linewidth=2, marker="o", markersize=4,
                label=f"{LABELS[o]}  ({hist[-1]['test_bleu']:.2f})")
    a1.set_title("Validation loss (final value in legend)", fontsize=11, color=INK, loc="left")
    a1.set_ylabel("validation loss (nats / token)", color=INK)
    a2.set_title("Test BLEU, 32-token greedy continuation", fontsize=11, color=INK, loc="left")
    a2.set_ylabel("BLEU", color=INK)
    for a in (a1, a2):
        a.set_xlabel("training tokens (x dataset)", color=INK)
        a.set_xticks([i / 10 for i in range(1, 11)])
        style(a)
        a.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sweep(sweep: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for o in OPTIMIZERS:
        pts = sorted((h["lr"], h["history"][-1]["val_loss"]) for h in sweep if h["optimizer"] == o)
        if pts:
            ax.plot(*zip(*pts), color=COLORS[o], linewidth=2, marker="o", markersize=6, label=LABELS[o])
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate (log scale)", color=INK)
    ax.set_ylabel("validation loss after 0.1x dataset", color=INK)
    ax.set_title("Learning-rate sweep (short runs)", fontsize=11, color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("checkpoints/part2"), help="folder with <run>/history.json")
    p.add_argument("--out", type=Path, default=Path("results/part2"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main_runs, sweep = load(args.root)

    for o, h in main_runs.items():  # keep the small history files next to the plots
        shutil.copy(args.root / h["run_name"] / "history.json", args.out / f"{o}_history.json")
    plot_curves(main_runs, args.out / "curves.png")
    lines = ["| optimizer | peak lr | final val loss | final val ppl | final test BLEU | best test BLEU |",
             "|---|---|---|---|---|---|"]
    for o in OPTIMIZERS:
        if o in main_runs:
            hist = main_runs[o]["history"]
            lines.append(f"| {LABELS[o]} | {main_runs[o]['lr']:g} | {hist[-1]['val_loss']:.4f} | "
                         f"{hist[-1]['val_ppl']:.2f} | {hist[-1]['test_bleu']:.2f} | "
                         f"{max(r['test_bleu'] for r in hist):.2f} |")
    if sweep:
        plot_sweep(sweep, args.out / "sweep.png")
        lines += ["", "LR sweep (0.1x dataset):", "", "| optimizer | lr | val loss |", "|---|---|---|"]
        for h in sorted(sweep, key=lambda h: (h["optimizer"], h["lr"])):
            lines.append(f"| {LABELS[h['optimizer']]} | {h['lr']:g} | {h['history'][-1]['val_loss']:.4f} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
