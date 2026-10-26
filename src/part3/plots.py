"""Part 3: likelihood vs diversity of each decoding config, from results/part3/summary.json.

    python -m src.part3.plots --results results/part3

  tradeoff.png   x: distinct-2 (diversity), y: perplexity under Pythia-410M (log);
                 the human reference is the target point
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMILY = {"greedy": "greedy / beam", "beam": "greedy / beam", "topk": "top-k", "topp": "top-p"}
COLORS = {"greedy / beam": "#2a78d6", "top-k": "#eb6834", "top-p": "#1baf7a", "reference": "#0b0b0b"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def family(name: str) -> str:
    if name == "reference":
        return "reference"
    return next(v for k, v in FAMILY.items() if name.startswith(k))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/part3"))
    args = p.parse_args()
    rows = [r for r in json.loads((args.results / "summary.json").read_text()) if r["config"] != "beam1"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    seen = set()
    for r in rows:
        fam = family(r["config"])
        ax.scatter(r["distinct2"], r["ppl_scorer"], s=260 if fam == "reference" else 55, color=COLORS[fam],
                   marker="*" if fam == "reference" else "o", edgecolor="white", linewidth=1.5, zorder=3,
                   label=None if fam in seen else fam)
        seen.add(fam)
        name = "human reference" if fam == "reference" else r["config"].replace("_t0.7", " (T=0.7)")
        # hand-placed offsets where points sit close together
        dx, dy, ha = {"greedy": (6, 7, "left"), "beam2": (7, -4, "left"),
                      "beam4": (6, -12, "left"), "topk50_t0.7": (7, -10, "left"),
                      "topp0.9_t0.7": (-7, 6, "right"), "reference": (9, -3, "left")}.get(r["config"], (6, 6, "left"))
        ax.annotate(name, (r["distinct2"], r["ppl_scorer"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=INK, ha=ha)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_xlabel("distinct-2 (share of unique bigrams; higher = more diverse)", color=INK)
    ax.set_ylabel("perplexity under Pythia-410M (log scale)", color=INK)
    ax.set_title("Decoding: likelihood vs diversity (greedy = beam1, omitted)", fontsize=11, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(args.results / "tradeoff.png", dpi=160)
    plt.close(fig)
    print(f"wrote {args.results / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
