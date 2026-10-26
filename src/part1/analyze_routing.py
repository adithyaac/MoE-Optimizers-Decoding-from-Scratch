"""What do the experts specialise on? Routing by source language vs by token type.

    python -m src.part1.analyze_routing --ckpt-root checkpoints/part1 --out results/part1/routing

For each MoE variant, routes n test examples per language and reports, per layer,
the fraction of routing slots each expert receives for:
  - source language (vi / ja)          -> language specialisation?
  - token type (lang tag, digit, punctuation, CJK, Latin, ...) -> functional specialisation?
Also the Jensen-Shannon divergence between the vi and ja distributions per layer
(0 = routed identically, ln 2 = 0.69 = completely disjoint experts).

Writes <out>/<variant>.json and <out>/<variant>_token_types.png.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import torch
from tokenizers import Tokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.part1.data import PAD_ID, TranslationDataset, collate  # noqa: E402
from src.part1.evaluate import load_model  # noqa: E402

VARIANTS = ["2_moe_4e_top1", "3_moe_4e_top2", "4_moe_shared1_3e_top1", "5_moe_4e_top2_active"]
TYPES = ["lang tag", "digit", "punct", "CJK", "Latin", "other"]


def token_type(tokenizer: Tokenizer, tid: int) -> str:
    if tokenizer.id_to_token(tid).startswith("<") and tid < 5:
        return "lang tag"  # only <vi>/<ja> appear on the source side
    s = tokenizer.decode([tid])
    if re.fullmatch(r"\s*[0-9]+\s*", s):
        return "digit"
    if re.fullmatch(r"\s*[^\w\s]+\s*", s):
        return "punct"
    if any("぀" <= ch <= "鿿" for ch in s):  # hiragana, katakana, CJK ideographs
        return "CJK"
    if re.search(r"[A-Za-zÀ-ỹ]", s):  # Latin incl. Vietnamese diacritics
        return "Latin"
    return "other"  # partial UTF-8 byte pieces etc.


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    m = (p + q) / 2
    kl = lambda a, b: (a * (a / b).log()).nan_to_num().sum()  # noqa: E731
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


@torch.no_grad()
def analyse(model, ds: TranslationDataset, tokenizer: Tokenizer, n: int, device, batch_size: int = 256) -> dict:
    n_half = len(ds) // 2
    idx = list(range(min(n, n_half))) + list(range(n_half, n_half + min(n, n_half)))
    type_of = torch.tensor([TYPES.index(token_type(tokenizer, t)) for t in range(tokenizer.get_vocab_size())])
    moe = model.moe_layers()
    E = moe[0].n_experts
    by_lang = torch.zeros(len(moe), 2, E)  # [layer, lang, expert]
    by_type = torch.zeros(len(moe), len(TYPES), E)  # [layer, type, expert]
    for b in range(0, len(idx), batch_size):
        batch = collate([ds[i] for i in idx[b : b + batch_size]])
        x, y, lang = batch["input_ids"].to(device), batch["targets"].to(device), batch["lang"].to(device)
        model(x, y, mask=x != PAD_ID)
        src = ((x != PAD_ID) & (y == -100)).cpu()  # source side: <lang> tag + source tokens
        types = type_of[x.cpu()]
        langs = lang.cpu()[:, None].expand_as(x)
        for li, m in enumerate(moe):
            topk = m.last_topk.view(*x.shape, m.top_k).cpu()
            for k in range(m.top_k):
                e = topk[..., k][src]
                by_lang[li].index_put_((langs[src], e), torch.ones(len(e)), accumulate=True)
                by_type[li].index_put_((types[src], e), torch.ones(len(e)), accumulate=True)

    frac = lambda c: c / c.sum(-1, keepdim=True).clamp(min=1)  # noqa: E731
    fl, ft = frac(by_lang), frac(by_type)
    return {
        "experts": E,
        "types": TYPES,
        "type_counts": by_type.sum(-1)[0].tolist(),
        "js_vi_ja": [js_divergence(fl[li, 0], fl[li, 1]) for li in range(len(moe))],
        "lang_frac": fl.tolist(),
        "type_frac": ft.tolist(),
    }


def plot(res: dict, title: str, path: Path) -> None:
    """One heatmap per layer: rows = vi, ja, then token types; columns = experts."""
    lf, tf = torch.tensor(res["lang_frac"]), torch.tensor(res["type_frac"])
    keep = [i for i, c in enumerate(res["type_counts"]) if c >= 100]  # skip near-empty types
    rows = ["vi (all)", "ja (all)"] + [res["types"][i] for i in keep]
    L, E = lf.shape[0], res["experts"]
    ncols = 3  # 2 x 3 grid for 6 layers: readable on a portrait page
    nrows = (L + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.2 + 2.6 * ncols * E / 4, 0.6 + nrows * (0.35 + 0.3 * len(rows))),
                             sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for ax in axes[L:]:
        ax.set_visible(False)
    for li, ax in enumerate(axes[:L]):
        mat = torch.cat([lf[li], tf[li, keep]])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r in range(mat.shape[0]):
            for e in range(E):
                v = mat[r, e].item()
                ax.text(e, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else "#1f2937")
        ax.axhline(1.5, color="white", linewidth=2)  # separate language rows from type rows
        ax.set_title(f"layer {li}  (JS vi/ja {res['js_vi_ja'][li]:.3f})", fontsize=9)
        ax.set_xticks(range(E), [f"E{e}" for e in range(E)], fontsize=8)
        ax.set_yticks(range(len(rows)), rows, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="fraction of routing slots")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", type=Path, default=Path("checkpoints/part1"))
    p.add_argument("--data", type=Path, default=Path("data/part1"))
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("results/part1/routing"))
    p.add_argument("--n", type=int, default=2000, help="test examples per language")
    p.add_argument("--replot", action="store_true", help="only redraw figures from existing <out>/*.json")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        for f in sorted(args.out.glob("*.json")):
            plot(json.loads(f.read_text()), f"{f.stem}: source-token routing by language and token type",
                 args.out / f"{f.stem}_token_types.png")
        print(f"redrew figures in {args.out}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ds = TranslationDataset(args.data / "test.pt")
    for v in VARIANTS:
        ckpt = args.ckpt_root / v / "model.pt"
        if not ckpt.exists():
            print(f"skip {v}: no {ckpt}")
            continue
        res = analyse(load_model(ckpt, device), ds, tokenizer, args.n, device)
        (args.out / f"{v}.json").write_text(json.dumps(res, indent=1))
        plot(res, f"{v}: source-token routing by language and token type", args.out / f"{v}_token_types.png")
        print(f"{v}: JS(vi, ja) per layer = {[round(j, 3) for j in res['js_vi_ja']]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
