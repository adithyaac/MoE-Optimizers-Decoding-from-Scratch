"""Part 3: compare decoding strategies with Pythia-160M on ROCStories.

    uv run python -m src.part3.run --n 1000

Task: given a story's first sentence (`prompt`), generate the rest (`continuation`).
For every decoding config, on the same n test stories:
  - perplexity of the generated text, under Pythia-160M itself and under a larger
    scorer model (Pythia-410M, same tokenizer) -- how likely / fluent the text is
  - BLEU and ROUGE-L against the reference continuation (the "accuracy" metrics)
  - diversity: distinct-1/2 and repeated trigram rate
  - wall-clock time (for the beam-width analysis)
Writes results/part3/<config>/generations.jsonl, results/part3/summary.{json,csv}
and results/part3/beam_time.png.
"""

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import sacrebleu
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import src.utils  # noqa: F401  (network fixes)
from src.part3.decoding import beam_search, pick_greedy, pick_top_k, pick_top_p, sample_loop
from src.utils import load_hf_parquet

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL = "EleutherAI/pythia-160m"
DATASET = "hamishivi/ROCStories"

# name -> (kind, params). Sampling configs use temperature 1.0 unless noted.
CONFIGS = {
    "greedy": ("sample", {"pick": pick_greedy}),
    "beam1": ("beam", {"width": 1}),
    "beam2": ("beam", {"width": 2}),
    "beam4": ("beam", {"width": 4}),
    "topk10": ("sample", {"pick": pick_top_k(10)}),
    "topk50": ("sample", {"pick": pick_top_k(50)}),
    "topk50_t0.7": ("sample", {"pick": pick_top_k(50, temperature=0.7)}),
    "topp0.9": ("sample", {"pick": pick_top_p(0.9)}),
    "topp0.95": ("sample", {"pick": pick_top_p(0.95)}),
    "topp0.9_t0.7": ("sample", {"pick": pick_top_p(0.9, temperature=0.7)}),
}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def generate_all(model, prompt_ids: list[list[int]], kind: str, params: dict, max_new: int,
                 stop_ids: set[int], banned: torch.Tensor, batch_size: int, device, seed: int) -> list[list[int]]:
    """Batch prompts by exact length (no padding needed), run one decoding config."""
    by_len = defaultdict(list)
    for i, ids in enumerate(prompt_ids):
        by_len[len(ids)].append(i)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = [None] * len(prompt_ids)
    bs = batch_size // params.get("width", 1)  # beam search multiplies the batch by width
    for L in sorted(by_len):
        idxs = by_len[L]
        for b in range(0, len(idxs), bs):
            chunk = idxs[b : b + bs]
            prompts = torch.tensor([prompt_ids[i] for i in chunk], device=device)
            if kind == "beam":
                res = beam_search(model, prompts, max_new, stop_ids, params["width"], banned=banned)
            else:
                res = sample_loop(model, prompts, max_new, stop_ids, params["pick"], gen, banned)
            for i, r in zip(chunk, res):
                out[i] = r
    return out


@torch.no_grad()
def perplexity(model, prompt_ids: list[list[int]], cont_ids: list[list[int]], device, batch_size: int = 32) -> float:
    """exp(mean NLL) of the continuation tokens given the prompt, token-weighted.
    Right padding is safe here: with causal attention, real tokens never see later pads."""
    nll, n = 0.0, 0
    pairs = [(p, c) for p, c in zip(prompt_ids, cont_ids) if c]
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b : b + batch_size]
        T = max(len(p) + len(c) for p, c in chunk)
        x = torch.zeros(len(chunk), T, dtype=torch.long)
        y = torch.full((len(chunk), T), -100, dtype=torch.long)
        for r, (p, c) in enumerate(chunk):
            seq = p + c
            x[r, : len(seq)] = torch.tensor(seq)
            y[r, len(p) - 1 : len(seq) - 1] = torch.tensor(c)  # position t predicts token t+1
        logits = model(input_ids=x.to(device)).logits.float()
        loss = F.cross_entropy(logits.transpose(1, 2), y.to(device), ignore_index=-100, reduction="sum")
        nll += loss.item()
        n += int((y != -100).sum())
    return math.exp(nll / n)


def rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F1 on lowercase whitespace tokens (longest common subsequence)."""
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in h:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(h), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def diversity(gens: list[list[int]]) -> dict:
    """distinct-n = unique n-grams / total n-grams over all generations (higher = more varied);
    rep3 = mean fraction of each generation's trigrams that repeat an earlier trigram in it."""
    out = {}
    for n in (1, 2):
        grams = [tuple(g[i : i + n]) for g in gens for i in range(len(g) - n + 1)]
        out[f"distinct{n}"] = len(set(grams)) / max(1, len(grams))
    reps = []
    for g in gens:
        tri = [tuple(g[i : i + 3]) for i in range(len(g) - 2)]
        if tri:
            reps.append(1 - len(set(tri)) / len(tri))
    out["rep3"] = sum(reps) / max(1, len(reps))
    return out


def plot_beam_time(rows: list[dict], path: Path) -> None:
    beams = sorted((r for r in rows if r["config"].startswith("beam")), key=lambda r: r["width"])
    if not beams:
        return
    w = [r["width"] for r in beams]
    t = [r["time_s"] for r in beams]
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    ax.plot(w, t, marker="o", color="#2563eb", linewidth=2, markersize=7)
    for x, y in zip(w, t):
        ax.annotate(f"{y:.0f}s", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(w)
    ax.set_xlabel("beam width")
    ax.set_ylabel(f"wall-clock time for {beams[0]['n']} stories (s)")
    ax.set_title("Beam search time vs. width", fontsize=11)
    ax.grid(axis="y", color="#e5e7eb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(0, max(t) * 1.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="test stories")
    p.add_argument("--max-new", type=int, default=64, help="max generated tokens")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--scorer", default="EleutherAI/pythia-410m", help="larger model for generation ppl ('none' to skip)")
    p.add_argument("--out", type=Path, default=Path("results/part3"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    scorer = (AutoModelForCausalLM.from_pretrained(args.scorer, dtype=torch.float32).to(device).eval()
              if args.scorer != "none" else None)

    test = load_hf_parquet(DATASET)["test"]
    picks = sorted(random.Random(args.seed).sample(range(len(test["prompt"])), args.n))
    prompts = [test["prompt"][i] for i in picks]
    refs = [test["continuation"][i] for i in picks]
    prompt_ids = [tok(s).input_ids for s in prompts]
    ref_ids = [tok(" " + s).input_ids for s in refs]
    # A story is one paragraph: newline tokens are banned (Pythia's most likely token after
    # the first sentence is "\n", which would otherwise end most generations immediately).
    # Generation stops at end-of-text or after max_new tokens.
    stop_ids = {tok.eos_token_id}
    banned = torch.tensor([i for i in range(len(tok)) if "\n" in tok.decode([i])], device=device)

    rows = []
    ref_row = {"config": "reference", "n": args.n, "ppl_self": perplexity(model, prompt_ids, ref_ids, device)}
    if scorer is not None:
        ref_row["ppl_scorer"] = perplexity(scorer, prompt_ids, ref_ids, device)
    ref_row |= diversity(ref_ids) | {"avg_len": sum(map(len, ref_ids)) / len(ref_ids)}
    print(ref_row)
    rows.append(ref_row)

    for name in args.configs:
        kind, params = CONFIGS[name]
        sync(device)
        t0 = time.perf_counter()
        gens = generate_all(model, prompt_ids, kind, params, args.max_new, stop_ids, banned,
                            args.batch_size, device, args.seed)
        sync(device)
        elapsed = time.perf_counter() - t0
        hyps = [tok.decode(g).strip() for g in gens]

        row = {"config": name, "n": args.n, "time_s": elapsed, "width": params.get("width")}
        row["ppl_self"] = perplexity(model, prompt_ids, gens, device)
        if scorer is not None:
            row["ppl_scorer"] = perplexity(scorer, prompt_ids, gens, device)
        row["bleu"] = sacrebleu.corpus_bleu(hyps, [refs]).score
        row["rouge_l"] = sum(rouge_l(h, r) for h, r in zip(hyps, refs)) / len(refs)
        row |= diversity(gens) | {"avg_len": sum(map(len, gens)) / len(gens),
                                  "empty": sum(1 for g in gens if not g)}
        rows.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

        d = args.out / name
        d.mkdir(exist_ok=True)
        with open(d / "generations.jsonl", "w", encoding="utf-8") as f:
            for pr, ref, hyp in zip(prompts, refs, hyps):
                f.write(json.dumps({"prompt": pr, "reference": ref, "generation": hyp}, ensure_ascii=False) + "\n")

    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    plot_beam_time(rows, args.out / "beam_time.png")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
