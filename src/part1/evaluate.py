"""Evaluate one trained Part 1 variant on the test split.

    uv run python -m src.part1.evaluate --ckpt checkpoints/part1/2_moe_4e_top1/model.pt

Writes to results/part1/<variant>/:
  metrics.json       test loss / perplexity (full test set), BLEU (subset), per language
  generations.jsonl  source, reference, hypothesis for every BLEU sample
  expert_usage.json  routing counts [layer, lang, side, expert]   (MoE only)
  expert_heatmap.png fraction of source tokens per expert, per layer and language (MoE only)
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import sacrebleu
import torch
from tokenizers import Tokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.part1.data import IGNORE, LANGS, PAD_ID, TranslationDataset, make_loader  # noqa: E402
from src.part1.model import Transformer, TransformerConfig  # noqa: E402

LANG_NAMES = {v: k for k, v in LANGS.items()}  # 0 -> "vi", 1 -> "ja"
SIDES = ("source", "target")


def load_model(path: Path, device) -> Transformer:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Transformer(TransformerConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


@torch.no_grad()
def perplexity_and_routing(model: Transformer, loader, device, amp_dtype) -> tuple[dict, torch.Tensor | None]:
    """Token-weighted test loss per language, plus routing counts for MoE models.

    counts[layer, lang, side, expert]: how many routing slots of real tokens went to
    each expert. side 0 = source tokens (<lang> + source), 1 = target side (<sep> + English).
    """
    moe = model.moe_layers()
    counts = torch.zeros(len(moe), 2, 2, moe[0].n_experts, dtype=torch.long) if moe else None
    loss_sum, n_tok = defaultdict(float), defaultdict(int)
    for batch in loader:
        x, y, lang = batch["input_ids"].to(device), batch["targets"].to(device), batch["lang"].to(device)
        mask = x != PAD_ID
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            # per-token loss (B, T), so we can split it by language
            _, tok_loss, _ = model(x, y, mask=mask, reduction="none")
        valid = y != IGNORE
        for l in (0, 1):
            sel = valid & (lang == l)[:, None]
            loss_sum[l] += tok_loss[sel].sum().item()
            n_tok[l] += int(sel.sum())

        if moe:
            side = valid.long()  # positions with an English target = target side
            lang_tok = lang[:, None].expand_as(x)
            for li, m in enumerate(moe):
                k = m.top_k
                topk = m.last_topk.view(*x.shape, k)  # (B, T, k)
                sel = mask[..., None].expand(-1, -1, k)
                cell = lang_tok * 2 + side  # (B, T): which (lang, side) bucket each token is in
                flat = (cell[..., None].expand(-1, -1, k)[sel] * m.n_experts + topk[sel]).cpu()
                counts[li] += torch.bincount(flat, minlength=4 * m.n_experts).view(2, 2, m.n_experts)

    metrics = {}
    for l in (0, 1):
        metrics[f"test_loss_{LANG_NAMES[l]}"] = loss_sum[l] / n_tok[l]
        metrics[f"test_ppl_{LANG_NAMES[l]}"] = math.exp(loss_sum[l] / n_tok[l])
    total = sum(loss_sum.values()) / sum(n_tok.values())
    metrics |= {"test_loss": total, "test_ppl": math.exp(total)}
    return metrics, counts


@torch.no_grad()
def greedy_translate(model: Transformer, prompts: torch.Tensor, eos: int, max_new: int) -> list[list[int]]:
    """Greedy decoding for a batch of prompts that all have the SAME length.

    Same-length batches avoid padding, which matters because the model uses
    absolute positions and has no padding mask. No KV cache: each step re-runs
    the whole prefix, which is fine for a 17M model and short sentences.
    """
    out = prompts
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=prompts.device)
    for _ in range(max_new):
        if out.shape[1] >= model.config.N_ctx:
            break
        logits, _, _ = model(out, last_only=True)  # (B, vocab): only the next-token scores
        nxt = logits.argmax(-1)
        nxt = torch.where(finished, eos, nxt)
        out = torch.cat([out, nxt[:, None]], dim=1)
        finished |= nxt == eos
        if finished.all():
            break
    gens = []
    for row in out[:, prompts.shape[1]:].tolist():
        gens.append(row[: row.index(eos)] if eos in row else row)
    return gens


def bleu_eval(model, ds: TranslationDataset, tokenizer: Tokenizer, n_per_lang: int, device, amp_dtype,
              batch_size: int = 256, seed: int = 0) -> tuple[dict, list[dict]]:
    """BLEU on n_per_lang test sentences per language. The SAME English sentences are
    used for vi and ja, so the two language scores are directly comparable."""
    n_half = len(ds) // 2  # first half vi->en, second half ja->en (same en order)
    # holds only if build_examples dropped nothing (true at n_ctx=256); otherwise pairs misalign
    assert len(ds) % 2 == 0 and (ds.lang[:n_half] == 0).all() and (ds.lang[n_half:] == 1).all()
    picks = random.Random(seed).sample(range(n_half), min(n_per_lang, n_half))
    items = [i for i in picks] + [i + n_half for i in picks]
    eos = tokenizer.token_to_id("<eos>")

    by_len = defaultdict(list)  # group by prompt length -> no padding needed
    for i in items:
        by_len[int(ds.tgt_start[i])].append(i)

    hyps = {}
    for plen, idxs in by_len.items():
        for b in range(0, len(idxs), batch_size):
            chunk = idxs[b : b + batch_size]
            prompts = torch.stack([ds[i][0][:plen] for i in chunk]).long().to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                gens = greedy_translate(model, prompts, eos, max_new=min(160, model.config.N_ctx - plen))
            hyps.update(zip(chunk, gens))

    records, metrics = [], {}
    for l in (0, 1):
        refs, hs = [], []
        for i in (items[: len(picks)] if l == 0 else items[len(picks):]):
            seq, ts, _ = ds[i]
            ref = tokenizer.decode(seq[ts:-1].tolist())
            hyp = tokenizer.decode(hyps[i])
            refs.append(ref)
            hs.append(hyp)
            records.append({"lang": LANG_NAMES[l], "source": tokenizer.decode(seq[1 : ts - 1].tolist()),
                            "reference": ref, "hypothesis": hyp})
        metrics[f"bleu_{LANG_NAMES[l]}"] = sacrebleu.corpus_bleu(hs, [refs]).score
    all_h = [r["hypothesis"] for r in records]
    all_r = [r["reference"] for r in records]
    metrics["bleu"] = sacrebleu.corpus_bleu(all_h, [all_r]).score
    metrics["bleu_n_sentences"] = len(records)
    return metrics, records


def plot_heatmap(counts: torch.Tensor, title: str, path: Path, side: int = 0) -> None:
    """Rows = layers, columns = experts, one panel per source language.
    Cell = fraction of that language's routing slots (on `side`) sent to the expert."""
    frac = counts[:, :, side].float()
    frac = frac / frac.sum(-1, keepdim=True)  # (L, 2, E)
    L, _, E = frac.shape
    fig, axes = plt.subplots(1, 2, figsize=(2.2 + 1.1 * E * 2, 0.6 + 0.5 * L), sharey=True,
                             constrained_layout=True)
    for l, ax in enumerate(axes):
        im = ax.imshow(frac[:, l], cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(L):
            for e in range(E):
                v = frac[i, l, e].item()
                ax.text(e, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else "#1f2937")
        ax.set_title(f"{LANG_NAMES[l]} {SIDES[side]} tokens", fontsize=10)
        ax.set_xticks(range(E), [f"E{e}" for e in range(E)])
        ax.set_yticks(range(L), [f"layer {i}" for i in range(L)])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="fraction of routing slots")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="path to model.pt")
    p.add_argument("--data", type=Path, default=Path("data/part1"))
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--out", type=Path, default=None, help="default: results/part1/<variant>")
    p.add_argument("--bleu-n", type=int, default=1000, help="test sentences per language for BLEU")
    p.add_argument("--max-test-batches", type=int, default=None, help="limit (smoke tests)")
    args = p.parse_args()
    name = args.ckpt.parent.name
    out = args.out or Path("results/part1") / name
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 only on Ampere+ (A100, L4); T4 "supports" it via slow emulation, so use fp16 there
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    loader = make_loader(args.data / "test.pt", 256, shuffle=False, num_workers=0)
    if args.max_test_batches:
        loader = [b for _, b in zip(range(args.max_test_batches), loader)]
    metrics, counts = perplexity_and_routing(model, loader, device, amp_dtype)
    print(f"test ppl {metrics['test_ppl']:.2f} (vi {metrics['test_ppl_vi']:.2f}, ja {metrics['test_ppl_ja']:.2f})")

    bleu, records = bleu_eval(model, TranslationDataset(args.data / "test.pt"), tokenizer, args.bleu_n,
                              device, amp_dtype)
    metrics |= bleu
    print(f"BLEU {bleu['bleu']:.2f} (vi {bleu['bleu_vi']:.2f}, ja {bleu['bleu_ja']:.2f}) on {bleu['bleu_n_sentences']} sentences")

    (out / "metrics.json").write_text(json.dumps({"variant": name, **metrics}, indent=2))
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if counts is not None:
        (out / "expert_usage.json").write_text(json.dumps(
            {"dims": ["layer", "lang(vi,ja)", "side(source,target)", "expert"], "counts": counts.tolist()}))
        plot_heatmap(counts, f"{name}: expert routing by source language", out / "expert_heatmap.png", side=0)
        plot_heatmap(counts, f"{name}: expert routing (English target side)", out / "expert_heatmap_target.png", side=1)
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
