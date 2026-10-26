# Advanced NLP — Assignment 2 (Monsoon 2026)

Mixture of Experts, optimizers and decoding strategies, in PyTorch.

| | Checkpoints (HuggingFace) | Training logs (WandB) |
|---|---|---|
| Part 1 — MoE | [anlp-a2-part1-1_dense](https://huggingface.co/<hf-user>/anlp-a2-part1-1_dense), [-2_moe_4e_top1](https://huggingface.co/<hf-user>/anlp-a2-part1-2_moe_4e_top1), [-3_moe_4e_top2](https://huggingface.co/<hf-user>/anlp-a2-part1-3_moe_4e_top2), [-4_moe_shared1_3e_top1](https://huggingface.co/<hf-user>/anlp-a2-part1-4_moe_shared1_3e_top1), [-5_moe_4e_top2_active](https://huggingface.co/<hf-user>/anlp-a2-part1-5_moe_4e_top2_active) | [anlp-a2-part1](https://wandb.ai/<wandb-entity>/anlp-a2-part1) |
| Part 2 — Optimizers | [anlp-a2-part2-adamw](https://huggingface.co/<hf-user>/anlp-a2-part2-adamw), [-mars](https://huggingface.co/<hf-user>/anlp-a2-part2-mars), [-lion](https://huggingface.co/<hf-user>/anlp-a2-part2-lion), [-muon](https://huggingface.co/<hf-user>/anlp-a2-part2-muon) | [anlp-a2-part2](https://wandb.ai/<wandb-entity>/anlp-a2-part2) |
| Part 3 — Decoding | pretrained [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m) (no training) | — |

Each HuggingFace repo holds `model.pt` (`{"model": state_dict, "config": dict}`), its
evaluation results and a model card linking the WandB run. The Part 1 WandB project also
contains one crashed `1_dense` run (CUDA OOM before a memory fix); the finished run with the
same name is the one reported.

## Layout

```
main.py                    single entry point: `python main.py <command> [args]` (see below)
configs/part1/             one YAML per FFN variant (only what differs from the dense defaults)
assets/tokenizer.json      shared vi/ja/en byte-level BPE tokenizer (16k)
notebooks/colab.ipynb      the Colab runner used for all GPU runs
src/
  utils.py                 dataset loading (parquet via huggingface_hub), network fixes
  hf_upload.py             upload a checkpoint + results + model card to HuggingFace
  part1/                   tokenizer, data, model (dense + MoE), train, evaluate, analyze_routing, plots
  part2/                   data, optimizers (AdamW, MARS, Lion, Muon), train, plots
  part3/                   decoding (greedy, top-k, top-p, beam search), run
results/                   small result files: metrics, plots, routing analysis, raw generations
```

## Setup and running

```bash
uv sync                                      # Python 3.14 locally; Colab's Python also works
uv run python main.py --help
```

All GPU runs were done on Google Colab (T4) with `notebooks/colab.ipynb`; the commands are:

```bash
# Part 1
uv run python main.py tokenizer --vocab-size 16000            # already trained: assets/tokenizer.json
uv run python main.py part1-data --out data/part1
uv run python main.py part1-train --config configs/part1/2_moe_4e_top1.yaml
uv run python main.py part1-eval --ckpt checkpoints/part1/2_moe_4e_top1/model.pt
uv run python main.py part1-routing && uv run python main.py part1-plots --wandb-path <entity>/anlp-a2-part1

# Part 2
uv run python main.py part2-data --out data/part2
uv run python main.py part2-train --optimizer muon            # adamw | mars | lion | muon
uv run python main.py part2-plots

# Part 3
uv run python main.py part3 --n 1000
```

Training is resumable: rerunning the same command continues from `<out>/ckpt.pt`.

## Part 1 — Mixture of Experts

Decoder-only transformer (6 layers, d=384, 6 heads, N_ctx=256, tied embeddings) trained to
translate vi→en and ja→en as a conditional LM: `<vi|ja> source <sep> english <eos>`, loss on
the English tokens only. The FFN is a dense MLP or an MoE layer (top-k softmax router,
dropless dispatch, optional shared experts, Switch-Transformer load-balancing loss, weight 0.01).

| # | FFN | total params | active params |
|---|---|---|---|
| 1 | dense MLP, d_ff 1536 | 16.89M | 16.89M |
| 2 | 4 experts × 384, top-1 | 16.91M | 11.58M |
| 3 | 4 experts × 384, top-2 | 16.91M | 13.36M |
| 4 | 1 shared + 3 routed × 384, top-1 | 16.90M | 13.36M |
| 5 | 4 experts × 768, top-2 (active-matched) | 23.99M | 16.90M |

Every variant: 40M training tokens (~1 epoch), same data order, AdamW (lr 1e-3, β=(0.9,0.95),
wd 0.1), batch 128 sentences, 2% warmup + cosine. Test perplexity on the full test set (49.6k
sentences); BLEU (sacrebleu) on 1000 sentences per language with greedy decoding.

| variant | test ppl | ppl vi | ppl ja | BLEU | BLEU vi | BLEU ja |
|---|---|---|---|---|---|---|
| dense | 9.71 | 7.86 | 11.99 | 27.21 | 32.23 | 21.96 |
| 4e top-1 | 9.05 | 7.43 | 11.02 | 28.49 | 33.13 | 23.58 |
| 4e top-2 | 8.98 | 7.36 | 10.95 | 28.61 | 33.24 | 23.44 |
| 1 shared + 3e top-1 | 9.14 | 7.45 | 11.22 | 28.26 | 33.12 | 23.31 |
| 4e top-2 (active-matched) | **8.87** | **7.21** | **10.92** | 28.50 | 33.00 | **23.84** |

Plots: `results/part1/plots/` (validation curves, test metrics, expert load during training),
per-language expert heatmaps on the HF repos (`results/expert_heatmap*.png`), routing by token
type in `results/part1/routing/`.

## Part 2 — Optimizers

The dense model of Part 1, trained from scratch on
[browndw/human-ai-parallel-corpus](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus)
(next-token prediction; 90/5/5 split by source document so a human text and its LLM
continuations stay in one split; 48.4M training tokens = 1× dataset). One optimizer per
category of Wen et al. (2025), Table 1, implemented from scratch in `src/part2/optimizers.py`
(`torch.optim.Optimizer` used only as a base class):

| category | optimizer | defaults |
|---|---|---|
| AdamW | AdamW | lr 1e-3, β (0.9, 0.95), wd 0.1 |
| variance-reduced AdamW | MARS (approx.) | lr 1e-3, β (0.95, 0.99), γ 0.025, wd 0.1; AdamW for 1-D params |
| memory-efficient | Lion | lr 1e-4, β (0.9, 0.99), wd 1.0 |
| matrix-based | Muon | lr 1e-3, momentum 0.95 (Nesterov), 5 Newton–Schulz steps, update scaled 0.2·√max(m,n); AdamW for embeddings and 1-D params |

Validation loss (whole validation split) and test BLEU (256 test windows: 64-token prompt,
32-token greedy continuation vs the true continuation) are recorded every 0.1× dataset tokens.
Results: `results/part2/` (`curves.png`, `summary.md`, per-optimizer `*_history.json`).

| optimizer | peak lr | final val loss | final val ppl | final test BLEU |
|---|---|---|---|---|
| AdamW | 1e-3 | 3.606 | 36.82 | 3.26 |
| MARS | 1e-3 | 3.732 | 41.74 | 2.80 |
| Lion | 1e-4 | 4.490 | 89.10 | 1.55 |
| **Muon** | 1e-3 | **3.304** | **27.22** | **3.66** |

## Part 3 — Decoding strategies

Pythia-160M, decoding implemented on `model.forward()` with a KV cache (no `generate()`),
on 1000 ROCStories test stories: given the first sentence, generate up to 64 tokens
(newline tokens banned, since a story is one paragraph). Generation perplexity is measured
under Pythia-160M and under Pythia-410M; "accuracy" is BLEU / ROUGE-L against the true
continuation. Raw generations: `results/part3/<config>/generations.jsonl`.

| config | time (s) | ppl (self) | ppl (Pythia-410M) | BLEU | ROUGE-L | distinct-2 | rep. trigrams |
|---|---|---|---|---|---|---|---|
| reference (human) | - | 24.89 | 16.96 | - | - | 0.544 | 0.008 |
| greedy | 23.4 | 2.19 | 2.32 | 0.54 | 0.119 | 0.076 | 0.705 |
| beam1 | 26.5 | 2.19 | 2.32 | 0.54 | 0.119 | 0.076 | 0.705 |
| beam2 | 41.8 | 2.04 | 2.13 | 0.58 | 0.115 | 0.076 | 0.720 |
| beam4 | 76.4 | 1.90 | 1.96 | 0.52 | 0.104 | 0.073 | 0.736 |
| topk10 | 24.8 | 8.58 | 10.54 | 0.56 | 0.128 | 0.330 | 0.045 |
| topk50 | 24.7 | 15.25 | 19.67 | 0.43 | 0.118 | 0.444 | 0.014 |
| topk50_t0.7 | 24.6 | 7.76 | 9.46 | 0.79 | 0.131 | 0.331 | 0.081 |
| topp0.9 | 28.0 | 27.36 | 38.09 | 0.18 | 0.112 | 0.582 | 0.007 |
| topp0.95 | 28.2 | 39.33 | 57.56 | 0.24 | 0.107 | 0.636 | 0.006 |
| topp0.9_t0.7 | 28.9 | 6.35 | 7.65 | 0.79 | 0.131 | 0.300 | 0.140 |

Beam search time grows linearly with width (≈ 9.4 s + 16.6 s × width on a T4); see
`results/part3/beam_time.png`.
