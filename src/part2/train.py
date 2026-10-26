"""Part 2: pretrain the dense Part 1 model on the human-AI parallel corpus with one optimizer.

    uv run python -m src.part2.train --optimizer muon

Trains for 1x the training split (one epoch). At every 0.1x of the dataset it logs
  - validation loss / perplexity (whole validation split)
  - test BLEU: greedy-continue 64-token prompts from the test split by 32 tokens and
    compare with the true next 32 tokens
to WandB and to <out>/history.json (used by src.part2.plots). Resumable like Part 1.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sacrebleu  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from src.part1.evaluate import greedy_translate  # noqa: E402  (generic same-length greedy decoding)
from src.part1.model import Transformer, TransformerConfig, count_params  # noqa: E402
from src.part1.train import lr_at  # noqa: E402  (warmup + cosine, same schedule as Part 1)
from src.part2.data import windows  # noqa: E402
from src.part2.optimizers import DEFAULTS, make_optimizer  # noqa: E402


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True, choices=list(DEFAULTS))
    p.add_argument("--config", type=Path, default=Path("configs/part1/1_dense.yaml"), help="model (dense)")
    p.add_argument("--data", type=Path, default=Path("data/part2"))
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--out", type=Path, default=None, help="default: checkpoints/part2/<optimizer>")
    p.add_argument("--run-name", default=None, help="default: optimizer name (set for sweeps)")
    p.add_argument("--lr", type=float, default=None, help="default: per-optimizer value in optimizers.DEFAULTS")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--data-frac", type=float, default=1.0, help="fraction of one epoch to train (sweeps: 0.1)")
    p.add_argument("--batch-size", type=int, default=64, help="windows of N_ctx tokens per step")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=float, default=0.1, help="fraction of the dataset")
    p.add_argument("--bleu-n", type=int, default=256, help="test prompts for BLEU")
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="anlp-a2-part2")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()
    args.run_name = args.run_name or args.optimizer
    args.out = args.out or Path("checkpoints/part2") / args.run_name
    return args


@torch.no_grad()
def val_loss(model, val_windows, device, amp_dtype, batch_size, max_batches=None) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, b in enumerate(range(0, len(val_windows), batch_size)):
        if max_batches is not None and i >= max_batches:
            break
        w = val_windows[b : b + batch_size].long().to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            _, loss, _ = model(w[:, :-1], w[:, 1:])
        total += loss.item() * w[:, 1:].numel()
        n += w[:, 1:].numel()
    model.train()
    return total / n


@torch.no_grad()
def test_bleu(model, prompts, refs, tokenizer, device, amp_dtype, eos: int) -> float:
    model.eval()
    hyps = []
    for b in range(0, len(prompts), 128):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            gens = greedy_translate(model, prompts[b : b + 128].to(device), eos, max_new=refs.shape[1])
        hyps += [tokenizer.decode(g) for g in gens]
    model.train()
    return sacrebleu.corpus_bleu(hyps, [[tokenizer.decode(r.tolist()) for r in refs]]).score


def main() -> None:
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = TransformerConfig.from_yaml(args.config)
    assert not cfg.use_moe, "Part 2 uses the dense model"
    model = Transformer(cfg).to(device)
    opt = make_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and not use_bf16)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos = tokenizer.token_to_id("<eos>")

    T = cfg.N_ctx
    train_w = windows(torch.load(args.data / "train.pt"), T)  # (n, T+1)
    val_w = windows(torch.load(args.data / "validation.pt"), T)
    test_w = windows(torch.load(args.data / "test.pt"), T)
    pick = random.Random(args.seed).sample(range(len(test_w)), min(args.bleu_n, len(test_w)))
    bleu_prompts, bleu_refs = test_w[pick, :64].long(), test_w[pick, 64:96].long()  # 64-token prompt, 32 to predict

    dataset_tokens = len(train_w) * T  # 1x dataset
    budget = int(args.data_frac * dataset_tokens)
    order = torch.randperm(len(train_w), generator=torch.Generator().manual_seed(args.seed))

    # ---- resume
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "ckpt.pt"
    state = {"tokens": 0, "step": 0, "pos": 0, "next_eval": args.eval_every, "wandb_id": None, "history": []}
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        print(f"resumed from {ckpt_path} at {state['tokens'] / 1e6:.1f}M tokens")

    run = wandb.init(
        project=args.wandb_project, name=args.run_name, id=state["wandb_id"], resume="allow", mode=args.wandb_mode,
        tags=["part2", args.optimizer] + (["sweep"] if args.data_frac < 1 else []),
        config={**vars(cfg), **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "lr_used": base_lrs[0], "params": count_params(model), "dataset_tokens": dataset_tokens},
    )
    state["wandb_id"] = run.id
    if run.url:
        (args.out / "wandb_url.txt").write_text(run.url)
    print(f"{args.run_name}: {count_params(model) / 1e6:.2f}M params, dataset {dataset_tokens / 1e6:.1f}M tokens, "
          f"budget {budget / 1e6:.1f}M, lr {base_lrs[0]:g}, device {device}")

    def save_ckpt():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "state": state}, ckpt_path)

    def evaluate():
        vl = val_loss(model, val_w, device, amp_dtype, 128, args.max_val_batches)
        bleu = test_bleu(model, bleu_prompts, bleu_refs, tokenizer, device, amp_dtype, eos)
        rec = {"tokens": state["tokens"], "dataset_frac": state["tokens"] / dataset_tokens,
               "val_loss": vl, "val_ppl": math.exp(vl), "test_bleu": bleu}
        state["history"].append(rec)
        wandb.log({"val/loss": vl, "val/ppl": rec["val_ppl"], "test/bleu": bleu, "tokens": state["tokens"],
                   "dataset_frac": rec["dataset_frac"]}, step=state["step"])
        print(f"  eval @ {rec['dataset_frac']:.2f}x dataset: val loss {vl:.4f} (ppl {rec['val_ppl']:.2f}), "
              f"test BLEU {bleu:.2f}")

    # ---- train
    model.train()
    t0, tok0 = time.time(), state["tokens"]
    while state["tokens"] < budget:
        idx = order[state["pos"] : state["pos"] + args.batch_size]
        w = train_w[idx].long().to(device, non_blocking=True)
        x, y = w[:, :-1], w[:, 1:]
        frac = state["tokens"] / budget
        mult = _schedule(frac, args)  # same warmup + cosine shape for every optimizer
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * mult

        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        state["tokens"] += x.numel()
        state["step"] += 1
        state["pos"] += args.batch_size

        if state["step"] % args.log_every == 0:
            tps = (state["tokens"] - tok0) / (time.time() - t0)
            wandb.log({"train/loss": loss.item(), "train/lr": opt.param_groups[0]["lr"],
                       "train/grad_norm": grad_norm.item(), "train/tokens_per_sec": tps,
                       "tokens": state["tokens"], "dataset_frac": state["tokens"] / dataset_tokens},
                      step=state["step"])
            print(f"step {state['step']:6d} | {state['tokens'] / 1e6:6.2f}M tok | loss {loss.item():.4f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {tps / 1e3:.0f}k tok/s")

        done = state["tokens"] >= budget or state["pos"] + args.batch_size > len(order)
        if done or state["tokens"] / dataset_tokens >= state["next_eval"] - 1e-9:
            evaluate()
            while state["next_eval"] <= state["tokens"] / dataset_tokens + 1e-9:
                state["next_eval"] += args.eval_every
            save_ckpt()
        if done:
            break

    (args.out / "history.json").write_text(json.dumps(
        {"optimizer": args.optimizer, "run_name": args.run_name, "lr": base_lrs[0],
         "dataset_tokens": dataset_tokens, "history": state["history"]}, indent=2))
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, args.out / "model.pt")
    print(f"saved final weights and history to {args.out}")
    wandb.finish()


def _schedule(frac: float, args) -> float:
    """Multiplier in [0, 1]: the Part 1 warmup + cosine schedule with peak 1 (lr_at with lr=1)."""
    return lr_at(frac, argparse.Namespace(lr=1.0, warmup=args.warmup, min_lr_ratio=args.min_lr_ratio))


if __name__ == "__main__":
    main()
