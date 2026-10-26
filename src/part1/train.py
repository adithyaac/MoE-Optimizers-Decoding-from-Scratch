"""Train one Part 1 variant on a fixed token budget.

    uv run python -m src.part1.train --config configs/part1/2_moe_4e_top1.yaml

All variants share the training defaults below, so only the FFN differs.
Resumable: rerunning the same command continues from <out>/ckpt.pt (Colab disconnects).
"""

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation

import torch  # noqa: E402
import wandb  # noqa: E402

from src.part1.data import PAD_ID, make_loader
from src.part1.model import Transformer, TransformerConfig, count_active_params, count_params


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/part1"))
    p.add_argument("--out", type=Path, default=None, help="default: checkpoints/part1/<config name>")
    p.add_argument("--tokens", type=float, default=40e6, help="training budget (non-pad input tokens)")
    p.add_argument("--batch-size", type=int, default=128, help="sequences per batch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=float, default=0.02, help="fraction of budget")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--aux-coef", type=float, default=0.01, help="MoE load-balancing loss weight")
    p.add_argument("--eval-every", type=float, default=0.1, help="fraction of budget between evals")
    p.add_argument("--max-val-batches", type=int, default=None, help="limit eval (smoke tests)")
    p.add_argument("--log-every", type=int, default=50, help="steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--wandb-project", default="anlp-a2-part1")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()
    args.out = args.out or Path("checkpoints/part1") / args.config.stem
    return args


def lr_at(frac: float, args) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * lr. frac = tokens_seen / budget."""
    if frac < args.warmup:
        return args.lr * frac / args.warmup
    t = min(1.0, (frac - args.warmup) / (1 - args.warmup))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def make_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    # Weight decay only on matrices (Linear / Embedding weights), not on biases or LayerNorm.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model: Transformer, loader, device, amp_dtype, max_batches=None) -> dict:
    """Token-weighted mean loss over English target tokens -> perplexity."""
    model.eval()
    loss_sum, n_tok = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = batch["input_ids"].to(device), batch["targets"].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            _, loss, _ = model(x, y, mask=x != PAD_ID)
        n = int((y != -100).sum())
        loss_sum += loss.item() * n
        n_tok += n
    model.train()
    val_loss = loss_sum / n_tok
    return {"val/loss": val_loss, "val/ppl": math.exp(val_loss)}


def expert_load(model: Transformer, mask: torch.Tensor) -> dict:
    """Fraction of routing slots each expert got in the last batch, per layer."""
    out = {}
    m = mask.reshape(-1)
    for li, moe in enumerate(model.moe_layers()):
        counts = torch.bincount(moe.last_topk[m].reshape(-1), minlength=moe.n_experts).float()
        for e, c in enumerate((counts / counts.sum()).tolist()):
            out[f"load/layer{li}_expert{e}"] = c
    return out


def main() -> None:
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 on Ampere+ (A100, L4); fp16 + loss scaling on older GPUs (T4). Not is_bf16_supported():
    # it returns True on a T4, where bf16 is emulated and slow.
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = TransformerConfig.from_yaml(args.config)
    model = Transformer(cfg).to(device)
    opt = make_optimizer(model, args)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and not use_bf16)

    train_loader = make_loader(args.data / "train.pt", args.batch_size, shuffle=True,
                               seed=args.seed, num_workers=args.num_workers)
    val_loader = make_loader(args.data / "validation.pt", 256, shuffle=False, num_workers=args.num_workers)
    sampler = train_loader.batch_sampler

    # ---- resume
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "ckpt.pt"
    state = {"tokens": 0, "step": 0, "epoch": 0, "batch_in_epoch": 0, "next_eval": args.eval_every, "wandb_id": None}
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        for _ in range(state["epoch"]):  # replay the sampler RNG so the data order matches
            for _ in iter(sampler):
                pass
        sampler.skip = state["batch_in_epoch"]
        print(f"resumed from {ckpt_path} at {state['tokens'] / 1e6:.1f}M tokens")

    n_total, n_active = count_params(model), count_active_params(model)
    run = wandb.init(
        project=args.wandb_project, name=args.config.stem, id=state["wandb_id"], resume="allow",
        mode=args.wandb_mode, tags=["part1", "moe" if cfg.use_moe else "dense"],
        config={**vars(cfg), **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "params_total": n_total, "params_active": n_active},
    )
    state["wandb_id"] = run.id
    if run.url:
        (args.out / "wandb_url.txt").write_text(run.url)
    print(f"{args.config.stem}: {n_total / 1e6:.2f}M total, {n_active / 1e6:.2f}M active params, "
          f"device {device}, amp {amp_dtype if device.type == 'cuda' else 'off'}")

    def save_ckpt():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "state": state}, ckpt_path)

    # ---- train
    model.train()
    t0, tok0 = time.time(), state["tokens"]
    done = state["tokens"] >= args.tokens  # rerunning a finished run must not train further
    while not done:
        for batch in train_loader:
            n_real = int((batch["input_ids"] != PAD_ID).sum())  # on CPU: no GPU sync per step
            x, y = batch["input_ids"].to(device, non_blocking=True), batch["targets"].to(device, non_blocking=True)
            mask = x != PAD_ID
            frac = state["tokens"] / args.tokens
            for g in opt.param_groups:
                g["lr"] = lr_at(frac, args)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, lm_loss, aux = model(x, y, mask=mask)
                loss = lm_loss + args.aux_coef * aux
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # so clipping sees the true gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            state["tokens"] += n_real
            state["step"] += 1
            state["batch_in_epoch"] += 1

            if state["step"] % args.log_every == 0:
                tps = (state["tokens"] - tok0) / (time.time() - t0)
                log = {"train/loss": lm_loss.item(), "train/aux_loss": aux.item(), "train/lr": opt.param_groups[0]["lr"],
                       "train/grad_norm": grad_norm.item(), "train/tokens_per_sec": tps, "tokens": state["tokens"]}
                if cfg.use_moe:
                    log |= expert_load(model, mask)
                wandb.log(log, step=state["step"])
                print(f"step {state['step']:6d} | {state['tokens'] / 1e6:6.2f}M tok | loss {lm_loss.item():.4f} "
                      f"| aux {aux.item():.3f} | lr {log['train/lr']:.2e} | {tps / 1e3:.0f}k tok/s")

            done = state["tokens"] >= args.tokens
            if done or frac >= state["next_eval"]:
                metrics = evaluate(model, val_loader, device, amp_dtype, args.max_val_batches)
                wandb.log({**metrics, "tokens": state["tokens"]}, step=state["step"])
                print(f"  eval @ {state['tokens'] / 1e6:.1f}M tok: val loss {metrics['val/loss']:.4f}, ppl {metrics['val/ppl']:.2f}")
                while state["next_eval"] <= state["tokens"] / args.tokens:
                    state["next_eval"] += args.eval_every
                save_ckpt()
            if done:
                break
        else:
            state["epoch"] += 1
            state["batch_in_epoch"] = 0

    torch.save({"model": model.state_dict(), "config": vars(cfg)}, args.out / "model.pt")
    print(f"saved final weights to {args.out / 'model.pt'}")
    wandb.finish()


if __name__ == "__main__":
    main()
