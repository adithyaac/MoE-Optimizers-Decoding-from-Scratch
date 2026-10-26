"""Decoder-only transformer for Part 1, with a dense MLP or a Mixture-of-Experts FFN.

Shape / parameter check for all five variants:
    uv run python -m src.part1.model
"""

from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class TransformerConfig:
    vocab_size: int = 16000
    N_ctx: int = 256  # max sequence length
    D_model: int = 384  # embedding / residual stream width
    n_layers: int = 6
    n_heads: int = 6  # D_model must be divisible by n_heads
    d_ff: int = 1536  # dense MLP hidden size (commonly 4 * D_model)
    dropout: float = 0.1
    use_moe: bool = False
    # MoE fields (ignored when use_moe is False)
    n_experts: int = 4  # routed experts
    top_k: int = 1  # routed experts active per token
    n_shared: int = 0  # always-active experts, not routed
    d_expert: int = 384  # hidden size of every expert (routed and shared)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransformerConfig":
        cfg = yaml.safe_load(Path(path).read_text())["model"]
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in known})


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.D_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.D_model, config.D_model)
        self.k_proj = nn.Linear(config.D_model, config.D_model)
        self.v_proj = nn.Linear(config.D_model, config.D_model)
        self.out_proj = nn.Linear(config.D_model, config.D_model)
        self.dropout_layer = nn.Dropout(config.dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_model) -> (B, T, D_model)
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(attn_output)
        output = self.dropout_layer(output)
        return output


class MLP(nn.Module):
    """Standard 2-layer FFN. Also used as a single expert inside MoE."""

    def __init__(self, config: TransformerConfig, d_hidden: int | None = None):
        super().__init__()
        d_hidden = d_hidden or config.d_ff
        self.ffn = nn.Sequential(
            nn.Linear(config.D_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, config.D_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoE(nn.Module):
    """Top-k routed experts, plus optional always-on shared experts.

    For each token x:   y = sum_{shared s} E_s(x) + sum_{e in topk(x)} g_e(x) * E_e(x)
    where g = softmax(router(x)) over the routed experts.

    Dropless: every token is processed by its k experts (no capacity limit).
    After forward, `aux_loss` holds the Switch-Transformer load-balancing loss and
    `last_topk` holds the chosen expert ids (N_tokens, k), for usage analysis.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_experts, self.top_k = config.n_experts, config.top_k
        assert 1 <= self.top_k <= self.n_experts
        self.router = nn.Linear(config.D_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config, config.d_expert) for _ in range(self.n_experts)])
        self.shared = nn.ModuleList([MLP(config, config.d_expert) for _ in range(config.n_shared)])
        self.aux_loss = torch.tensor(0.0)
        self.last_topk: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D) -> (B, T, D);  mask: (B, T) bool, True = real token
        B, T, D = x.shape
        flat = x.reshape(-1, D)  # (N, D), N = B*T

        probs = F.softmax(self.router(flat).float(), dim=-1)  # (N, E), fp32 for stability
        top_p, top_i = probs.topk(self.top_k, dim=-1)  # (N, k)
        if self.top_k > 1:
            # Renormalise so the k weights sum to 1. Not done for k=1: the weight
            # would always be 1 and the router would get no gradient from the LM loss.
            top_p = top_p / top_p.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)  # tokens routed to e
            if tok.numel() == 0:
                continue
            y = expert(flat[tok]) * top_p[tok, slot, None].to(flat.dtype)
            out.index_add_(0, tok, y)
        for expert in self.shared:
            out = out + expert(flat)

        # Load-balancing loss (Switch Transformer, eq. 4), over real tokens only:
        #   f_e = fraction of routing slots sent to e,  P_e = mean router prob of e
        #   loss = E * sum_e f_e * P_e   -> minimal (= 1) when both are uniform.
        m = mask.reshape(-1) if mask is not None else torch.ones(B * T, dtype=torch.bool, device=x.device)
        assign = F.one_hot(top_i[m], self.n_experts).float().sum(1)  # (N_real, E)
        f = assign.mean(0) / self.top_k
        P = probs[m].mean(0)
        self.aux_loss = self.n_experts * (f * P).sum()
        self.last_topk = top_i.detach()

        return out.reshape(B, T, D)


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.D_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.D_model)
        self.ffn = MoE(config) if config.use_moe else MLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        x = x + (self.ffn(h, mask) if isinstance(self.ffn, MoE) else self.ffn(h))
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.D_model)
        self.position_embedding = nn.Embedding(config.N_ctx, config.D_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.D_model)
        self.lm_head = nn.Linear(config.D_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def moe_layers(self) -> list[MoE]:
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, mask: torch.Tensor | None = None,
        reduction: str = "mean", last_only: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        # idx: (B, T) token ids, T <= N_ctx;  mask: (B, T) bool, True = real (non-pad) token
        # returns (logits, LM loss, MoE aux loss (0 for dense)):
        #   without targets: logits (B, T, vocab) — or (B, vocab) if last_only — and loss None
        #   with targets:    logits None, loss = mean over target tokens, or (B, T) per-token
        #                    losses (0 at ignored positions) if reduction="none"
        B, T = idx.shape

        assert T <= self.config.N_ctx, f"Cannot forward sequence of length {T}, block size is only {self.config.N_ctx}"
        z = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        z = self.dropout(z)
        for block in self.blocks:
            z = block(z, mask)
        z = self.final_norm(z)

        moe = self.moe_layers()
        aux = torch.stack([m.aux_loss for m in moe]).mean() if moe else torch.zeros((), device=idx.device)

        if targets is None:
            return self.lm_head(z[:, -1] if last_only else z), None, aux
        # Project only the positions that have a target: the (B, T, vocab) logits tensor is
        # by far the largest activation, and most positions are source or padding.
        valid = targets != -100
        tok_loss = F.cross_entropy(self.lm_head(z[valid]).float(), targets[valid], reduction="none")
        if reduction == "none":
            return None, torch.zeros(B, T, device=idx.device).masked_scatter(valid, tok_loss), aux
        return None, tok_loss.mean(), aux


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_active_params(model: Transformer) -> int:
    """Parameters used for one token: total minus the routed experts it skips."""
    inactive = sum(
        (m.n_experts - m.top_k) * count_params(m.experts[0]) for m in model.moe_layers()
    )
    return count_params(model) - inactive


if __name__ == "__main__":
    for path in sorted(Path("configs/part1").glob("*.yaml")):
        cfg = TransformerConfig.from_yaml(path)
        model = Transformer(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 32))
        logits, _, _ = model(idx)
        assert logits.shape == (2, 32, cfg.vocab_size), logits.shape
        _, loss, aux = model(idx, targets=idx)
        print(f"{path.stem:<22} total {count_params(model) / 1e6:6.2f}M  "
              f"active {count_active_params(model) / 1e6:6.2f}M  "
              f"loss {loss.item():.3f}  aux {aux.item():.3f}")
