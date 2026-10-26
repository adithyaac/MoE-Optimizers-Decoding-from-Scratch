"""Optimizers implemented from scratch (torch.optim.Optimizer is used only as a base class).

One per category of Wen et al. (2025), "Fantastic Pretraining Optimizers", Table 1:
  AdamW  - baseline                      (Loshchilov & Hutter, 2019)
  MARS   - variance-reduced AdamW        (Yuan et al., 2025)
  Lion   - memory-efficient              (Chen et al., 2023)
  Muon   - matrix-based                  (Jordan et al., 2024)

Weight decay is decoupled everywhere: p <- p - lr * wd * p, applied before the update.
Each param group may set its own hyperparameters; `make_optimizer` builds the groups.
"""

import math

import torch
from torch.optim import Optimizer


def adamw_update(p, g, state, lr, beta1, beta2, eps, wd):
    """One AdamW step for a single tensor (shared by AdamW and by the 1-D fallbacks)."""
    if "step" not in state:  # (not `if not state`: MARS stores prev_grad in the same dict)
        state["step"] = 0
        state["m"] = torch.zeros_like(p)
        state["v"] = torch.zeros_like(p)
    state["step"] += 1
    t, m, v = state["step"], state["m"], state["v"]
    m.lerp_(g, 1 - beta1)  # m = b1 m + (1 - b1) g          first moment (momentum)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)  # v = b2 v + (1 - b2) g^2   second moment
    m_hat_scale = 1 / (1 - beta1**t)  # bias correction: moments start at 0
    denom = (v / (1 - beta2**t)).sqrt_().add_(eps)
    p.mul_(1 - lr * wd)
    p.addcdiv_(m, denom, value=-lr * m_hat_scale)


class AdamW(Optimizer):
    """m_t = b1 m + (1-b1) g;  v_t = b2 v + (1-b2) g^2;  p -= lr * m_hat / (sqrt(v_hat) + eps)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is not None:
                    adamw_update(p, p.grad, self.state[p], group["lr"], b1, b2, group["eps"], group["weight_decay"])


class MARS(Optimizer):
    """MARS-AdamW, approximate version (Yuan et al., 2025).

    Variance reduction: instead of the raw gradient, Adam is fed a corrected gradient
        c_t = g_t + gamma * (b1 / (1 - b1)) * (g_t - g_{t-1})
    which cancels part of the minibatch noise (STORM-style). g_{t-1} is the previous
    step's gradient. c_t is rescaled to norm <= 1 per tensor. As in the reference
    implementation, MARS is applied to matrices and plain AdamW to 1-D params
    (biases, LayerNorm), via the `mars` flag on each group.
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.0,
                 betas_1d=(0.9, 0.95)):
        super().__init__(params, dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay,
                                      betas_1d=betas_1d, mars=True))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if not group["mars"]:
                    adamw_update(p, g, st, group["lr"], *group["betas_1d"], group["eps"], group["weight_decay"])
                    continue
                b1, b2 = group["betas"]
                prev = st.get("prev_grad")
                c = g if prev is None else g + group["gamma"] * (b1 / (1 - b1)) * (g - prev)
                norm = c.norm()
                if norm > 1:
                    c = c / norm
                st["prev_grad"] = g.clone()
                adamw_update(p, c, st, group["lr"], b1, b2, group["eps"], group["weight_decay"])


class Lion(Optimizer):
    """Lion (Chen et al., 2023): only ONE state tensor (momentum) instead of Adam's two.

        update = sign(b1 * m + (1 - b1) * g)      every coordinate moves by exactly +-lr
        m      = b2 * m + (1 - b2) * g
    Because |update| = 1 everywhere (larger than Adam's typical step), Lion needs a
    ~3-10x smaller lr and a correspondingly larger weight decay.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                update = (m * b1 + g * (1 - b1)).sign_()
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
                m.lerp_(g, 1 - b2)


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately orthogonalise G (replace its singular values by ~1): returns ~U V^T
    for G = U S V^T. Quintic Newton-Schulz iteration with the coefficients from Jordan
    et al. (2024), tuned to push singular values into ~[0.7, 1.2] in 5 steps."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:  # iterate on the wide orientation: X X^T is the smaller matrix
        X = X.T
    X = X / (X.norm() + 1e-7)  # spectral norm <= Frobenius norm <= 1, needed for convergence
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(Optimizer):
    """Muon (Jordan et al., 2024): momentum, then orthogonalise the update matrix.

    For each 2-D weight in the transformer blocks:
        buf = mu * buf + g;  u = g + mu * buf (Nesterov)
        O   = NewtonSchulz(u)                  ~ U V^T: same "directions", all singular values 1
        p  -= lr * 0.2 * sqrt(max(rows, cols)) * O
    The 0.2 * sqrt(max dim) factor (Liu et al., 2025, "Muon is scalable") matches the
    update RMS of AdamW, so the same lr / weight decay work for both. Embeddings and
    1-D params are not "hidden matrices" and use AdamW (`muon` flag on each group).
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0,
                 betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay, betas=betas, eps=eps, muon=True))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if not group["muon"]:
                    adamw_update(p, g, st, group["lr"], *group["betas"], group["eps"], group["weight_decay"])
                    continue
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(group["momentum"]).add_(g)
                u = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                O = newton_schulz(u, group["ns_steps"]).to(p.dtype)
                scale = 0.2 * math.sqrt(max(p.shape))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(O, alpha=-group["lr"] * scale)


# Per-optimizer defaults for this model (~17M params, batch 64 x 256 tokens).
DEFAULTS = {
    "adamw": dict(lr=1e-3, weight_decay=0.1),
    "mars": dict(lr=1e-3, weight_decay=0.1),
    "lion": dict(lr=1e-4, weight_decay=1.0),  # lr / 10, wd * 10 relative to AdamW (Lion paper)
    "muon": dict(lr=1e-3, weight_decay=0.1),  # RMS-matched to AdamW, so same lr
}


def make_optimizer(name: str, model: torch.nn.Module, lr: float | None = None, weight_decay: float | None = None):
    """Build the optimizer with two kinds of params:
      matrices (dim >= 2): weight decay; MARS / Muon special update (Muon: block weights only)
      vectors  (dim <  2): biases and LayerNorm, no weight decay, AdamW fallback for MARS / Muon
    """
    hp = dict(DEFAULTS[name])
    if lr is not None:
        hp["lr"] = lr
    if weight_decay is not None:
        hp["weight_decay"] = weight_decay
    named = list(model.named_parameters())  # tied lm_head appears once (as token_embedding)
    vectors = [p for _, p in named if p.dim() < 2]
    if name == "muon":
        hidden = [p for n, p in named if p.dim() >= 2 and n.startswith("blocks.")]
        embeds = [p for n, p in named if p.dim() >= 2 and not n.startswith("blocks.")]
        return Muon([{"params": hidden},
                     {"params": embeds, "muon": False},
                     {"params": vectors, "muon": False, "weight_decay": 0.0}], **hp)
    matrices = [p for _, p in named if p.dim() >= 2]
    if name == "mars":
        return MARS([{"params": matrices}, {"params": vectors, "mars": False, "weight_decay": 0.0}], **hp)
    cls = {"adamw": AdamW, "lion": Lion}[name]
    return cls([{"params": matrices}, {"params": vectors, "weight_decay": 0.0}], **hp)
