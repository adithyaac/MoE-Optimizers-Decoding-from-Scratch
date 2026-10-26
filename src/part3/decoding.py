"""Decoding strategies implemented on top of model.forward() with a KV cache.

model.generate() is never used. Every function takes a batch of prompts that all
have the SAME length (so no padding / attention mask is needed) and returns the
generated token ids per prompt, cut before the first stop token. `banned` token ids
(e.g. newlines) get probability 0 at every step.

The KV cache: after the prompt is processed once ("prefill"), each step feeds only
the newest token and reuses the cached keys/values of all earlier positions, so a
step costs O(1) new tokens instead of re-running the whole prefix.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def _ban(logits: torch.Tensor, banned: torch.Tensor | None) -> torch.Tensor:
    if banned is not None:
        logits[:, banned] = float("-inf")  # these tokens can never be generated
    return logits


@torch.no_grad()
def prefill(model, prompts: torch.Tensor, banned: torch.Tensor | None = None):
    """Run the prompt once. Returns next-token logits (B, V) and the KV cache."""
    out = model(input_ids=prompts, use_cache=True)
    return _ban(out.logits[:, -1].float(), banned), out.past_key_values


@torch.no_grad()
def step(model, tokens: torch.Tensor, cache, banned: torch.Tensor | None = None):
    """Feed one new token per sequence (B,), reusing the cache."""
    out = model(input_ids=tokens[:, None], past_key_values=cache, use_cache=True)
    return _ban(out.logits[:, -1].float(), banned), out.past_key_values


def trim(rows: list[list[int]], stop_ids: set[int]) -> list[list[int]]:
    """Cut each sequence before its first stop token."""
    out = []
    for row in rows:
        cut = next((i for i, t in enumerate(row) if t in stop_ids), len(row))
        out.append(row[:cut])
    return out


# ------------------------------------------------- token pickers (one step)


def pick_greedy(logits: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    return logits.argmax(-1)


def pick_top_k(k: int, temperature: float = 1.0) -> Callable:
    """Sample from the k most likely tokens (renormalised)."""
    def pick(logits: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        top_vals, top_idx = (logits / temperature).topk(k, dim=-1)  # exactly k, even with ties
        choice = torch.multinomial(F.softmax(top_vals, -1), 1, generator=gen)  # index into the k
        return top_idx.gather(-1, choice).squeeze(1)
    return pick


def pick_top_p(p: float, temperature: float = 1.0) -> Callable:
    """Nucleus sampling: sample from the smallest set of tokens whose probability mass >= p."""
    def pick(logits: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        logits = logits / temperature
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = F.softmax(sorted_logits, -1)
        # drop a token if the mass of the tokens strictly before it already reaches p
        # (so the most likely token is always kept)
        drop = probs.cumsum(-1) - probs >= p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        choice = torch.multinomial(F.softmax(sorted_logits, -1), 1, generator=gen)  # index into sorted
        return order.gather(-1, choice).squeeze(1)
    return pick


# ------------------------------------------------------ token-by-token loop


@torch.no_grad()
def sample_loop(model, prompts: torch.Tensor, max_new: int, stop_ids: set[int], pick: Callable,
                gen: torch.Generator | None = None, banned: torch.Tensor | None = None) -> list[list[int]]:
    """Greedy / top-k / top-p: pick one token per step until every row has stopped."""
    stop = torch.tensor(sorted(stop_ids), device=prompts.device)
    logits, cache = prefill(model, prompts, banned)
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=prompts.device)
    out = []
    for t in range(max_new):
        nxt = pick(logits, gen)
        nxt = torch.where(finished, stop[0], nxt)  # rows that are done just repeat a stop token
        out.append(nxt)
        finished |= torch.isin(nxt, stop)
        if finished.all() or t == max_new - 1:
            break
        logits, cache = step(model, nxt, cache, banned)
    return trim(torch.stack(out, 1).tolist(), stop_ids)


# ------------------------------------------------------------- beam search


@torch.no_grad()
def beam_search(model, prompts: torch.Tensor, max_new: int, stop_ids: set[int], width: int,
                length_penalty: float = 1.0, banned: torch.Tensor | None = None) -> list[list[int]]:
    """Keep the `width` highest log-probability partial sequences per prompt.

    Each step expands every beam by every vocabulary token (width * V candidates per
    prompt), keeps the top `width`, and reorders the KV cache so row i holds the
    history of the beam it was extended from. Finished beams stay in the pool and
    can only be extended by a stop token at zero cost, so their score is frozen.
    The final answer is the beam with the best score / length**length_penalty.
    width=1 is exactly greedy decoding.
    """
    B, dev = len(prompts), prompts.device
    stop = torch.tensor(sorted(stop_ids), device=dev)
    pad = int(stop[0])

    logits, cache = prefill(model, prompts, banned)
    V = logits.shape[-1]
    scores, tok = F.log_softmax(logits, -1).topk(width, dim=-1)  # (B, w): first tokens
    cache.reorder_cache(torch.arange(B, device=dev).repeat_interleave(width))  # B -> B*w rows
    seqs = tok.reshape(-1, 1)  # (B*w, 1)
    finished = torch.isin(seqs[:, 0], stop)
    lengths = torch.ones(B * width, device=dev)

    for _ in range(1, max_new):
        if finished.all():
            break
        logits, cache = step(model, seqs[:, -1], cache, banned)
        logp = F.log_softmax(logits, -1)  # (B*w, V)
        logp[finished] = float("-inf")
        logp[finished, pad] = 0.0  # a finished beam keeps its score
        cand = (scores.reshape(-1, 1) + logp).view(B, width * V)
        scores, idx = cand.topk(width, dim=-1)  # (B, w)
        origin = (torch.arange(B, device=dev)[:, None] * width + idx // V).reshape(-1)  # parent row
        tok = (idx % V).reshape(-1, 1)
        cache.reorder_cache(origin)
        seqs = torch.cat([seqs[origin], tok], dim=1)
        lengths = lengths[origin] + (~finished[origin]).float()
        finished = finished[origin] | torch.isin(tok[:, 0], stop)

    final = scores.reshape(-1) / lengths ** length_penalty
    best = torch.arange(B, device=dev) * width + final.view(B, width).argmax(-1)
    return trim(seqs[best].tolist(), stop_ids)
