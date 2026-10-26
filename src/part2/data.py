"""Data for Part 2: next-token pretraining on browndw/human-ai-parallel-corpus.

    uv run python -m src.part2.data --out data/part2

The corpus has one row per text (human prompt, human continuation, and six LLM
continuations of the same prompt) and no splits. We split by the source document
(the part of doc_id before '@'), so a human text and its LLM continuations always
land in the same split -- otherwise test text would leak into training.

Each split is stored as one token stream: every text followed by <eos>, concatenated.
Training cuts the stream into non-overlapping windows of N_ctx + 1 tokens.
"""

import argparse
import zlib
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.utils import load_hf_parquet

DATASET = "browndw/human-ai-parallel-corpus"


def split_of(doc_id: str) -> str:
    """Deterministic 90 / 5 / 5 split on the source document."""
    h = zlib.crc32(doc_id.split("@")[0].encode()) % 100
    return "train" if h < 90 else "validation" if h < 95 else "test"


def windows(stream: torch.Tensor, n_ctx: int) -> torch.Tensor:
    """(n, n_ctx + 1) non-overlapping views; window i = stream[i*n_ctx : i*n_ctx + n_ctx + 1].
    Consecutive windows share one token so every token is a target exactly once."""
    n = (len(stream) - 1) // n_ctx
    return stream[: n * n_ctx + 1].unfold(0, n_ctx + 1, n_ctx)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("data/part2"))
    args = p.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    eos = tok.token_to_id("<eos>")
    d = load_hf_parquet(DATASET)["train"]
    texts = {"train": [], "validation": [], "test": []}
    for doc_id, text in zip(d["doc_id"], d["text"]):
        texts[split_of(doc_id)].append(text)

    args.out.mkdir(parents=True, exist_ok=True)
    for split, ts in texts.items():
        ids = []
        for e in tok.encode_batch(ts):
            ids.extend(e.ids)
            ids.append(eos)
        stream = torch.tensor(ids, dtype=torch.int32)
        torch.save(stream, args.out / f"{split}.pt")
        print(f"{split}: {len(ts)} texts, {len(stream) / 1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
