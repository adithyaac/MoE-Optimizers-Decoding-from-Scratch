"""Data pipeline for Part 1: vi/ja -> en translation as conditional LM.

Step 1 (once, locally or on Colab):
    uv run python -m src.part1.data --out data/part1
  tokenizes every split and saves it, so training never re-tokenizes.
Step 2 (in training): make_loader(...) builds length-bucketed, dynamically
  padded batches from the saved files.

Sequence format:  <vi|ja> source... <sep> english... <eos>
Only the English tokens and <eos> are trained on (the source is context).
"""

import argparse
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, Sampler

from src.part1.tokenizer import COL_EN, COL_JA, COL_VI, DATASET
from src.utils import load_hf_parquet

PAD_ID = 0  # checked in the tokenizer sanity check
IGNORE = -100  # cross_entropy ignore_index
LANGS = {COL_VI: 0, COL_JA: 1}  # language id stored per example (for the heatmap)


# ---------------------------------------------------------------- tokenizing


def build_examples(tokenizer: Tokenizer, split: dict, n_ctx: int) -> dict:
    """Turn every triplet into two examples: vi->en and ja->en.

    A sequence of L tokens gives L-1 (input, target) pairs after the shift in
    collate, so it fits the model when L <= n_ctx + 1. Overlong sources are
    truncated (the target is kept whole); if the English alone does not fit,
    the example is dropped.

    Returns flat, compact storage (one big tensor instead of ~900k lists):
      "ids":       int32 (total_tokens,)  all sequences concatenated
      "offsets":   int64 (N + 1,)         sequence i is ids[offsets[i]:offsets[i+1]]
      "tgt_start": int32 (N,)             index (within sequence i) of the first English token
      "lang":      int8  (N,)             LANGS[...] of the source language
    """
    max_len = n_ctx + 1
    sep, eos = tokenizer.token_to_id("<sep>"), tokenizer.token_to_id("<eos>")
    en_ids = [e.ids for e in tokenizer.encode_batch(split[COL_EN])]

    flat, lengths, tgt_start, lang = [], [], [], []
    n_trunc = n_drop = 0
    for col in (COL_VI, COL_JA):
        src_ids = [e.ids for e in tokenizer.encode_batch(split[col])]
        lang_tok = tokenizer.token_to_id(f"<{col}>")
        for src, en in zip(src_ids, en_ids):
            room = max_len - len(en) - 3  # 3 = <lang>, <sep>, <eos>
            if room < 1:
                n_drop += 1
                continue
            if len(src) > room:
                src = src[:room]
                n_trunc += 1
            seq = [lang_tok, *src, sep, *en, eos]
            flat.extend(seq)
            lengths.append(len(seq))
            tgt_start.append(len(src) + 2)  # skip <lang>, src, <sep>
            lang.append(LANGS[col])

    print(f"  truncated {n_trunc} sources, dropped {n_drop} examples")
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64)
    offsets[1:] = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
    return {
        "ids": torch.tensor(flat, dtype=torch.int32),
        "offsets": offsets,
        "tgt_start": torch.tensor(tgt_start, dtype=torch.int32),
        "lang": torch.tensor(lang, dtype=torch.int8),
    }


# ------------------------------------------------------------------- dataset


class TranslationDataset(Dataset):
    def __init__(self, path: Path):
        data = torch.load(path)
        self.ids, self.offsets = data["ids"], data["offsets"]
        self.tgt_start, self.lang = data["tgt_start"], data["lang"]
        self.lengths = (self.offsets[1:] - self.offsets[:-1]).tolist()

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        seq = self.ids[self.offsets[i] : self.offsets[i + 1]]
        return seq, int(self.tgt_start[i]), int(self.lang[i])


class BucketBatchSampler(Sampler[list[int]]):
    """Yields lists of indices; each list is one batch of similar lengths.

    Sorting the whole dataset by length would make every batch the same
    "difficulty" in a fixed order; sorting only within random chunks keeps
    padding low while still giving random batches.
    """

    def __init__(self, lengths: list[int], batch_size: int, chunk_batches: int = 100,
                 shuffle: bool = True, seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.chunk_size = batch_size * chunk_batches
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.skip = 0  # batches to skip at the start of the next epoch (resuming mid-epoch)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(idx)  # rng state carries over, so each epoch differs
        batches = []
        for c in range(0, len(idx), self.chunk_size):
            chunk = sorted(idx[c : c + self.chunk_size], key=self.lengths.__getitem__)
            batches += [chunk[b : b + self.batch_size] for b in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        skip, self.skip = self.skip, 0
        yield from batches[skip:]

    def __len__(self) -> int:
        n_chunks, rem = divmod(len(self.lengths), self.chunk_size)
        per_chunk = (self.chunk_size + self.batch_size - 1) // self.batch_size
        return n_chunks * per_chunk + (rem + self.batch_size - 1) // self.batch_size


def collate(batch: list[tuple[torch.Tensor, int, int]]) -> dict[str, torch.Tensor]:
    """Pad a batch and apply the next-token shift.

    For a sequence s of length L:  input = s[:-1],  target = s[1:]
    so target[j] = s[j+1]. English starts at s[tgt_start], i.e. at
    target[tgt_start - 1] (the position holding <sep> predicts the first
    English token). Everything before that is source -> IGNORE.
    """
    T = max(len(seq) for seq, _, _ in batch) - 1  # dynamic padding
    B = len(batch)
    input_ids = torch.full((B, T), PAD_ID, dtype=torch.long)
    targets = torch.full((B, T), IGNORE, dtype=torch.long)
    for b, (seq, tgt_start, _) in enumerate(batch):
        seq = seq.long()
        n = len(seq) - 1
        input_ids[b, :n] = seq[:-1]
        targets[b, tgt_start - 1 : n] = seq[tgt_start:]
    lang = torch.tensor([lang for _, _, lang in batch], dtype=torch.long)
    return {"input_ids": input_ids, "targets": targets, "lang": lang}


def make_loader(path: Path, batch_size: int, shuffle: bool, seed: int = 0,
                num_workers: int = 2) -> DataLoader:
    ds = TranslationDataset(path)
    sampler = BucketBatchSampler(ds.lengths, batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


# ---------------------------------------------------------------------- main


def sanity_check(tokenizer: Tokenizer, path: Path) -> None:
    loader = make_loader(path, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    x, y = batch["input_ids"], batch["targets"]
    print(f"batch shapes: input {tuple(x.shape)}, targets {tuple(y.shape)}, lang {batch['lang'].tolist()}")
    b = int((x != PAD_ID).sum(1).argmin())  # shortest example: has padding to look at
    print(f"{'pos':>4}  {'input':<16} target")
    for t in range(x.shape[1]):
        tgt = "-100" if y[b, t] == IGNORE else tokenizer.id_to_token(int(y[b, t]))
        print(f"{t:>4}  {tokenizer.id_to_token(int(x[b, t])):<16} {tgt}")
    # targets are inputs shifted left by one, wherever the target is not masked
    m = y[:, :-1] != IGNORE
    assert (y[:, :-1][m] == x[:, 1:][m]).all(), "shift is wrong"
    print("shift check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("data/part1"))
    parser.add_argument("--n-ctx", type=int, default=256)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    assert tokenizer.token_to_id("<pad>") == PAD_ID
    ds = load_hf_parquet(DATASET)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        print(f"{name}:")
        examples = build_examples(tokenizer, ds[name], args.n_ctx)
        torch.save(examples, args.out / f"{name}.pt")
        print(f"  {len(examples['tgt_start'])} examples, {len(examples['ids'])} tokens")

    sanity_check(tokenizer, args.out / "validation.pt")


if __name__ == "__main__":
    main()
