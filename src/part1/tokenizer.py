"""Train a shared byte-level BPE tokenizer for vi / ja / en.

Usage: uv run python -m src.part1.tokenizer --vocab-size 16000
Skeleton: fill in the TODOs (1 -> 4).
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
import numpy as np

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from src.utils import load_hf_parquet

DATASET = "belumind/en-vi-ja-curated-500k-triplets"

# These define your prompt format: <vi> source <sep> english <eos>
SPECIAL_TOKENS = ["<pad>", "<eos>", "<sep>", "<vi>", "<ja>"]

COL_EN = "en"
COL_VI = "vi"
COL_JA = "ja"


def text_iterator(split, batch_size: int = 1000) -> Iterator[list[str]]:
    for i in range(0, len(split[COL_EN]), batch_size):
        yield [t for col in (COL_VI, COL_EN, COL_JA) for t in split[col][i : i + batch_size]]

def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train(tokenizer: Tokenizer, split, vocab_size: int) -> Tokenizer:
    tr = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_iterator(split), trainer=tr)
    return tokenizer


def sanity_check(tokenizer: Tokenizer, split) -> None:
    for lang in [COL_VI, COL_EN, COL_JA]:
        print(f"sanity check: {split[lang][0]}")
        encoded = tokenizer.encode(split[lang][0])
        print(f"encoded: {encoded.tokens} (len={len(encoded.tokens)})")
        decoded = tokenizer.decode(encoded.ids)
        print(f"decoded: {decoded}")

    encs = {lang: tokenizer.encode_batch(split[lang][:2000]) for lang in (COL_VI, COL_EN, COL_JA)}

    en_mean = np.mean([len(e.ids) for e in encs[COL_EN]])
    for lang in (COL_VI, COL_EN, COL_JA):
        mean = np.mean([len(e.ids) for e in encs[lang]])
        print(f"{lang}: mean tokens/sentence = {mean:.1f}, ratio to English = {mean / en_mean:.2f}")

    for src in (COL_VI, COL_JA):
        lengths = [
            len(encs[src][i].ids) + len(encs[COL_EN][i].ids) + 3
            for i in range(2000)
        ]
        print(f"{src}->en: mean length = {np.mean(lengths):.1f}, "
            f"99th percentile = {np.percentile(lengths, 99):.0f}, max = {max(lengths)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--out", type=Path, default=Path("assets/tokenizer.json"))
    args = parser.parse_args()

    ds = load_hf_parquet(DATASET)
    train_split = ds["train"]  # train ONLY on the training split, never on test

    tokenizer = train(build_tokenizer(), train_split, args.vocab_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))
    print(f"saved to {args.out} (vocab size {tokenizer.get_vocab_size()})")

    sanity_check(tokenizer, train_split)


if __name__ == "__main__":
    main()
