"""Single entry point for the assignment. Each command forwards its arguments to a module.

    uv run python main.py <command> [args...]
    uv run python main.py <command> --help

Commands:
  tokenizer        Part 1: train the shared vi/ja/en BPE tokenizer
  part1-data       Part 1: tokenize the translation dataset
  part1-train      Part 1: train one FFN variant      (--config configs/part1/<variant>.yaml)
  part1-eval       Part 1: test ppl, BLEU, expert heatmaps   (--ckpt .../model.pt)
  part1-routing    Part 1: routing by language vs token type
  part1-plots      Part 1: comparison plots
  part2-data       Part 2: tokenize the human-AI parallel corpus
  part2-train      Part 2: pretrain with one optimizer   (--optimizer adamw|mars|lion|muon)
  part2-plots      Part 2: val loss / BLEU vs tokens, LR sweep
  part3            Part 3: decoding strategies on Pythia-160M
  upload           upload a checkpoint + results to HuggingFace
"""

import importlib
import sys

COMMANDS = {
    "tokenizer": "src.part1.tokenizer",
    "part1-data": "src.part1.data",
    "part1-train": "src.part1.train",
    "part1-eval": "src.part1.evaluate",
    "part1-routing": "src.part1.analyze_routing",
    "part1-plots": "src.part1.plots",
    "part2-data": "src.part2.data",
    "part2-train": "src.part2.train",
    "part2-plots": "src.part2.plots",
    "part3": "src.part3.run",
    "upload": "src.hf_upload",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 1)
    module = COMMANDS[sys.argv[1]]
    sys.argv = [f"main.py {sys.argv[1]}"] + sys.argv[2:]  # the module's argparse sees only its own args
    importlib.import_module(module).main()


if __name__ == "__main__":
    main()
