"""Upload a trained checkpoint (+ its eval results) to a public HuggingFace model repo.

    uv run python -m src.hf_upload --ckpt checkpoints/part1/1_dense/model.pt \
        --results results/part1/1_dense --repo anlp-a2-part1-1_dense

Needs a write token: `huggingface-cli login`, or the HF_TOKEN environment variable.
Prints the repo URL; put it in the README / report.
"""

import argparse
import json
from pathlib import Path

import src.utils  # noqa: F401  (clears SSLKEYLOGFILE before any HTTPS)
from huggingface_hub import HfApi


def model_card(name: str, config: dict, metrics: dict, wandb_url: str | None) -> str:
    lines = [f"# {name}", "", "Advanced NLP (Monsoon 2026), Assignment 2.", ""]
    if wandb_url:
        lines += [f"Training logs: {wandb_url}", ""]
    lines += ["## Config", "```json", json.dumps(config, indent=2), "```", ""]
    if metrics:
        lines += ["## Test metrics", "```json", json.dumps(metrics, indent=2), "```", ""]
    lines += ["Load with `torch.load('model.pt')` -> {'model': state_dict, 'config': dict}; "
              "see the assignment repository for the model code."]
    return "\n".join(lines)


def upload(ckpt: Path, repo: str, results: Path | None = None) -> str:
    api = HfApi()
    repo_id = repo if "/" in repo else f"{api.whoami()['name']}/{repo}"
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

    import torch
    config = torch.load(ckpt, map_location="cpu", weights_only=False).get("config", {})
    metrics = {}
    if results and (results / "metrics.json").exists():
        metrics = json.loads((results / "metrics.json").read_text())
    wandb_file = ckpt.parent / "wandb_url.txt"
    wandb_url = wandb_file.read_text().strip() if wandb_file.exists() else None

    api.upload_file(path_or_fileobj=str(ckpt), path_in_repo="model.pt", repo_id=repo_id)
    api.upload_file(path_or_fileobj=model_card(repo_id.split("/")[1], config, metrics, wandb_url).encode(),
                    path_in_repo="README.md", repo_id=repo_id)
    if results and results.exists():
        api.upload_folder(folder_path=str(results), path_in_repo="results", repo_id=repo_id)
    url = f"https://huggingface.co/{repo_id}"
    print(url)
    return url


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--repo", required=True, help="'name' (under your account) or 'user/name'")
    p.add_argument("--results", type=Path, default=None)
    args = p.parse_args()
    upload(args.ckpt, args.repo, args.results)


if __name__ == "__main__":
    main()
