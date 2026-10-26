"""Shared helpers."""

import os
from pathlib import Path

# A network monitor on the dev machine sets SSLKEYLOGFILE to a named pipe; OpenSSL
# then aborts on Windows ("no OPENSSL_Applink") at the first HTTPS request.
# Must be cleared before anything opens an SSL connection.
os.environ.pop("SSLKEYLOGFILE", None)
try:  # same machine: TLS interception needs the OS certificate store (optional dependency)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

SPLITS = ("train", "validation", "test")


def load_hf_parquet(repo_id: str) -> dict[str, dict[str, list]]:
    """Download a HF dataset's parquet files and return {split: {column: list}}.

    Used instead of `datasets.load_dataset` because importing `datasets` crashes
    on some Windows / Python 3.14 setups (aiohttp + OpenSSL). Files are cached
    by huggingface_hub, so this only downloads once.
    """
    root = Path(snapshot_download(repo_id, repo_type="dataset", allow_patterns="*.parquet"))
    out = {}
    for split in SPLITS:
        # Files are named e.g. "test.parquet" or "data/test-00000-of-00001.parquet".
        files = sorted(f for f in root.rglob("*.parquet") if f.name.split("-")[0].split(".")[0] == split)
        if files:
            tables = [pq.read_table(f) for f in files]
            out[split] = {col: sum((t.column(col).to_pylist() for t in tables), []) for col in tables[0].schema.names}
    if not out:  # no split-named files (e.g. one file per source): everything is "train"
        tables = [pq.read_table(f) for f in sorted(root.rglob("*.parquet"))]
        out["train"] = {col: sum((t.column(col).to_pylist() for t in tables), []) for col in tables[0].schema.names}
    return out
