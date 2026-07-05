"""Inspect ds4sd/DocLayNet-v1.1 parquet schema (one test shard)."""
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO = "ds4sd/DocLayNet-v1.1"
shard = hf_hub_download(
    REPO, "data/test-00000-of-00002-635b47e9044a436c.parquet",
    repo_type="dataset")

ds = load_dataset("parquet", data_files=[shard], split="train")
print("rows:", len(ds))
print("columns:", ds.column_names, "\n")

ex = ds[0]
for k, v in ex.items():
    desc = type(v).__name__
    if isinstance(v, (str, int, float, bool)):
        desc += f" = {str(v)[:60]}"
    elif isinstance(v, list):
        desc += f" (len={len(v)})"
        if v:
            desc += f" first={str(v[0])[:80]}"
    elif isinstance(v, dict):
        desc += f" keys={list(v.keys())}"
    print(f"{k:22s} {desc}")
