"""Peek at MP-DocVQA schema (lmms-lab mirror) before downloading a subset."""
from datasets import load_dataset, get_dataset_split_names

name = "lmms-lab/MP-DocVQA"
print("splits:", get_dataset_split_names(name))

split = get_dataset_split_names(name)[0]
ds = load_dataset(name, split=split, streaming=True)
sample = next(iter(ds))
for k, v in sample.items():
    desc = type(v).__name__
    if isinstance(v, (str, int, float, bool)):
        desc += f" = {str(v)[:70]}"
    elif isinstance(v, list):
        desc += f" (len={len(v)})"
    print(f"{k:25s} {desc}")
