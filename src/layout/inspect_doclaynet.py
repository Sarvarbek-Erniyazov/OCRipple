"""Peek at DocLayNet-small schema safely without split inspection."""
from datasets import load_dataset

name = "pierreguillou/DocLayNet-small"

# Xatolik berayotgan get_dataset_split_names olib tashlandi
print("Dataset yuklanmoqda...")
ds = load_dataset(name, split="train", streaming=True, trust_remote_code=True)

sample = next(iter(ds))
print("\n--- DATASET SCHEMA ---")
for k, v in sample.items():
    desc = type(v).__name__
    if isinstance(v, (str, int, float, bool)):
        desc += f" = {str(v)[:60]}"
    elif isinstance(v, list):
        desc += f" (len={len(v)})"
        if v and not isinstance(v[0], (list, dict)):
            desc += f" e.g. {str(v[:3])[:60]}"
    print(f"{k:22s} {desc}")
