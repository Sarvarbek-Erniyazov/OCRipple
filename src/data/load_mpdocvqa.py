"""
OCRipple - MP-DocVQA subset loader (lmms-lab mirror).
Streams the val split, keeps first N_DOCS unique documents,
saves page images named by their page_id + qas.json with QA pairs.
"""
import ast
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

N_DOCS = 50
MIN_QAS = 300
OUT = Path("data/mpdocvqa")

def parse_list(s):
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return [s]

def main():
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/MP-DocVQA", split="val", streaming=True)

    kept_docs, qas = set(), []
    pages_per_doc = defaultdict(set)

    for ex in tqdm(ds, desc="streaming"):
        doc_id = str(ex["doc_id"])
        if doc_id not in kept_docs and len(kept_docs) >= N_DOCS:
            if len(qas) >= MIN_QAS:
                break
            continue
        kept_docs.add(doc_id)

        page_ids = parse_list(ex["page_ids"])
        for i, pid in enumerate(page_ids, start=1):
            img = ex.get(f"image_{i}")
            if img is None:
                continue
            p = OUT / "pages" / f"{pid}.png"
            if not p.exists():
                img.save(p)
            pages_per_doc[doc_id].add(p.name)

        qas.append({
            "qid": str(ex["questionId"]),
            "doc_id": doc_id,
            "question": ex["question"],
            "answers": parse_list(ex["answers"]),
            "answer_page_idx": int(ex["answer_page_idx"]),
            "page_ids": page_ids,
        })

    meta = {d: sorted(ps) for d, ps in pages_per_doc.items()}
    (OUT / "qas.json").write_text(json.dumps(qas, indent=2), encoding="utf-8")
    (OUT / "doc_pages.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"docs: {len(kept_docs)}, questions: {len(qas)}, "
          f"pages: {sum(len(v) for v in meta.values())}")

if __name__ == "__main__":
    main()
