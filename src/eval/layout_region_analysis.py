"""
OCRipple - does OCR-error impact depend on page layout composition?
For each question's gold answer page, flag whether it contains a Table
region, then compare downstream ANLS for Table-pages vs non-Table-pages,
per engine.
"""
import json
from pathlib import Path

import pandas as pd

QAS = Path("data/mpdocvqa/qas.json")
LAYOUT = Path("results/tables/layout_regions.json")
RESULTS = Path("results/tables")
ENGINES = ["tesseract", "deepseek_ocr", "paddle"]

def main():
    qas = json.loads(QAS.read_text(encoding="utf-8"))
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))

    # map qid -> gold page id + table flag
    qid_meta = {}
    for q in qas:
        page_ids = q["page_ids"]
        idx = q["answer_page_idx"]
        if idx >= len(page_ids):
            continue
        gold_page = page_ids[idx]
        regions = layout.get(gold_page, [])
        classes = [r["class"] for r in regions]
        has_table = "Table" in classes
        has_list = "List-item" in classes
        has_picture = "Picture" in classes
        qid_meta[str(q["qid"])] = {
            "gold_page": gold_page, "has_table": has_table,
            "has_list": has_list, "has_picture": has_picture,
        }

    n_table = sum(1 for v in qid_meta.values() if v["has_table"])
    print(f"Questions with Table on gold page: {n_table}/{len(qid_meta)}")

    summary_rows = []
    for eng in ENGINES:
        rag = pd.read_csv(RESULTS / f"rag_{eng}_per_question.csv")
        rag["qid"] = rag["qid"].astype(str)
        rag["has_table"] = rag["qid"].map(lambda q: qid_meta.get(q, {}).get("has_table"))
        rag = rag.dropna(subset=["has_table"])

        grp = rag.groupby("has_table").agg(
            anls_mean=("anls", "mean"), hit_mean=("hit", "mean"),
            n=("qid", "count")).reset_index()

        print(f"\n=== {eng}: downstream quality by page composition ===")
        print(grp.to_string(index=False))

        for _, row in grp.iterrows():
            summary_rows.append({
                "engine": eng, "has_table": row["has_table"],
                "anls_mean": round(row["anls_mean"], 3),
                "hit_mean": round(row["hit_mean"], 3),
                "n": int(row["n"]),
            })

    out = pd.DataFrame(summary_rows)
    out.to_csv(RESULTS / "layout_region_analysis.csv", index=False)
    print(f"\nsaved: {RESULTS}/layout_region_analysis.csv")

if __name__ == "__main__":
    main()
