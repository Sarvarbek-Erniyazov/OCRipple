"""
OCRipple - CER/WER on arXiv synthetic scans.
Compares engine output vs perfect PDF ground truth, per degradation level.
Usage: python src/eval/compute_cer.py --engine tesseract
"""
import argparse
import json
from pathlib import Path

import jiwer
import pandas as pd

GT = Path("data/arxiv_scans/generated/gt")
OUT = Path("outputs/ocr")
RESULTS = Path("results/tables")

def normalize(s: str) -> str:
    # light normalization: collapse whitespace, lowercase
    return " ".join(s.lower().split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()

    rows = []
    for lv in ["clean", "light", "medium", "heavy"]:
        for hyp_path in sorted((OUT / args.engine / "arxiv" / lv).glob("*.txt")):
            gt_path = GT / f"{hyp_path.stem}.txt"
            if not gt_path.exists():
                continue
            gt = normalize(gt_path.read_text(encoding="utf-8"))
            hyp = normalize(hyp_path.read_text(encoding="utf-8"))
            if not gt.strip():
                continue
            cer = jiwer.cer(gt, hyp) if hyp.strip() else 1.0
            wer = jiwer.wer(gt, hyp) if hyp.strip() else 1.0
            rows.append({"engine": args.engine, "level": lv,
                         "page": hyp_path.stem,
                         "cer": round(min(cer, 1.0), 4),
                         "wer": round(min(wer, 1.0), 4)})

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"cer_{args.engine}_per_page.csv", index=False)

    summary = (df.groupby("level")[["cer", "wer"]]
                 .agg(["mean", "median"]).round(4))
    summary = summary.reindex(["clean", "light", "medium", "heavy"])
    print(f"\n=== {args.engine}: CER/WER by degradation level ===")
    print(summary)
    summary.to_csv(RESULTS / f"cer_{args.engine}_summary.csv")

if __name__ == "__main__":
    main()
