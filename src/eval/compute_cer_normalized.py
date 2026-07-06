"""Normalized CER: strips markdown/formatting before comparing."""
import argparse
import re
from pathlib import Path

import jiwer
import pandas as pd

GT = Path("data/arxiv_scans/generated/gt")
OUT = Path("outputs/ocr")
RESULTS = Path("results/tables")

def normalize(s: str) -> str:
    s = re.sub(r'[#*_`|>-]', ' ', s)          # markdown/table symbols
    s = re.sub(r'\$\$?.*?\$\$?', ' ', s)       # latex math
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
            rows.append({"level": lv, "page": hyp_path.stem, "cer": round(min(cer,1.0),4)})

    df = pd.DataFrame(rows)
    summary = df.groupby("level")[["cer"]].agg(["mean","median"]).reindex(
        ["clean","light","medium","heavy"])
    print(f"\n=== {args.engine}: NORMALIZED CER ===")
    print(summary)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / f"cer_normalized_{args.engine}_summary.csv")

if __name__ == "__main__":
    main()
