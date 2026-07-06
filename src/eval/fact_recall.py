"""
OCRipple - Stage 2: fact (email) recall vs degradation level.
Extracts emails from ground truth, checks if they survive OCR
at each degradation level. Direct CER->downstream-task link.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

GT = Path("data/arxiv_scans/generated/gt")
OUT = Path("outputs/ocr")
RESULTS = Path("results/tables")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()

    rows = []
    for lv in ["clean", "light", "medium", "heavy"]:
        for gt_path in sorted(GT.glob("*.txt")):
            gt_text = gt_path.read_text(encoding="utf-8", errors="ignore")
            targets = set(EMAIL_RE.findall(gt_text))
            if not targets:
                continue
            hyp_path = OUT / args.engine / "arxiv" / lv / gt_path.name
            hyp_text = hyp_path.read_text(encoding="utf-8", errors="ignore") if hyp_path.exists() else ""
            found = sum(1 for t in targets if t in hyp_text)
            rows.append({"level": lv, "page": gt_path.stem,
                        "n_targets": len(targets), "n_found": found,
                        "recall": found / len(targets)})

    df = pd.DataFrame(rows)
    summary = df.groupby("level")["recall"].mean().reindex(
        ["clean", "light", "medium", "heavy"])
    print(f"\n=== {args.engine}: email recall by degradation level ===")
    print(summary)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / f"email_recall_{args.engine}.csv")

if __name__ == "__main__":
    main()
