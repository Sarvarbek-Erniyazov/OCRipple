"""
OCRipple - Stage 1 downstream experiment: retrieval + answer-presence.
For each engine's OCR text, measure:
  - Hit@k: is the gold answer page retrieved in top-k?
  - Answer Presence (ANLS-based): is the gold answer text findable
    in the retrieved evidence?
Usage: python src/rag/retrieve_and_score.py --engine tesseract --k 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein

QAS = Path("data/mpdocvqa/qas.json")
DOC_PAGES = Path("data/mpdocvqa/doc_pages.json")
OCR_ROOT = Path("outputs/ocr")
RESULTS = Path("results/tables")

def anls(pred: str, gold: str) -> float:
    pred, gold = pred.lower().strip(), gold.lower().strip()
    if not pred and not gold:
        return 1.0
    m = max(len(pred), len(gold))
    if m == 0:
        return 1.0
    dist = Levenshtein.distance(pred, gold)
    nl = dist / m
    return 1 - nl if nl < 0.5 else 0.0

def best_anls_in_text(text: str, gold_answers: list, window=60) -> float:
    """Slide a window over text, return best ANLS against any gold answer."""
    text_l = text.lower()
    best = 0.0
    for ans in gold_answers:
        ans_l = ans.lower().strip()
        if not ans_l:
            continue
        if ans_l in text_l:
            return 1.0  # exact substring, fastest path
        # fallback: check windows around each occurrence-like scan (coarse)
        step = max(1, window // 2)
        for i in range(0, max(1, len(text_l) - window), step):
            chunk = text_l[i:i + window]
            score = anls(chunk, ans_l)
            best = max(best, score)
    return best

def load_engine_text(engine, page_id):
    p = OCR_ROOT / engine / "mpdocvqa" / f"{page_id}.txt"
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    qas = json.loads(QAS.read_text(encoding="utf-8"))
    doc_pages = json.loads(DOC_PAGES.read_text(encoding="utf-8"))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []
    for q in qas:
        doc_id = q["doc_id"]
        page_ids = q["page_ids"]  # ordered pages of this doc as seen in the question
        gold_page_id = page_ids[q["answer_page_idx"]] if q["answer_page_idx"] < len(page_ids) else None

        page_texts = [load_engine_text(args.engine, pid) for pid in page_ids]
        non_empty = [(pid, t) for pid, t in zip(page_ids, page_texts) if t.strip()]
        if not non_empty:
            rows.append({"qid": q["qid"], "hit": False, "anls": 0.0, "n_pages": 0})
            continue

        ids, texts = zip(*non_empty)
        page_embs = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        q_emb = model.encode([q["question"]], convert_to_numpy=True, show_progress_bar=False)[0]

        sims = page_embs @ q_emb / (
            np.linalg.norm(page_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        top_idx = np.argsort(-sims)[:args.k]
        top_ids = [ids[i] for i in top_idx]
        evidence = " ".join(texts[i] for i in top_idx)

        hit = gold_page_id in top_ids
        score = best_anls_in_text(evidence, q["answers"])

        rows.append({"qid": q["qid"], "doc_id": doc_id, "hit": hit,
                    "anls": score, "n_pages": len(non_empty)})

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"rag_{args.engine}_per_question.csv", index=False)

    print(f"\n=== {args.engine}: retrieval + answer-presence (k={args.k}) ===")
    print(f"Hit@{args.k}:        {df['hit'].mean():.3f}")
    print(f"Answer ANLS (mean): {df['anls'].mean():.3f}")
    print(f"Questions scored:   {len(df)}")

if __name__ == "__main__":
    main()
