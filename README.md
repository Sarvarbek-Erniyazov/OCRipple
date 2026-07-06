# OCRipple 🌊

**Measuring how OCR errors ripple through document-understanding pipelines.**

Everyone builds RAG and document-QA systems on top of OCR. But does a lower
character error rate (CER) actually mean better downstream answers? OCRipple
answers this with controlled experiments across multiple OCR engines, four
scan-degradation levels, and two downstream tasks (retrieval + answer
presence on real documents, fact-recall on synthetic ones).

## Key finding

**Raw CER does not predict downstream answer quality — and can point the
wrong way entirely.**

| Engine | CER (clean) | CER (heavy) | Hit@3 | Answer ANLS | Email recall (heavy) |
|---|---|---|---|---|---|
| Tesseract | **7.5%** | 12.5% | 0.833 | 0.712 | 0.750 |
| DeepSeek-OCR | 16.5% | **15.8%** | 0.797 | 0.816 | **1.000** |
| PaddleOCR | 47.6% | 48.4% | 0.827 | **0.848** | **1.000** |

PaddleOCR has by far the worst raw CER (driven by a text-ordering artifact,
not misrecognition — see Limitations) yet produces the *best* downstream
answer quality. Tesseract has the best raw CER yet the worst answer quality
and loses 25% of ground-truth email addresses under heavy scan degradation,
while both VLM-family engines (DeepSeek-OCR, and — pending — Unlimited-OCR)
retain 100%.

## Pipeline
Scan → Layout Detection (RF-DETR via Roboflow, mAP@50 73.8%)
→ OCR (Tesseract | PaddleOCR | DeepSeek-OCR | Unlimited-OCR*)
→ [Stage 1] Retrieval + Answer-Presence (MP-DocVQA, real scans)
→ [Stage 2] Fact (email) Recall vs. Degradation (synthetic arXiv scans)
*Unlimited-OCR run in progress — see Status.

## Datasets

- **arXiv synthetic scans** — 3 papers (Attention, BERT, ResNet), 4 controlled
  degradation levels (clean/light/medium/heavy), perfect ground truth from the
  original PDF text layer. 172 images total.
- **MP-DocVQA subset** (lmms-lab mirror) — 50 real scanned industrial
  documents, 300 questions with gold answers and gold answer-pages. 350 pages.
- **DocLayNet-v1.1 subset** (ds4sd) — 500 pages, 11 layout classes, used to
  train the layout-detection model.

## OCR engines evaluated

| Engine | Type | Where it ran |
|---|---|---|
| Tesseract 5 | classical pixel-based | local CPU |
| PaddleOCR (PP-OCRv6) | production OCR | local CPU |
| DeepSeek-OCR | vision-language model | Kaggle T4 |
| Baidu Unlimited-OCR | vision-language model, R-SWA | Kaggle T4 (pending) |

## Layout detection

RF-DETR (Small), trained on 500 DocLayNet-v1.1 pages, 11 classes, 6839
annotated boxes. **mAP@50: 73.8%** · Precision 76.9% · Recall 71.8% · F1 74.3%.
Deployed as a hosted Roboflow API.

## Known limitations

- **PaddleOCR text ordering**: on multi-column author blocks, PaddleOCR
  groups all names, then all affiliations, then all emails — instead of
  reading each column top-to-bottom. Individual tokens are recognized
  correctly, but the raw CER is inflated by ~40 points from ordering alone.
  Confirmed by manual diff (see `report/technical_report.md`).
- **CER normalization**: markdown/table symbols produced by VLM engines are
  stripped before CER comparison; this is a coarse normalization and slightly
  affects hyphenated words identically across engines.
- **Hit@3 baseline**: ~35% of MP-DocVQA questions have ≤3 pages total, so
  Hit@3 is trivially high for those; the pattern holds on the harder subset
  (>3 pages) too — see `results/tables/`.

## Status

- ✅ Data pipeline: arXiv synthetic scans (172, 4 levels) + MP-DocVQA (350
  pages / 300 QAs) + DocLayNet subset (500 pages)
- ✅ Layout detection: trained + deployed (Roboflow)
- ✅ 3/4 OCR engines run to completion: Tesseract, PaddleOCR, DeepSeek-OCR
- ⏳ Unlimited-OCR: running on Kaggle T4 (Save & Run All commit), pending
- ✅ Stage 1 (retrieval + answer-presence) and Stage 2 (fact recall) complete
  for 3 engines
- 🚧 Final report + Unlimited-OCR integration in progress

## Reproducing

See `src/` for the full pipeline: `degradation/generate.py` (synthetic scans),
`data/load_mpdocvqa.py`, `layout/to_coco.py` + `infer_layout.py`,
`ocr_engines/run_ocr.py`, `eval/compute_cer.py`,
`eval/compute_cer_normalized.py`, `eval/fact_recall.py`, `rag/retrieve_and_score.py`,
`eval/master_summary.py`.

## References

- Baidu Unlimited-OCR: [GitHub](https://github.com/baidu/Unlimited-OCR) ·
  [HuggingFace](https://huggingface.co/baidu/Unlimited-OCR) — 3B params (500M
  active, MoE), Reference Sliding Window Attention (R-SWA), reports SOTA on
  OmniDocBench v1.5/v1.6. See `report/technical_report.md` for how our
  independent test relates to these claims.
- DeepSeek-OCR: [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- PaddleOCR: [GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- Tesseract: [GitHub](https://github.com/tesseract-ocr/tesseract)
- MP-DocVQA (lmms-lab mirror), DocLayNet-v1.1 (ds4sd)
