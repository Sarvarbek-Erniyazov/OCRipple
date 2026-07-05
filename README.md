# OCRipple 🌊

**Measuring how OCR errors ripple through RAG pipelines.**

Everyone builds RAG on scanned PDFs. But how much does a 3% OCR
character error rate actually cost you in final answer accuracy?
OCRipple answers this with controlled experiments across 4 OCR
engines, 4 scan-degradation levels, and layout-aware error analysis.

## Key Questions
1. How does OCR CER map to downstream QA accuracy (ANLS)?
2. Which error types (numbers, tables, line skips) hurt RAG the most?
3. Do errors in specific layout regions (tables vs. body text) matter more?

## Pipeline
Scan → Layout Detection (RF-DETR via Roboflow) → OCR
(Tesseract | PaddleOCR | DeepSeek-OCR | Unlimited-OCR) → RAG → QA eval

## Status
🚧 Day 1/7 — data preparation

## Datasets
- MP-DocVQA (subset) — real scans + QA pairs
- arXiv synthetic scans — perfect ground truth, 4 degradation levels
- DocLayNet (subset) — layout detection training
