# OCRipple — References

## OCR engines under evaluation
1. **Baidu Unlimited-OCR** (2026) — 3B MoE (~500M active), R-SWA
   (Reference Sliding Window Attention): constant KV-cache, 40+ pages
   in one forward pass, SOTA on OmniDocBench v1.5/v1.6.
   - GitHub: https://github.com/baidu/Unlimited-OCR
   - HF: https://huggingface.co/baidu/Unlimited-OCR
2. **DeepSeek-OCR** — vision-token compression baseline.
   - https://github.com/deepseek-ai/DeepSeek-OCR
3. **PaddleOCR** — lightweight production OCR (PP-OCR family).
   - https://github.com/PaddlePaddle/PaddleOCR
4. **Tesseract 5** — classical open-source OCR baseline.
   - https://github.com/tesseract-ocr/tesseract

## Datasets
- MP-DocVQA (lmms-lab mirror) — multi-page document VQA
- DocLayNet v1.1 (ds4sd) — layout detection, 11 classes
- arXiv papers (1706.03762, 1512.03385, 1810.04805) — synthetic scan GT
