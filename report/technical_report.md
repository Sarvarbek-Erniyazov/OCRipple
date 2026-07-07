
## Output integrity audit (all 4 engines)

Before trusting any engine's CER/downstream numbers, we audited raw output
files for empty or near-empty results across all 522 pages:

| Engine | Empty outputs | Rate |
|---|---|---|
| Tesseract | 5 | 1.0% |
| DeepSeek-OCR | 1 | 0.2% |
| PaddleOCR | 0 | 0.0% |
| Unlimited-OCR | 279 | **53.4%** |

Tesseract, DeepSeek-OCR, and PaddleOCR fall within normal noise (isolated
corrupt source images) and their reported metrics are trustworthy.
Unlimited-OCR's 53.4% empty rate is not model failure — `timing.json` shows
these pages consumed normal-to-long inference time (e.g. 22.9s for
`attention_p002.png`) before yielding an empty post-processed result — but a
bug in our `<|det|>...<|/det|>` tag-stripping regex, which silently consumed
entire outputs on densely-tagged pages. Raw untagged output was not
persisted, so this cannot be corrected retroactively. **Unlimited-OCR's
downstream numbers are therefore not reported as valid results in this
report.**
