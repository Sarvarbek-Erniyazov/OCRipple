"""
OCRipple - OCR runner.
Runs a chosen engine over (a) arXiv synthetic scans (all 4 levels)
and (b) MP-DocVQA pages. Writes one .txt per image + timing log.
Usage:
    python src/ocr_engines/run_ocr.py --engine tesseract
    python src/ocr_engines/run_ocr.py --engine tesseract --set arxiv
"""
import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

ARXIV = Path("data/arxiv_scans/generated")
MPDOC = Path("data/mpdocvqa/pages")
OUT = Path("outputs/ocr")

def get_engine(name):
    if name == "tesseract":
        import pytesseract, shutil
        if shutil.which("tesseract") is None:
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        def run(img_path):
            return pytesseract.image_to_string(str(img_path), lang="eng")
        return run
    if name == "paddle":
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        def run(img_path):
            result = ocr.ocr(str(img_path), cls=False)
            lines = []
            for page in result or []:
                for item in page or []:
                    lines.append(item[1][0])
            return "\n".join(lines)
        return run
    raise ValueError(f"unknown engine: {name}")

def collect_images(which):
    tasks = []  # (image_path, relative_output_stem)
    if which in ("arxiv", "all"):
        for lv in ["clean", "light", "medium", "heavy"]:
            for p in sorted((ARXIV / lv).glob("*.png")):
                tasks.append((p, f"arxiv/{lv}/{p.stem}"))
    if which in ("mpdocvqa", "all"):
        for p in sorted(MPDOC.glob("*.png")):
            tasks.append((p, f"mpdocvqa/{p.stem}"))
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["tesseract", "paddle"])
    ap.add_argument("--set", default="all", choices=["arxiv", "mpdocvqa", "all"])
    args = ap.parse_args()

    run = get_engine(args.engine)
    tasks = collect_images(args.set)
    print(f"{args.engine}: {len(tasks)} images")

    timing = []
    for img_path, stem in tqdm(tasks, desc=args.engine):
        out_path = OUT / args.engine / f"{stem}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():          # resumable
            continue
        t0 = time.perf_counter()
        try:
            text = run(img_path)
        except Exception as e:
            text = ""
            print(f"\nERROR on {img_path.name}: {e}")
        dt = time.perf_counter() - t0
        out_path.write_text(text, encoding="utf-8")
        timing.append({"image": str(img_path), "seconds": round(dt, 3)})

    log = OUT / args.engine / "timing.json"
    existing = json.loads(log.read_text()) if log.exists() else []
    log.write_text(json.dumps(existing + timing, indent=2))
    total = sum(t["seconds"] for t in timing)
    print(f"done. new items: {len(timing)}, time: {total:.1f}s")

if __name__ == "__main__":
    main()
