"""
OCRipple — synthetic scan generator.
Takes born-digital PDFs (perfect text ground truth) and produces
scan-degraded images at 4 controlled levels: clean / light / medium / heavy.
"""
import argparse, json, random
from pathlib import Path

import cv2
import numpy as np
import fitz  # PyMuPDF

random.seed(42)
np.random.seed(42)

DPI = 200

def render_pages(pdf_path: Path):
    """PDF -> list of (page_image BGR, ground_truth_text)."""
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        yield img, page.get_text("text")

def degrade(img: np.ndarray, level: str) -> np.ndarray:
    """Apply controlled scan-like degradation."""
    out = img.copy()
    if level == "clean":
        return out

    params = {
        "light":  dict(blur=1, noise=4,  jpeg=80, rotate=0.3),
        "medium": dict(blur=2, noise=10, jpeg=55, rotate=0.8),
        "heavy":  dict(blur=3, noise=18, jpeg=35, rotate=1.5),
    }[level]

    # 1) slight rotation (scanner skew)
    h, w = out.shape[:2]
    angle = random.uniform(-params["rotate"], params["rotate"])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderValue=(255, 255, 255))

    # 2) blur (bad focus / low dpi)
    k = params["blur"] * 2 + 1
    out = cv2.GaussianBlur(out, (k, k), 0)

    # 3) gaussian noise (sensor noise)
    noise = np.random.normal(0, params["noise"], out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4) uneven illumination (shadow gradient)
    grad = np.tile(np.linspace(0.92, 1.0, w), (h, 1))
    out = np.clip(out * grad[..., None], 0, 255).astype(np.uint8)

    # 5) jpeg compression artifacts
    _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, params["jpeg"]])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default="data/arxiv_scans/pdfs")
    ap.add_argument("--out_dir", default="data/arxiv_scans/generated")
    args = ap.parse_args()

    pdf_dir, out_dir = Path(args.pdf_dir), Path(args.out_dir)
    levels = ["clean", "light", "medium", "heavy"]
    manifest = []

    for pdf in sorted(pdf_dir.glob("*.pdf")):
        for pno, (img, gt_text) in enumerate(render_pages(pdf)):
            gt_path = out_dir / "gt" / f"{pdf.stem}_p{pno:03d}.txt"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text(gt_text, encoding="utf-8")
            for lv in levels:
                img_path = out_dir / lv / f"{pdf.stem}_p{pno:03d}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_path), degrade(img, lv))
                manifest.append({"doc": pdf.stem, "page": pno,
                                 "level": lv, "image": str(img_path),
                                 "gt": str(gt_path)})
        print(f"done: {pdf.name}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total items: {len(manifest)}")

if __name__ == "__main__":
    main()
