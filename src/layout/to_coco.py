"""
OCRipple - DocLayNet subset -> COCO export for Roboflow upload.
Samples N_PAGES from the test shard, writes images + _annotations.coco.json,
zips everything into data/layout/roboflow_upload.zip
"""
import json
import random
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "ds4sd/DocLayNet-v1.1"
SHARD = "data/test-00000-of-00002-635b47e9044a436c.parquet"
N_PAGES = 500
OUT = Path("data/layout/roboflow_upload")

# DocLayNet v1.1 category ids (1-indexed)
CATEGORIES = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
    5: "Page-footer", 6: "Page-header", 7: "Picture",
    8: "Section-header", 9: "Table", 10: "Text", 11: "Title",
}

def main():
    random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    shard = hf_hub_download(REPO, SHARD, repo_type="dataset")  # cached
    ds = load_dataset("parquet", data_files=[shard], split="train")

    idxs = random.sample(range(len(ds)), N_PAGES)

    coco = {
        "info": {"description": "DocLayNet-v1.1 subset for OCRipple layout model"},
        "licenses": [], "images": [], "annotations": [],
        "categories": [{"id": k, "name": v, "supercategory": "layout"}
                       for k, v in CATEGORIES.items()],
    }

    ann_id = 1
    cat_counts = {v: 0 for v in CATEGORIES.values()}

    for img_id, i in enumerate(tqdm(idxs, desc="exporting"), start=1):
        ex = ds[i]
        img = ex["image"]
        w, h = img.size
        fname = f"page_{img_id:04d}.png"
        img.save(OUT / fname)

        coco["images"].append(
            {"id": img_id, "file_name": fname, "width": w, "height": h})

        for bbox, cat in zip(ex["bboxes"], ex["category_id"]):
            x, y, bw, bh = bbox
            # clamp to image bounds, skip degenerate boxes
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw <= 1 or bh <= 1:
                continue
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": int(cat),
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0, "segmentation": [],
            })
            cat_counts[CATEGORIES[int(cat)]] += 1
            ann_id += 1

    (OUT / "_annotations.coco.json").write_text(
        json.dumps(coco), encoding="utf-8")

    zip_path = OUT.parent / "roboflow_upload.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUT.iterdir():
            zf.write(f, f.name)

    print(f"\nimages: {len(coco['images'])}, annotations: {len(coco['annotations'])}")
    print("per-category:", json.dumps(cat_counts, indent=2))
    print(f"zip ready: {zip_path.resolve()}")

if __name__ == "__main__":
    main()
