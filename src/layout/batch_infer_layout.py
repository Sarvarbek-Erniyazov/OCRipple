"""
OCRipple - batch layout inference over all MP-DocVQA pages.
Saves per-page region predictions to results/tables/layout_regions.json
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from tqdm import tqdm

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)
MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "ocripple-layout/1")

PAGES_DIR = Path("data/mpdocvqa/pages")
OUT_PATH = Path("results/tables/layout_regions.json")

def main():
    pages = sorted(PAGES_DIR.glob("*.png"))
    print(f"{len(pages)} pages to process")

    results = {}
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print(f"resuming, {len(results)} already done")

    for p in tqdm(pages):
        if p.stem in results:
            continue
        try:
            resp = client.infer(str(p), model_id=MODEL_ID)
            preds = resp.get("predictions", [])
            results[p.stem] = [
                {"class": pr.get("class"), "confidence": pr.get("confidence"),
                 "x": pr.get("x"), "y": pr.get("y"),
                 "width": pr.get("width"), "height": pr.get("height")}
                for pr in preds
            ]
        except Exception as e:
            results[p.stem] = []
            print(f"ERR {p.name}: {repr(e)[:100]}")
        # save incrementally every 20 pages, in case of interruption
        if len(results) % 20 == 0:
            OUT_PATH.write_text(json.dumps(results), encoding="utf-8")
        time.sleep(0.1)  # be gentle on the free-tier API

    OUT_PATH.write_text(json.dumps(results), encoding="utf-8")
    print(f"done: {len(results)} pages -> {OUT_PATH}")

if __name__ == "__main__":
    main()
