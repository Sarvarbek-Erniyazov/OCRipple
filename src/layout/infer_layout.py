"""
OCRipple - layout inference via Roboflow hosted model (direct, no workflow).
Usage: python src/layout/infer_layout.py --image path/to/page.png
"""
import argparse
import json
import os

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "ocripple-layout/1")

def detect(image_path: str):
    return client.infer(image_path, model_id=MODEL_ID)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    result = detect(args.image)
    preds = result.get("predictions", [])
    print(f"detections: {len(preds)}")
    for p in preds[:10]:
        print(f"  {p.get('class'):16s} conf={p.get('confidence'):.2f} "
              f"x={p.get('x'):.0f} y={p.get('y'):.0f} "
              f"w={p.get('width'):.0f} h={p.get('height'):.0f}")
