# app.py
import base64
import io
import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import requests

app = FastAPI(title="UI Element Detector")

class DetectRequest(BaseModel):
    image: Optional[str] = None  # base64 or url

def load_image_from_base64(b64: str) -> Image.Image:
    try:
        header_removed = b64.split(",")[-1]
        img_bytes = base64.b64decode(header_removed)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

def load_image_from_url(url: str) -> Image.Image:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {e}")

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_ui_elements_cv2(img_bgr: np.ndarray):
    """
    Heuristic detector:
    - finds rectangular-ish contours (likely buttons/cards/panels)
    - uses aspect ratio and area heuristics to guess type
    - finds regions of dense text by using morphological operations to detect horizontal strips (text blocks)
    Returns list of dicts: {type, confidence, description, bounds}
    """
    results = []
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Blur + adaptive threshold to find shapes and text regions
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert for contours (white objects on black bg)
    th_inv = 255 - th

    # Morph to close shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,7))
    closed = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        area = wc * hc
        if area < 0.002 * (w*h):  # ignore tiny things
            continue
        aspect = wc / float(hc + 1e-8)
        # classify heuristically
        if 1.8 <= aspect <= 8.0 and hc < 0.15 * h:
            typ = "button"
            conf = 0.7 + min(0.25, (aspect-1.8)/6.2)
            desc = "Rectangular horizontal element — likely a button"
        elif aspect < 1.2 and area > 0.01 * (w*h):
            typ = "card"
            conf = 0.7
            desc = "Square-ish large region — likely a card or panel"
        elif hc < 0.12 * h and wc > 0.4 * w:
            typ = "text_block"
            conf = 0.6
            desc = "Wide horizontal strip — likely a heading / text block"
        else:
            typ = "panel"
            conf = 0.45
            desc = "Generic panel/container"
        results.append({
            "type": typ,
            "confidence": round(float(conf), 2),
            "description": desc,
            "bounds": {"x": int(x), "y": int(y), "w": int(wc), "h": int(hc)}
        })

    # Optional: try to find small circular icons using HoughCircles
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=20, minRadius=6, maxRadius=60)
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = map(int, c)
            results.append({
                "type": "icon",
                "confidence": 0.55,
                "description": f"Small circular element (icon) radius {r}",
                "bounds": {"x": cx-r, "y": cy-r, "w": 2*r, "h": 2*r}
            })
    # Sort by confidence desc
    results = sorted(results, key=lambda r: -r["confidence"])
    return results

@app.post("/detect-ui-elements")
def detect_ui_elements(req: DetectRequest):
    if not req.image:
        raise HTTPException(status_code=400, detail="Missing 'image' field (base64 string or image url).")
    # load image
    if req.image.strip().startswith("http://") or req.image.strip().startswith("https://"):
        pil = load_image_from_url(req.image.strip())
    else:
        pil = load_image_from_base64(req.image.strip())
    img_bgr = pil_to_cv2(pil)
    elements = detect_ui_elements_cv2(img_bgr)
    return {"elements": elements}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
