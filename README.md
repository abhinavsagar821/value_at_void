# UI Element Detector — README

## Overview

This project implements a prototype API that detects UI elements (buttons, cards, text blocks, icons, panels) in screenshots and returns a structured JSON describing each detected element.

Main features:

* `POST /detect-ui-elements` endpoint
* Accepts image as base64 string or image URL
* Returns elements with `type`, `description`, `confidence`, and `bounds` (`x,y,w,h`)
* Modular detector interface so you can swap heuristic detector for ML models (Grounding DINO, SAM, YOLO, etc.)

## Prototype Architecture

* **Backend:** Python, FastAPI
* **Detector (prototype):** OpenCV heuristic (contour shapes + Hough circles)
* **Future detector options:** Grounding DINO (open-vocabulary detection), SAM for masks, CLIP/GPT for label refinement

## JSON output schema (example)

```json
{
  "elements": [
    {
      "type": "button",
      "confidence": 0.91,
      "description": "Primary CTA button in the top-left section",
      "bounds": { "x": 100, "y": 40, "w": 180, "h": 50 }
    }
  ]
}
```

## How the prototype detector works

* Converts image to grayscale
* Uses thresholding + morphological close to find rectangular shapes
* Extracts contours and heuristically classifies by aspect ratio and area
* Detects circular shapes using HoughCircles as icons

## Trade-offs and limitations

* **Heuristic detector** is fast and requires no ML weights, but:

  * Not robust to diverse UI styles
  * More false positives/negatives than model-based detectors
* **Model-based detectors** (Grounding DINO + SAM):

  * Better accuracy and open-vocabulary detection
  * Require GPU and heavier dependencies
  * More setup time (model weights, GPU infra)

## How to run locally

1. Create venv & install:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn pillow numpy opencv-python requests
   ```
2. Run:

   ```bash
   uvicorn app:app --reload --port 8000
   ```
3. Endpoint:

   * `POST /detect-ui-elements`
   * Body: `{"image": "<base64-or-url>"}`
   * Response: JSON with `elements` list

## How to replace the detector with a model (recommended next steps)

1. Add a detector wrapper class `ModelDetector` that exposes `detect(image: np.ndarray) -> List[Element]`.
2. Integrate SOTA detectors:

   * **Grounding DINO**: for open-vocabulary detection (gives boxes for textual queries like "button", "card", "icon").
   * **SAM**: use boxes from Grounding DINO as prompts to get precise masks and refined boxes.
   * **CLIP/GPT**: run cropped regions through CLIP or a small classifier to refine types and generate richer descriptions. For natural-language descriptions, call a multimodal LLM.
3. Run inference on GPU; add caching and batch processing for scale.

## What I would improve with more time

* Fine-tune a detector on a curated UI dataset (Figma snapshots, RICO, ORB, internal datasets).
* Add a small frontend to upload screenshots and visualize bounding boxes.
* Quantitative benchmarks (precision/recall on a labeled test set).
* Better heuristics for nested components (menus, toolbars) and alignment-aware grouping.

## Deliverable checklist

* `app.py` — FastAPI prototype
* `/samples/` — place input images and output JSONs there
* `README.md` — this file
* Optional: UI + deployment (Railway/Vercel) if desired

