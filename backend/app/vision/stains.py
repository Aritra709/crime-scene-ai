"""Blood-like stain candidate detection — zero-ML heuristic, fully explainable.

Red-dominant HSV mask + morphology, then per-blob confidence built from
visible evidence (area share, color purity, aspect). This is a triage hint:
it cannot tell blood from paint/rust/curry, and the report says so.

Output per blob:
  id, class='blood-like stain (candidate)', category='stain',
  bbox (pixels), confidence, area_pct, source='hsv-heuristic',
  basis=list of human-readable reasons.
"""

import cv2
import numpy as np

from .. import config

MIN_AREA_PCT = config.STAIN_MIN_AREA_PCT
MAX_AREA_PCT = config.STAIN_MAX_AREA_PCT


def detect_blood_like_stains(img_bgr) -> list[dict]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Red hue occupies two ends of the HSV circle (0-10 and 170-180).
    mask = (((h < 10) | (h > 170)) & (s > 60) & (v > 40)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    height, width = img_bgr.shape[:2]
    total = height * width
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = float(cv2.contourArea(cnt))
        area_pct = 100.0 * area / total
        if area_pct < MIN_AREA_PCT or area_pct > MAX_AREA_PCT:
            continue

        roi_mask = mask[y : y + h, x : x + w] > 0
        roi_s, roi_v = s[y : y + h, x : x + w], v[y : y + h, x : x + w]
        sat = float(roi_s[roi_mask].mean()) if roi_mask.sum() else 0.0
        val = float(roi_v[roi_mask].mean()) if roi_mask.sum() else 0.0

        purity = float(np.clip((sat - 60) / 150.0, 0.0, 1.0))
        spread = float(np.clip(area_pct / 5.0, 0.0, 1.0))
        confidence = round(0.40 + 0.35 * purity + 0.25 * spread, 3)

        basis = [
            f"red-dominant region (HSV hue + saturation filter)",
            f"area {area_pct:.2f}% of frame",
            f"mean saturation {sat:.0f}, brightness {val:.0f}",
            "NOT a blood confirmation — red paint/rust/dyes are indistinguishable by color alone",
        ]
        out.append({
            "id": f"s{i}",
            "class": "blood-like stain (candidate)",
            "category": "stain",
            "confidence": confidence,
            "bbox": {"x1": float(x), "y1": float(y), "x2": float(x + w), "y2": float(y + h)},
            "area_pct": round(area_pct, 3),
            "source": "hsv-heuristic",
            "basis": basis,
        })
    out.sort(key=lambda d: -d["area_pct"])
    return out