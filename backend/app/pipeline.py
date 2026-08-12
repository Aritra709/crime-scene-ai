"""Pipeline orchestration: image bytes → structured, explainable analysis.

Runs vision modules, collects every degradation note, then hands a
structured summary to the reasoning layer. The result is always a dict
consumable by the React draft editor — and every component that was missing
is listed in processing_notes (no silent degradation).
"""

import cv2
import numpy as np

from .vision import detector, ocr, stains, tamper
from .reasoning import reason


def run_pipeline(raw: bytes) -> dict:
    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    notes: list[str] = []
    if img_bgr is None:
        raise ValueError("image could not be decoded")

    objects, det_notes = detector.detect_objects(img_bgr)
    notes.extend(det_notes)

    stain_list = stains.detect_blood_like_stains(img_bgr)
    if not stain_list:
        notes.append("no red-dominant blobs above 0.02% area — no stain candidates")

    ocr_list, ocr_notes = ocr.read_text(img_bgr)
    notes.extend(ocr_notes)
    if ocr_list:
        notes.append(f"OCR extracted {len(ocr_list)} text regions")

    ela = tamper.ela_check(img_bgr)
    meta = tamper.read_exif(raw)

    analysis = {
        "width": int(img_bgr.shape[1]),
        "height": int(img_bgr.shape[0]),
        "objects": objects,
        "stains": stain_list,
        "ocr": ocr_list,
        "tamper": ela,
        "metadata": meta,
        "processing_notes": notes,
    }

    llm = reason(analysis)
    analysis["llm"] = llm
    if llm.get("source") == "mock":
        notes.append("LLM reasoning ran in offline mode (set OPENAI_API_KEY for API-based reasoning)")
    else:
        notes.append(f"LLM reasoning via API: {llm.get('model', 'unknown')}")
    analysis["processing_notes"] = notes
    return analysis