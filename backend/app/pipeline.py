"""Pipeline orchestration: image bytes → structured, explainable analysis.

Runs the vision modules (object detection, stain candidates, OCR, tamper/EXIF)
and collects every degradation note. Reasonings runs with an OpenAI-compatible
key when configured, otherwise as an offline rule-based draft (no key needed);
every missing component is still surfaced in processing_notes (no silent
degradation). The app never uploads photos off-device.
"""

import cv2
import numpy as np

from .vision import detector, stains, tamper, ocr


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

    ocr_items, ocr_notes = ocr.read_text(img_bgr)
    notes.extend(ocr_notes)

    ela = tamper.ela_check(img_bgr)
    meta = tamper.read_exif(raw)

    analysis = {
        "width": int(img_bgr.shape[1]),
        "height": int(img_bgr.shape[0]),
        "objects": objects,
        "stains": stain_list,
        "ocr": ocr_items,
        "tamper": ela,
        "metadata": meta,
        "processing_notes": notes,
    }
    return analysis