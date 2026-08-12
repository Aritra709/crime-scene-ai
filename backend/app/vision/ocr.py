"""OCR for visible text / plates / notes — OPTIONAL (needs Tesseract binary).

Degrades cleanly: when pytesseract or the tesseract binary is missing, the
engine reports availability=False and the pipeline notes it rather than failing.
Model choice is you-the-practitioner's call at deployment (EasyOCR works
offline too and handles Devanagari better than Tesseract for field use).
"""

import os

import cv2

_WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "") or (
    _WIN_DEFAULT if os.path.exists(_WIN_DEFAULT) else "tesseract"
)


class OcrEngine:
    def __init__(self):
        self.available = False
        self.reason = ""
        try:
            import pytesseract  # noqa: F401

            if os.path.exists(TESSERACT_CMD):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
                self.available = bool(pytesseract.get_tesseract_version())
            else:
                self.reason = f"ocr-unavailable: tesseract binary not found at {TESSERACT_CMD}"
        except Exception as exc:
            self.reason = str(exc)

    def read(self, img_bgr) -> tuple[list, list[str]]:
        if not self.available:
            return [], ["ocr-unavailable: pytesseract and/or Tesseract binary not installed"]
        import pytesseract

        try:
            data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT)
            out = []
            n = len(data["text"])
            for i in range(n):
                text = (data["text"][i] or "").strip()
                conf = data["conf"][i]
                if len(text) < 2 or conf < 40:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                if w <= 0 or h <= 0:
                    continue
                out.append({
                    "id": f"o{len(out)}",
                    "text": text,
                    "confidence": round(float(conf) / 100.0, 3),
                    "bbox": {"x1": float(x), "y1": float(y), "x2": float(x + w), "y2": float(y + h)},
                    "source": "ocr",
                })
            return out, []
        except Exception as exc:
            return [], [f"ocr-error: {exc}"]


_ENGINE = OcrEngine()


def read_text(img_bgr) -> tuple[list, list[str]]:
    return _ENGINE.read(img_bgr)