"""OCR for visible text / plates / notes — degrades cleanly, two engines.

1. Tesseract via pytesseract when the binary is present (full control,
   Devanagari possible with the right language packs).
2. RapidOCR (PP-OCRv4 via onnxruntime) otherwise — pure pip install, no
   system binary needed, so OCR works on hosted clouds too.

When neither engine is usable the pipeline notes it rather than failing.
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
        self.mode = ""
        self.r_engine = None

        try:
            import pytesseract  # noqa: F401

            if os.path.exists(TESSERACT_CMD):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
                if pytesseract.get_tesseract_version():
                    self.mode = "tesseract"
                    self.available = True
        except Exception:
            pass

        if not self.available:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self.r_engine = RapidOCR()
                self.mode = "rapidocr"
                self.available = True
            except Exception as exc:
                self.reason = f"ocr-unavailable: pytesseract/Tesseract and RapidOCR both unavailable: {exc}"

    def read(self, img_bgr) -> tuple[list, list[str]]:
        if not self.available:
            return [], [f"ocr-unavailable: {self.reason or 'no OCR engine usable'}"]

        if self.mode == "rapidocr":
            try:
                result, _ = self.r_engine(img_bgr)
                out = []
                for i, (box, text, score) in enumerate(result or []):
                    xs = [float(p[0]) for p in box]
                    ys = [float(p[1]) for p in box]
                    out.append({
                        "id": f"o{i}",
                        "text": str(text),
                        "confidence": round(float(score), 3),
                        "bbox": {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)},
                        "source": "ocr",
                    })
                return out, []
            except Exception as exc:
                return [], [f"ocr-error: {exc}"]

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