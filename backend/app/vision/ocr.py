"""OCR for visible text / plates / notes — three tiers, degrades cleanly.

Tier 1: Tesseract via pytesseract (local binary) — used when available.
Tier 2: OCR.Space remote API (free tier, pure urllib+base64, no native deps).
        Used when Tesseract is missing but OCR_SPACE_API_KEY is set.
        NOTE: the photo is uploaded to a third party — triage only.
Tier 3: unavailable — reported in processing_notes, never a crash.

Every degradation is surfaced as a note (vision modules never fail silently).
"""

import base64
import json
import os
import urllib.request

import cv2

from .. import config

_WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "") or (
    _WIN_DEFAULT if os.path.exists(_WIN_DEFAULT) else "tesseract"
)

_OCR_SPACE_URL = "https://api.ocr.space/parse/image"
_MAX_UPLOAD_BYTES = 1_000_000  # free-tier payload ceiling (~1 MB)
_MAX_SIDE_PX = 2000


class _LocalOcr:
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


class _RemoteOcr:
    """OCR.Space free tier. Coordinates come back in the sent image's space."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def read(self, img_bgr) -> tuple[list, list[str]]:
        try:
            h, w = img_bgr.shape[:2]
            inv_scale = 1.0
            if max(w, h) > _MAX_SIDE_PX:
                s = _MAX_SIDE_PX / max(w, h)
                inv_scale = 1.0 / s
                img_bgr = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))),
                                     interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".png", img_bgr)
            data_uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
            if len(data_uri) > _MAX_UPLOAD_BYTES:
                _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            body = {
                "apikey": self.api_key,
                "language": "eng",
                "isOverlayRequired": "true",
                "OCREngine": "2",
                "scale": "true",
                "base64Image": data_uri,
            }
            req = urllib.request.Request(
                _OCR_SPACE_URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            if payload.get("IsErroredOnProcessing"):
                return [], [f"ocr-error: OCR.Space: {payload.get('ErrorMessage', 'unknown')}"]
            parsed = (payload.get("ParsedResults") or [None])[0]
            if not parsed:
                return [], ["ocr-error: OCR.Space returned no parsed results"]
            out = []
            for line in parsed.get("TextOverlay", {}).get("Lines", []):
                words = [wd for wd in line.get("Words", []) if (wd.get("WordText") or "").strip()]
                if not words:
                    continue
                xs = [wd["Left"] for wd in words]
                ys = [wd["Top"] for wd in words]
                x2s = [wd["Left"] + wd["Width"] for wd in words]
                y2s = [wd["Top"] + wd["Height"] for wd in words]
                out.append({
                    "id": f"o{len(out)}",
                    "text": " ".join(wd["WordText"].strip() for wd in words),
                    "confidence": None,  # free tier does not report confidence
                    "bbox": {"x1": float(min(xs) * inv_scale), "y1": float(min(ys) * inv_scale),
                             "x2": float(max(x2s) * inv_scale), "y2": float(max(y2s) * inv_scale)},
                    "source": "ocr-space",
                })
            if not out and parsed.get("ParsedText", "").strip():
                out.append({"id": "o0", "text": parsed["ParsedText"].strip(),
                            "confidence": None, "bbox": None, "source": "ocr-space"})
            return out, []
        except Exception as exc:
            return [], [f"ocr-error: {exc}"]


class OcrEngine:
    def __init__(self):
        self.local = _LocalOcr()
        self.remote = _RemoteOcr(config.OCR_SPACE_API_KEY) if config.OCR_SPACE_API_KEY else None
        self.available = self.local.available or self.remote is not None
        self.reason = self.local.reason

    def read(self, img_bgr) -> tuple[list, list[str]]:
        if self.local.available:
            return self.local.read(img_bgr)
        if self.remote is not None:
            out, notes = self.remote.read(img_bgr)
            if out:
                notes.append("OCR ran via OCR.Space remote API — the photo is uploaded to a "
                             "third party; triage only, not for real evidence")
            return out, notes
        return [], ["ocr-unavailable: no local Tesseract and no OCR_SPACE_API_KEY "
                    "(remote OCR fallback) configured"]


_ENGINE = OcrEngine()


def read_text(img_bgr) -> tuple[list, list[str]]:
    return _ENGINE.read(img_bgr)
