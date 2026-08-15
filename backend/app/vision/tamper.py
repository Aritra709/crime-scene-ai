"""Basic forensic-flagging layer: JPEG ELA diff + EXIF sanity check.

Never a verdict — output is framed as "edit likelihood" with an explicit
note that conclusive determination needs forensic tooling. This module only
BUYS a flag for the officer to hand to a forensic examiner.
"""

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from .. import config

ELA_THRESHOLD = config.ELA_THRESHOLD

_GPSTAG_NAMES = {v: k for k, v in GPSTAGS.items()}


def _dms_to_decimal(coord_ref, coord):
    if coord is None:
        return None
    try:
        deg, minute, sec = coord
        ref = (coord_ref or "N").upper()
        sign = -1 if ref in ("S", "W") else 1
        return round(sign * (float(deg) + float(minute) / 60.0 + float(sec) / 3600.0), 6)
    except (TypeError, ValueError):
        return None


def read_exif(raw: bytes) -> dict:
    """Extract GPS + capture time from EXIF; None values are reported honestly."""
    result = {"gps": None, "captured_at": None, "has_exif": False, "notes": []}
    try:
        img = Image.open(__import__("io").BytesIO(raw))
        exif = img.getexif()
        if not exif:
            result["notes"].append("no EXIF block found — camera/timestamp provenance unavailable")
            return result
        result["has_exif"] = True

        for tag, value in exif.items():
            name = TAGS.get(tag, str(tag))
            if name == "DateTimeOriginal":
                result["captured_at"] = str(value)

        gps_ifd = exif.get_ifd(0x8825)
        if gps_ifd:
            lat = _dms_to_decimal(gps_ifd.get(_GPSTAG_NAMES.get("GPSLatitudeRef")),
                                  gps_ifd.get(_GPSTAG_NAMES.get("GPSLatitude")))
            lng = _dms_to_decimal(gps_ifd.get(_GPSTAG_NAMES.get("GPSLongitudeRef")),
                                  gps_ifd.get(_GPSTAG_NAMES.get("GPSLongitude")))
            if lat is not None and lng is not None:
                result["gps"] = {"lat": lat, "lng": lng}
    except Exception as exc:
        result["notes"].append(f"exif-parse-error: {exc}")
    return result


def ela_check(img_bgr) -> dict:
    """Error Level Analysis: re-encode at q90 and diff. Heuristic only."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return {"ela_score": 0.0, "threshold": ELA_THRESHOLD, "flag": "inconclusive",
                "notes": ["re-encode failed — cannot compute ELA"]}
    img2 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img_bgr.astype(np.int16), img2.astype(np.int16))
    score = float(diff.mean())

    if score < ELA_THRESHOLD:
        flag = "clean"
        note = "re-encode error level consistent with a single-save JPEG (no obvious resave/crop signature)"
    else:
        flag = "edit-likelihood"
        note = "elevated error level — consistent with a re-saved/edited JPEG; NOT a verdict"
    return {
        "ela_score": round(score, 3),
        "threshold": ELA_THRESHOLD,
        "flag": flag,
        "notes": [note,
                  "ELA is a heuristic; conclusive tamper determination requires forensic tooling (hash, sensor noise analysis)"],
    }