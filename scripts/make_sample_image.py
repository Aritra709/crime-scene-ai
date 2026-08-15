"""Generate a synthetic 'crime scene' photo for the offline demo.

Creates: dark concrete floor, a dark-red stain pool, a knife-like object,
a bottle, and text graffiti — enough for the stain heuristic (and YOLO, if
installed) to produce meaningful detections. Saves as scripts/sample_scene.jpg
with fake EXIF GPS + timestamp so the metadata path is exercised too.
"""

import struct
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "sample_scene.jpg"


def add_exif(path: Path, lat: float, lng: float, when: str):
    """Minimal EXIF injection: DateTimeOriginal + GPS (2-byte rationals)."""
    # Use piexif if available, otherwise skip EXIF injection for demo
    try:
        import piexif
        from PIL import Image

        img = Image.open(path)

        def dms(value):
            deg = int(value)
            minute = int((value - deg) * 60)
            sec = ((value - deg) * 60 - minute) * 60
            return ((deg, 1), (minute, 1), (int(sec * 1000), 1000))

        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: dms(abs(lat)),
            piexif.GPSIFD.GPSLongitudeRef: "E",
            piexif.GPSIFD.GPSLongitude: dms(abs(lng)),
        }

        exif_dict = {
            "0th": {piexif.ImageIFD.DateTime: when},
            "GPS": gps_ifd,
        }
        exif_bytes = piexif.dump(exif_dict)

        img.save(path, exif=exif_bytes)
        print(f"EXIF injected: GPS {lat},{lng} @ {when}")
    except ImportError:
        print("piexif not installed; skipping EXIF injection (demo still works)")


def main():
    rng = np.random.default_rng(7)
    h, w = 720, 1280

    base = rng.uniform(42, 60, (h, w)).astype(np.uint8)
    base = cv2.merge([base, (base * 1.05).astype(np.uint8), (base * 0.92).astype(np.uint8)])
    noise = rng.normal(0, 3, (h, w, 1)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def blob(center, axes, color, alpha):
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        layer = np.zeros((h, w, 3), np.uint8)
        layer[:] = color
        overlay = cv2.addWeighted(img, 1, layer, 1, 0)
        img[:, :] = np.where(mask[..., None] > 90, overlay, img)

    # dark red stain pool (splatter-ish) near center  (BGR: red channel high)
    blob((640, 380), (260, 120), (24, 12, 96), 0.9)
    blob((560, 330), (60, 40), (30, 18, 120), 0.9)
    blob((740, 440), (40, 30), (26, 14, 110), 0.9)

    # knife: gray blade with highlight + dark handle with rivets
    pts = np.array([[280, 540], [620, 470], [620, 490], [280, 560]], np.int32)
    cv2.fillPoly(img, [pts], (168, 168, 172))
    highlight = np.array([[300, 540], [600, 478], [600, 484], [300, 546]], np.int32)
    cv2.fillPoly(img, [highlight], (215, 215, 220))
    cv2.rectangle(img, (260, 552), (320, 578), (46, 42, 38), -1)
    cv2.rectangle(img, (262, 554), (272, 576), (120, 112, 104), -1)  # rivet
    cv2.rectangle(img, (292, 554), (302, 576), (120, 112, 104), -1)  # rivet
    cv2.line(img, (620, 470), (650, 462), (190, 188, 190), 3)        # tip

    # discarded bottle
    cv2.rectangle(img, (930, 520), (952, 640), (30, 110, 60), -1)
    cv2.rectangle(img, (928, 500), (954, 524), (20, 90, 50), -1)

    # graffiti text (OCR may pick up) — bright red in BGR
    cv2.putText(img, "STAY AWAY", (820, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 200), 3)

    cv2.imwrite(str(OUT), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {OUT} ({w}x{h})")

    when = datetime(2026, 8, 10, 14, 32, 5).strftime("%Y:%m:%d %H:%M:%S")
    add_exif(OUT, 28.6139, 77.2090, when)


if __name__ == "__main__":
    main()