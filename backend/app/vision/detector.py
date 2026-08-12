"""Object detection via YOLOv8n (ultralytics) — OPTIONAL, degrades to offline mode.

Maps COCO classes to crime-relevant categories so the reasoning layer gets a
small, consistent vocabulary. Every detection carries: class, category,
pixel bbox, confidence and source. No detections is a valid answer — the
pipeline reports 'yolo-unavailable' as a processing note instead of guessing.
"""

from .. import config

CRIME_MAP = {
    "person": ("person", "person"),
    "knife": ("bladed weapon (likely)", "weapon"),
    "scissors": ("sharp object (likely)", "weapon"),
    "bottle": ("bottle", "container"),
    "cup": ("cup", "container"),
    "wine glass": ("glass", "container"),
    "backpack": ("backpack", "personal item"),
    "handbag": ("handbag", "personal item"),
    "suitcase": ("suitcase", "personal item"),
    "cell phone": ("cell phone", "electronic device"),
    "laptop": ("laptop", "electronic device"),
    "car": ("vehicle (car)", "vehicle"),
    "motorcycle": ("vehicle (motorcycle)", "vehicle"),
    "truck": ("vehicle (truck)", "vehicle"),
    "bus": ("vehicle (bus)", "vehicle"),
    "bicycle": ("vehicle (bicycle)", "vehicle"),
    "book": ("book/paper", "discarded item"),
    "baseball bat": ("blunt object (likely)", "weapon"),
    "umbrella": ("umbrella", "discarded item"),
}


class Detector:
    def __init__(self):
        self.available = False
        self.reason = ""
        self.model = None
        try:
            from ultralytics import YOLO  # heavy import — only when installed

            self.model = YOLO(config.YOLO_MODEL_NAME)
            self.available = True
        except Exception as exc:  # no ultralytics / torch, or offline weights download
            self.reason = str(exc)

    def detect(self, img_bgr) -> tuple[list, str]:
        """Returns (detections, note). Source is always 'yolo' for real detections."""
        if not self.available:
            return [], "yolo-unavailable: ultralytics not installed or weights could not be downloaded"
        results = self.model.predict(img_bgr, conf=config.YOLO_CONF, verbose=False)[0]
        names = results.names
        out = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id]
            mapped = CRIME_MAP.get(name)
            if mapped is None:
                continue  # COCO class with no crime relevance (e.g. 'zebra') — skip silently
            label, category = mapped
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            out.append({
                "id": f"y{len(out)}",
                "class": label,
                "category": category,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "source": "yolo",
                "basis": [f"{name} detected by YOLOv8n (COCO), conf {box.conf[0]:.2f}"],
            })
        out.sort(key=lambda d: -d["confidence"])
        return out, ""


_DETECTOR = Detector()


def detect_objects(img_bgr) -> tuple[list, list[str]]:
    dets, note = _DETECTOR.detect(img_bgr)
    notes = [note] if note else []
    return dets, notes