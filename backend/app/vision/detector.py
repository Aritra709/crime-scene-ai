"""Object detection via YOLOv8n — OPTIONAL, degrades to offline mode.

Two interchangeable engines, same output schema:
- ultralytics (PyTorch) when installed;
- onnxruntime fallback with the bundled `yolov8n.onnx` (lightweight — used on
  CPU-only cloud hosts where installing torch is impractical).

Maps COCO classes to crime-relevant categories so the reasoning layer gets a
small, consistent vocabulary. Every detection carries: class, category,
pixel bbox, confidence and source. No detections is a valid answer — the
pipeline reports 'yolo-unavailable' as a processing note instead of guessing.
"""

import cv2
import numpy as np

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

_COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

_IMGSZ = 640
_PAD_COLOR = 114


def _letterbox(img_bgr):
    """Resize keeping aspect ratio, pad to square with gray. Returns lb, scale, dx, dy."""
    h, w = img_bgr.shape[:2]
    r = min(_IMGSZ / w, _IMGSZ / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((_IMGSZ, _IMGSZ, 3), _PAD_COLOR, dtype=np.uint8)
    dx, dy = (_IMGSZ - nw) // 2, (_IMGSZ - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, r, dx, dy


def _nms(boxes, scores, iou_thr):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a_i + a_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def detect_onnx(img_bgr, conf: float, iou: float) -> tuple[list, str]:
    """YOLOv8n inference via onnxruntime. Returns (detections, note)."""
    import onnxruntime as ort

    onnx_path = config.BASE_DIR / "yolov8n.onnx"
    if not onnx_path.exists():
        return [], "yolo-unavailable: yolov8n.onnx not found"
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp_name = session.get_inputs()[0].name

        lb, scale, dx, dy = _letterbox(img_bgr)
        blob = lb.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0
        out = session.run(None, {inp_name: blob})[0][0]  # [84, 8400]

        cx, cy, w, h = out[:4]  # decoded xywh in 640-space
        prob = out[4:]          # per-class probabilities (sigmoid'd by graph)
        best = prob.max(axis=0)
        cids = prob.argmax(axis=0)
        mask = best >= conf
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)[mask]
        scores = best[mask]
        cids = cids[mask]
        if not boxes.size:
            return [], ""
    except Exception as exc:
        return [], f"yolo-unavailable: onnx inference failed: {exc}"

    dets = []
    keep = _nms(boxes, scores, iou)
    for b, s, c in zip(boxes[keep], scores[keep], cids[keep]):
        ci = int(c)
        name = _COCO_NAMES[ci] if ci < len(_COCO_NAMES) else ""
        mapped = CRIME_MAP.get(name)
        if mapped is None:
            continue  # COCO class with no crime relevance — skip silently
        label, category = mapped
        dets.append({
            "id": f"y{len(dets)}",
            "class": label,
            "category": category,
            "confidence": round(float(s), 3),
            "bbox": {
                "x1": float((b[0] - dx) / scale),
                "y1": float((b[1] - dy) / scale),
                "x2": float((b[2] - dx) / scale),
                "y2": float((b[3] - dy) / scale),
            },
            "source": "yolo",
            "basis": [f"{name} detected by YOLOv8n (ONNX Runtime), conf {s:.2f}"],
        })
    dets.sort(key=lambda d: -d["confidence"])
    return dets, ""


class Detector:
    def __init__(self):
        self.available = False
        self.reason = ""
        self.model = None
        self.engine = "torch"
        try:
            from ultralytics import YOLO  # heavy import — only when installed

            self.model = YOLO(config.YOLO_MODEL_NAME)
            self.available = True
        except Exception as exc:  # no ultralytics / torch, or offline weights download
            self.engine = "onnx"
            self.reason = str(exc)
            try:
                import onnxruntime as ort  # noqa: F401

                if not (config.BASE_DIR / "yolov8n.onnx").exists():
                    raise FileNotFoundError("yolov8n.onnx not found")
                self.available = True
            except Exception as exc2:
                self.available = False
                self.reason = f"{self.reason} | onnx fallback failed: {exc2}"

    def detect(self, img_bgr) -> tuple[list, str]:
        """Returns (detections, note). Source is always 'yolo' for real detections."""
        if not self.available:
            return [], f"yolo-unavailable: {self.reason}"
        if self.engine == "onnx":
            return detect_onnx(img_bgr, config.YOLO_CONF, 0.45)
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


def _get_detector() -> "Detector":
    if not hasattr(_get_detector, "_instance"):
        _get_detector._instance = Detector()
    return _get_detector._instance


def detect_objects(img_bgr) -> tuple[list, list[str]]:
    dets, note = _get_detector().detect(img_bgr)
    notes = [note] if note else []
    return dets, notes