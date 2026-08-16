"""Video processing: frame extraction, detection, and object tracking with ByteTrack-style tracker."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from .detector import detect_objects
from .. import config


# ByteTrack-style Kalman filter for bbox tracking
class KalmanBoxTracker:
    """Simple Kalman filter for bbox tracking (ByteTrack style)."""
    count = 0

    def __init__(self, bbox: np.ndarray, cls_id: int, conf: float):
        """
        State: [x, y, w, h, vx, vy, vw, vh]  (center x, center y, width, height, velocities)
        Measurement: [x, y, w, h]
        """
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # State transition matrix
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i + 4] = 1
        # Measurement matrix
        self.kf.H = np.zeros((4, 8))
        for i in range(4):
            self.kf.H[i, i] = 1
        self.kf.R[2:, 2:] *= 10
        self.kf.P[4:, 4:] *= 1000
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Convert xyxy to xywh (center, width, height)
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.kf.x[:4] = np.array([[cx], [cy], [w], [h]])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls_id = cls_id
        self.conf = conf

    def update(self, bbox: np.ndarray, conf: float = 0):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.conf = max(self.conf, conf)
        # Convert xyxy to xywh
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.kf.update(np.array([[cx], [cy], [w], [h]]))

    def predict(self):
        self.kf.predict()
        if self.kf.x[2] <= 0:
            self.kf.x[2] = 0
        if self.kf.x[3] <= 0:
            self.kf.x[3] = 0
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        cx = float(self.kf.x[0, 0])
        cy = float(self.kf.x[1, 0])
        w = float(self.kf.x[2, 0])
        h = float(self.kf.x[3, 0])
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        state = np.array([x1, y1, x2, y2])
        self.history.append(state)
        return state

    def get_state(self) -> np.ndarray:
        cx = float(self.kf.x[0, 0])
        cy = float(self.kf.x[1, 0])
        w = float(self.kf.x[2, 0])
        h = float(self.kf.x[3, 0])
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        return np.array([x1, y1, x2, y2])


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bboxes."""
    if bboxes1.size == 0 or bboxes2.size == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]))
    bboxes1 = bboxes1.reshape(-1, 4)
    bboxes2 = bboxes2.reshape(-1, 4)
    x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0])
    y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1])
    x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2])
    y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / union


def associate_detections_to_trackers(
    dets: np.ndarray, trks: np.ndarray, iou_thresh: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian algorithm assignment (simplified greedy)."""
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty(0, dtype=int)
    if len(dets) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int), np.arange(len(trks))

    iou_matrix = iou_batch(dets, trks)
    matched_indices = []
    unmatched_dets = list(range(len(dets)))
    unmatched_trks = list(range(len(trks)))

    # Greedy matching by highest IoU
    if iou_matrix.size > 0:
        for _ in range(min(len(dets), len(trks))):
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_val = iou_matrix[max_idx]
            if max_val < iou_thresh:
                break
            d, t = max_idx
            matched_indices.append([d, t])
            unmatched_dets.remove(d)
            unmatched_trks.remove(t)
            iou_matrix[d, :] = -1
            iou_matrix[:, t] = -1

    matches = np.array(matched_indices, dtype=int) if matched_indices else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)


class ByteTracker:
    """ByteTrack-style multi-object tracker for all classes."""

    def __init__(
        self,
        iou_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray, cls_ids: np.ndarray, confs: np.ndarray) -> List[Dict]:
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos[:4]
            if trk.time_since_update > self.max_age:
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_thresh
        )

        for m in matches:
            self.trackers[m[1]].update(dets[m[0]], confs[m[0]])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i], cls_ids[i], confs[i])
            # Call update to initialize hits properly
            trk.update(dets[i], confs[i])
            self.trackers.append(trk)

        ret = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits and trk.time_since_update == 0:
                state = trk.get_state()
                x1, y1, x2, y2 = state
                ret.append({
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "track_id": trk.id,
                    "cls_id": trk.cls_id,
                    "confidence": float(trk.conf),
                })
        return ret


def extract_frames(
    video_path: Path, interval: int = None, max_frames: int = None
) -> Tuple[List[np.ndarray], List[int], str]:
    """Extract frames from video at regular intervals."""
    interval = interval or config.VIDEO_FRAME_INTERVAL
    max_frames = max_frames or config.VIDEO_MAX_FRAMES

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], "video-unavailable: could not open video file"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_numbers = []
    frame_idx = 0
    extracted = 0

    while extracted < max_frames and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_numbers.append(frame_idx)
        extracted += 1
        frame_idx += interval

    cap.release()

    if not frames:
        return [], [], "video-unavailable: no frames extracted"

    return frames, frame_numbers, ""


def label_persons(dets: List[Dict]) -> List[Dict]:
    """Label person detections in a single photo as person1, person2, ... sorted by position."""
    persons = [d for d in dets if d.get("category") == "person"]
    persons.sort(key=lambda d: (d["bbox"]["x1"], d["bbox"]["y1"]))
    for i, d in enumerate(persons, 1):
        label = f"person{i}"
        d["person_id"] = label
        d["class"] = label
    return dets


def _dets_to_arrays(dets: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert detection list to arrays for tracker."""
    if not dets:
        return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0)
    bboxes = np.array([[d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]] for d in dets])
    cls_ids = np.array([_cls_name_to_id(d.get("category", "")) for d in dets])
    confs = np.array([d.get("confidence", 0) for d in dets])
    return bboxes, cls_ids, confs


_CLS_MAP = {
    "person": 0, "weapon": 1, "vehicle": 2, "container": 3,
    "personal item": 4, "electronic device": 5, "discarded item": 6,
    "stain": 7, "ocr": 8,
}

def _cls_name_to_id(name: str) -> int:
    return _CLS_MAP.get(name, 99)


def _cls_id_to_name(cls_id: int) -> str:
    for k, v in _CLS_MAP.items():
        if v == cls_id:
            return k
    return "unknown"


def process_video(video_path: Path) -> Tuple[List[Dict], List[str]]:
    """Process video: extract frames, run detection, track objects across frames."""
    frames, frame_numbers, note = extract_frames(video_path)
    notes = [note] if note else []

    tracker = ByteTracker(
        iou_thresh=config.VIDEO_TRACK_IOU_THRESH,
        max_age=config.VIDEO_TRACK_MAX_AGE,
        min_hits=config.VIDEO_TRACK_MIN_HITS,
    )

    all_detections = []
    for i, frame in enumerate(frames):
        dets, det_notes = detect_objects(frame)
        notes.extend(det_notes)
        for det in dets:
            det["frame_idx"] = frame_numbers[i]
        all_detections.append(dets)

        # Track
        bboxes, cls_ids, confs = _dets_to_arrays(dets)
        tracked = tracker.update(bboxes, cls_ids, confs)

        # Merge track info back into detections
        for det in dets:
            det_bbox = np.array([det["bbox"]["x1"], det["bbox"]["y1"], det["bbox"]["x2"], det["bbox"]["y2"]])
            best_iou = 0
            best_track = None
            for trk in tracked:
                trk_bbox = np.array([trk["bbox"]["x1"], trk["bbox"]["y1"], trk["bbox"]["x2"], trk["bbox"]["y2"]])
                iou = iou_batch(det_bbox.reshape(1, 4), trk_bbox.reshape(1, 4))[0, 0]
                if iou > best_iou and iou >= config.VIDEO_TRACK_IOU_THRESH:
                    best_iou = iou
                    best_track = trk
            if best_track:
                det["track_id"] = best_track["track_id"]
                det["track_confidence"] = best_track["confidence"]

    # Track persons across frames with stable IDs
    person_tracks = {}
    next_person_id = 1

    for frame_idx, frame_dets in enumerate(all_detections):
        person_dets = [d for d in frame_dets if d.get("category") == "person"]
        for det in person_dets:
            tid = det.get("track_id")
            if tid is not None:
                if tid not in person_tracks:
                    person_tracks[tid] = f"person{next_person_id}"
                    next_person_id += 1
                det["person_id"] = person_tracks[tid]
                det["class"] = person_tracks[tid]

    # Flatten and add video source info
    tracked_dets = []
    for frame_dets in all_detections:
        for det in frame_dets:
            nd = dict(det)
            nd["source"] = "yolo-video"
            nd["basis"] = nd.get("basis", []) + [f"frame {nd['frame_idx']}"]
            tracked_dets.append(nd)

    tracked_dets.sort(key=lambda d: (d.get("frame_idx", 0), -d.get("confidence", 0)))
    return tracked_dets, notes