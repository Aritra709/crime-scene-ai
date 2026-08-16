"""Video processing: frame extraction, detection, and person tracking."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from .detector import detect_objects
from .. import config


def extract_frames(video_path: Path, interval: int = None, max_frames: int = None) -> Tuple[List[np.ndarray], List[int], str]:
    """Extract frames from video at regular intervals.
    Returns (frames, frame_numbers, note)."""
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


def compute_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def track_persons_across_frames(all_detections: List[List[Dict]], iou_threshold: float = 0.3) -> List[Dict]:
    """Track persons across frames using simple IoU matching.
    Returns detections with person_id assigned (person1, person2, etc.)."""
    person_tracks = {}  # track_id -> {last_bbox, last_frame_idx, person_label}
    next_track_id = 1

    for frame_idx, frame_dets in enumerate(all_detections):
        person_dets = [d for d in frame_dets if d["category"] == "person"]

        for det in person_dets:
            best_iou = 0.0
            best_track_id = None

            for track_id, track in person_tracks.items():
                iou = compute_iou(det["bbox"], track["last_bbox"])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                person_tracks[best_track_id]["last_bbox"] = det["bbox"]
                person_tracks[best_track_id]["last_frame_idx"] = frame_idx
                det["person_id"] = person_tracks[best_track_id]["person_label"]
            else:
                label = f"person{next_track_id}"
                person_tracks[next_track_id] = {
                    "last_bbox": det["bbox"],
                    "last_frame_idx": frame_idx,
                    "person_label": label
                }
                det["person_id"] = label
                next_track_id += 1

        # Update class label for person detections
        for det in person_dets:
            if "person_id" in det:
                det["class"] = det["person_id"]

    # Flatten all detections
    result = []
    for frame_dets in all_detections:
        result.extend(frame_dets)

    return result


def process_video(video_path: Path) -> Tuple[List[Dict], List[str]]:
    """Process video: extract frames, run detection, track persons.
    Returns (detections, notes)."""
    frames, frame_numbers, note = extract_frames(video_path)
    notes = [note] if note else []

    all_detections = []
    for i, frame in enumerate(frames):
        dets, det_notes = detect_objects(frame)
        notes.extend(det_notes)
        for det in dets:
            det["frame_idx"] = frame_numbers[i]
        all_detections.append(dets)

    # Track persons across frames
    tracked = track_persons_across_frames(all_detections)

    # Add video source info
    for det in tracked:
        det["source"] = "yolo-video"
        if "frame_idx" in det:
            det["basis"] = det.get("basis", []) + [f"frame {det['frame_idx']}"]

    return tracked, notes