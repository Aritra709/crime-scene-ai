import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .. import config
from ..pipeline import run_pipeline
from .. import reasoning
from ..vision import detector, ocr, video

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_photo(
    image: UploadFile,
    officer_id: str = Form(""),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
):
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")

    # Validate file size
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file too large (max {config.MAX_UPLOAD_MB} MB)")

    # Validate MIME type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if image.content_type not in allowed_types:
        raise HTTPException(400, f"unsupported file type: {image.content_type}")

    image_id = uuid.uuid4().hex[:12]
    image_path = config.IMAGES_DIR / f"{image_id}.jpg"

    try:
        analysis = run_pipeline(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Run reasoning layer
    llm_result = reasoning.reason(analysis)

    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(image_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    captured_at = analysis["metadata"].get("captured_at")
    gps = analysis["metadata"].get("gps")
    if gps is None and lat is not None and lng is not None:
        gps = {"lat": lat, "lng": lng}
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).isoformat()

    analysis.update({
        "image_id": image_id,
        "image_url": f"/api/images/{image_id}",
        "officer_id": officer_id,
        "gps": gps,
        "captured_at": captured_at,
        "llm": llm_result,
    })
    analysis["metadata"] = {
        **analysis["metadata"],
        "gps": gps,
        "captured_at": captured_at,
        "officer_id": officer_id,
    }
    return JSONResponse(analysis)


@router.post("/upload-video")
async def upload_video(
    video: UploadFile,
    officer_id: str = Form(""),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
):
    raw = await video.read()
    if not raw:
        raise HTTPException(400, "empty video upload")

    # Validate file size
    if len(raw) > config.MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(413, f"file too large (max {config.MAX_VIDEO_MB} MB)")

    # Validate MIME type
    allowed_types = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"}
    if video.content_type not in allowed_types:
        raise HTTPException(400, f"unsupported file type: {video.content_type}")

    video_id = uuid.uuid4().hex[:12]
    video_path = config.VIDEOS_DIR / f"{video_id}.mp4"

    # Save video file
    with open(video_path, "wb") as f:
        f.write(raw)

    try:
        detections, notes = video.process_video(video_path)
    except Exception as exc:
        raise HTTPException(400, f"video processing failed: {exc}") from exc

    # Create analysis dict similar to image pipeline
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Separate detections by type
    objects = [d for d in detections if d["category"] not in ("stain", "ocr")]
    stains_list = [d for d in detections if d["category"] == "stain"]
    ocr_items = [d for d in detections if d["category"] == "ocr"]

    analysis = {
        "width": width,
        "height": height,
        "objects": objects,
        "stains": stains_list,
        "ocr": ocr_items,
        "tamper": {"ela_score": 0, "threshold": config.ELA_THRESHOLD, "flag": "not_applicable", "notes": ["ELA not supported for video"]},
        "metadata": {"gps": None, "captured_at": None, "has_exif": False, "notes": notes},
        "processing_notes": notes,
    }

    # Run reasoning layer
    llm_result = reasoning.reason(analysis)

    captured_at = datetime.now(timezone.utc).isoformat()
    gps = {"lat": lat, "lng": lng} if lat is not None and lng is not None else None

    analysis.update({
        "video_id": video_id,
        "video_url": f"/api/videos/{video_id}",
        "officer_id": officer_id,
        "gps": gps,
        "captured_at": captured_at,
        "llm": llm_result,
    })
    analysis["metadata"] = {
        **analysis["metadata"],
        "gps": gps,
        "captured_at": captured_at,
        "officer_id": officer_id,
    }
    return JSONResponse(analysis)


@router.get("/videos/{video_id}")
async def get_video(video_id: str):
    video_path = config.VIDEOS_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        raise HTTPException(404, "video not found")
    from fastapi.responses import FileResponse
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/health")
async def health():
    det = detector._get_detector()
    ocr_engine = ocr._ENGINE

    return {
        "status": "ok",
        "yolo": "available" if det.available else f"unavailable: {det.reason}",
        "ocr": "available" if ocr_engine.available else f"unavailable: {ocr_engine.reason}",
        "llm": "openai" if config.OPENAI_API_KEY else "mock",
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "max_video_mb": config.MAX_VIDEO_MB,
        "video_frame_interval": config.VIDEO_FRAME_INTERVAL,
        "video_max_frames": config.VIDEO_MAX_FRAMES,
    }