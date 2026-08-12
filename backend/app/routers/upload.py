import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .. import config
from ..pipeline import run_pipeline

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

    image_id = uuid.uuid4().hex[:12]
    image_path = config.IMAGES_DIR / f"{image_id}.jpg"

    try:
        analysis = run_pipeline(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

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
    })
    analysis["metadata"] = {
        **analysis["metadata"],
        "gps": gps,
        "captured_at": captured_at,
        "officer_id": officer_id,
    }
    return JSONResponse(analysis)


@router.get("/health")
async def health():
    return {"status": "ok"}