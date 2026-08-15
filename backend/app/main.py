import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .routers import cases, upload

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

app = FastAPI(title="Crime Scene AI — Smart Evidence Capture Assistant", version="0.1.0")

# CORS: allow frontend origin in production, all in dev
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(cases.router)

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")


@app.get("/api/images/{image_id}")
async def serve_image(image_id: str):
    path = config.IMAGES_DIR / f"{image_id}.jpg"
    if not path.exists():
        from fastapi import HTTPException

        raise HTTPException(404, "image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.on_event("startup")
def _startup():
    db.init_db()