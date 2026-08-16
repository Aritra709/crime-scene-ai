import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "cases.db"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY", "")

YOLO_MODEL_NAME = os.environ.get("YOLO_MODEL_NAME", "yolov8n.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.35"))

STAIN_MIN_AREA_PCT = float(os.environ.get("STAIN_MIN_AREA_PCT", "0.02"))
STAIN_MAX_AREA_PCT = float(os.environ.get("STAIN_MAX_AREA_PCT", "60.0"))
ELA_THRESHOLD = float(os.environ.get("ELA_THRESHOLD", "3.5"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_MB", "50"))
VIDEO_FRAME_INTERVAL = int(os.environ.get("VIDEO_FRAME_INTERVAL", "30"))
VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "100"))
VIDEOS_DIR = DATA_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
