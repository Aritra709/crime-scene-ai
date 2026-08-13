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
