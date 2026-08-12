# Crime Scene AI — Hugging Face Spaces build (tagged Docker Space)
# Production-stage: builds the React frontend, then serves it + the FastAPI
# backend from a single process on the port HF injects ($PORT, default 7860).

# --- Stage 1: build React frontend ---
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: runtime ---
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf-cache
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt \
    && pip install pytesseract

COPY backend/ backend/
COPY --from=frontend-build /app/frontend/dist frontend/dist

WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]