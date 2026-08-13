<div align="center">

# 🕵️ Crime Scene AI

**Smart Evidence Capture Assistant — Smart India Hackathon Demo**

An end-to-end, offline-capable pipeline that turns a crime-scene photo into a
**reviewable draft report**: vision analysis with confidence scores, LLM-generated
narrative (English + Hindi), human-in-the-loop confirmation, chain-of-custody
logging, and pattern matching against past cases.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## ✨ What it does

1. Officer uploads a crime-scene photo (EXIF GPS + timestamp read automatically).
2. **Vision layer** extracts objects, blood-like stains, and text — *every*
   detection carries a **confidence score, source, and basis** (no black-box verdicts).
3. **Reasoning layer** drafts a preliminary observation report (English + Hindi),
   anomaly flags, and next steps.
4. **Officer reviews** — confirms, edits, or rejects every detection before anything is logged.
5. Confirmed report is logged to the case file with **chain-of-custody metadata**
   and matched against **past confirmed cases**.

> ⚖️ **Human-in-the-loop by design**: AI output is always a *draft* — nothing is
> evidence until the officer confirms it.

## 🧠 Design principles

1. **Working end-to-end demo** — runs fully offline with a mock LLM + heuristic vision; YOLO/OCR/LLM-API are plug-in upgrades.
2. **Explainability** — every detection carries a confidence score, source (`yolo` / `hsv-heuristic` / `ocr`) and basis.
3. **Offline / low-bandwidth** — vision heuristics + mock reasoning run on a laptop/edge box with zero network; EXIF-based capture, no cloud round-trip for the core demo.
4. **Human-in-the-loop** — nothing is logged until the officer confirms; the AI output is a *draft*.
5. **Multilingual** — narrative is generated in a canonical English form, rendered/translated at the client (IndicTrans/LLM translation at render time; the DB stores the canonical form so pattern matching stays honest).

## 🏗️ Architecture

```
Photo + EXIF (GPS, timestamp)
   │  POST /api/upload
   ▼
Vision layer (explainable, optional YOLO)
   ├─ Object detection   YOLOv8n (ultralytics, COCO) — mapped to crime-relevant
   │                      categories (knife→bladed weapon, vehicles, bags…)
   ├─ Blood-like stain   HSV red-hue heuristic + morphology + area/saturation basis
   ├─ OCR                EasyOCR/Tesseract (optional; skipped when absent)
   └─ Tamper check       JPEG re-encode ELA diff + EXIF sanity (never a verdict)
   ▼
Reasoning layer (LLM via API, or offline mock)
   Structured detections → strict-JSON prompt → narrative + anomaly flags + next steps
   ▼
Officer review (React UI)
   Confirm / edit / reject every detection, stain, OCR line, narrative, next step
   ▼
Case log (SQLite) — officer ID, GPS, timestamps, audit trail
   ▼
Pattern matching — category-overlap scoring vs past confirmed cases
```

## 📦 Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, uvicorn, OpenCV, Pillow, NumPy |
| Vision (opt-in) | YOLOv8n (`ultralytics`), EasyOCR / Tesseract |
| Reasoning | OpenAI-compatible LLM API, offline mock fallback |
| Frontend | React 18, TypeScript, Vite (mobile-first) |
| Storage | SQLite case log with audit trail |
| Deploy | Docker · Hugging Face Spaces · Streamlit Community Cloud |

## 📁 Repo layout

```
crime-scene-ai/
├── backend/               FastAPI service (Python 3.10+)
│   ├── app/main.py        API entrypoint, static image serving
│   ├── app/pipeline.py    orchestrates vision → reasoning
│   ├── app/vision/        detector.py (YOLO), stains.py, ocr.py, tamper.py
│   ├── app/reasoning.py   LLM call (OpenAI-compatible) + offline mock
│   ├── app/db.py          SQLite case log + pattern matching
│   └── data/              images + cases.db (gitignored)
├── frontend/              React + Vite + TS (mobile-first)
│   └── src/components/    UploadView, ImageAnnotator (canvas), ReportEditor, CaseList
├── scripts/               sample-image generator + end-to-end API test
├── gradio_app.py          Gradio port (Hugging Face Space)
└── streamlit_app.py       Streamlit port (Community Cloud, torch-free YOLO via ONNX)
```

## 🚀 Quick start

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # core (offline demo ready)
pip install -r requirements-ai.txt     # OPTIONAL: YOLO + OCR (needs internet once)
$env:OPENAI_API_KEY = "sk-..."        # OPTIONAL: real LLM reasoning
uvicorn app.main:app --port 8000
```

### Frontend (separate terminal)

```powershell
cd frontend
npm install
npm run dev                            # http://localhost:5173  (proxies /api → :8000)
```

### End-to-end test without any UI

```powershell
python scripts\make_sample_image.py    # synthetic "crime scene" photo (stain heuristic demo)
python scripts\demo_test.py            # upload → analysis → confirm → matches
python scripts\demo_test.py http://localhost:8000 scripts\sample_real_photo.jpg
                                       # same loop with a REAL photo → YOLO vehicle/person detection
```

### Single-file apps

```bash
python gradio_app.py      # http://localhost:7860
streamlit run streamlit_app.py
```

## 🔌 Enabling the real models (optional)

| Component | Install | Effect |
|---|---|---|
| YOLOv8n object detection | `pip install -r requirements-ai.txt` (first run downloads `yolov8n.pt` weights) | `source: "yolo"` detections for knife/car/bag/… |
| OCR | `pip install -r requirements-ai.txt` + Tesseract binary: `winget install UB-Mannheim.TesseractOCR` (override path via `$env:TESSERACT_CMD`) | `ocr` entries in analysis |
| LLM reasoning | `$env:OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`, `OPENAI_MODEL`) | `llm.source: "openai"` instead of `"mock"` |

Everything degrades gracefully: when a component is missing, the pipeline
continues and reports `processing_notes` explaining exactly what is missing
and why — no silent behavior.

## 📡 API surface

```
POST /api/upload                     multipart image (+ optional officer_id, lat, lng)
                                     → full analysis (detections, stains, ocr, tamper, llm draft)
POST /api/cases                      officer-confirmed report → case log entry
GET  /api/cases                      list (id, officer, timestamps, counts, llm source)
GET  /api/cases/{id}                 full detail incl. audit log + pattern matches
GET  /api/images/{image_id}          original uploaded image
GET  /api/health
```

## 🌐 Deploying

### Hugging Face Spaces (Docker Space, FastAPI + React on port 7860)

```powershell
git remote add space https://huggingface.co/spaces/<your-org>/<space-name>
git push space main        # HF builds the Dockerfile automatically
```

- Space card: `README.md.hf` · Build: `Dockerfile` (multi-stage: `npm ci && npm run build` → python:3.11-slim with tesseract-ocr)
- Optional LLM: add `OPENAI_API_KEY` as a Space secret (mock LLM used otherwise).

### Streamlit Community Cloud (free, no credit card)

1. Push the repo to GitHub (`.gitignore` already excludes `.venv`, `node_modules`, `backend/data`, `*.pt`).
2. On https://share.streamlit.io → **Create app** → connect the GitHub repo, entry file `streamlit_app.py`, root `requirements.txt`.

> Notes: the app sleeps after ~12h idle and wakes with one click. YOLO runs in
> the cloud too via the bundled `yolov8n.onnx` + onnxruntime (torch-free
> fallback). OCR (Tesseract) is unavailable in the cloud; the app reports it in
> `processing_notes` by design and still runs the stain heuristic + offline
> mock reasoning + full confirm/log/pattern-match flow.

## ⚠️ Honest limitations (demo-grade, not evidence-grade)

- Detections are **suggestions for triage**, never admissible evidence. The officer-confirmation step is the evidence boundary.
- ELA tamper flag is a heuristic (works on visibly re-saved/cropped JPEGs); it is always reported with `"inconclusive"` framing and a note.
- "Blood-like stain" heuristic detects red-dominant regions — it cannot distinguish blood from paint/rust/curry; the report says so.
- Pattern matching scores shared object categories — a triage aid, not case-linking evidence.

## 📄 License

[MIT](LICENSE)

Built for the **Smart India Hackathon (SIH)** demo track.
