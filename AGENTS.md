# AGENTS.md — Crime Scene AI

## Commands

- Backend server (dev): `cd backend; .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`
- Backend install: `cd backend; .\.venv\Scripts\python.exe -m pip install -r requirements.txt`
  (optional AI upgrades: `-r requirements-ai.txt` — YOLO + Tesseract)
- Frontend dev: `cd frontend; npm.cmd run dev` (use `npm.cmd`, npm.ps1 is blocked by execution policy)
- Frontend build/typecheck: `npm.cmd run build` (runs `tsc -b` + `vite build`)
- E2E test (backend must be running): `python scripts\make_sample_image.py; python scripts\demo_test.py`
- Run sample image regeneration: `python scripts\make_sample_image.py`

## Conventions

- Backend: FastAPI + plain dicts for vision/reasoning output (no ORM). Python 3.10.
- Vision modules in `backend/app/vision/` each return `(data, notes)` where notes
  explain degradations — never fail silently.
- LLM reasoning (`backend/app/reasoning.py`) returns fixed JSON schema keys:
  narrative, narrative_hi, anomaly_flags, next_steps. Falls back to offline mock.
- All Python run via the venv interpreter, never the global one.
