"""Streamlit Community Cloud front-end for Crime Scene AI.

Human-in-the-loop demo: upload one or more photos -> explainable triage draft
(objects, stain candidates, AI observation & suggestions, offline-safe) ->
click-to-drop numbered evidence markers -> officer writes/confirms the
narrative -> confirmed report logged to the case file with audit trail and
pattern matches -> JSON/PDF report export.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

import streamlit.elements.image as _st_img_elements
if not hasattr(_st_img_elements, "UseColumnWith"):
    _st_img_elements.UseColumnWith = bool
from st_canvas import streamlit_image_coordinates

from app import config, db, reasoning
from app.pipeline import run_pipeline

db.init_db()

st.set_page_config(page_title="Crime Scene AI — Evidence Capture Assistant", layout="wide")

_DEFAULTS = {
    "analyses": {},
    "images": {},
    "merged": None,
    "narrative": "",
    "last_case": None,
    "markers": [],
    "marker_seq": 0,
    "measurements": [],
    "measure_seq": 0,
    "photo_view": None,
    "_canvas_cache": {},
}
CANVAS_W = 1100
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid="stApp"] {
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 520px at 12% -12%, rgba(99, 102, 241, 0.16), transparent 60%),
    radial-gradient(900px 480px at 108% 4%, rgba(34, 211, 238, 0.10), transparent 55%),
    linear-gradient(180deg, #0b1020 0%, #070b16 100%);
}

[data-testid="stHeader"] { background: rgba(7, 11, 22, 0.55); backdrop-filter: blur(8px); }

[data-testid="stMainBlockContainer"], [data-testid="stBlockContainer"], .block-container {
  padding-top: 2.2rem; max-width: 1240px;
}

.hero { margin-bottom: 0.2rem; }
.hero .badge {
  display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
  color: #a5b4fc; border: 1px solid rgba(129, 140, 248, 0.45); border-radius: 999px;
  padding: 0.28rem 0.9rem; background: rgba(99, 102, 241, 0.12);
}
.hero h1 {
  margin: 0.7rem 0 0.35rem; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em;
  background: linear-gradient(92deg, #a5b4fc 0%, #67e8f9 60%, #6ee7b7 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p.sub { color: #94a3b8; font-size: 0.98rem; line-height: 1.6; max-width: 760px; margin: 0; }
.hero p.sub b { color: #e2e8f0; font-weight: 600; }
.hero hr.divider {
  margin: 1.3rem 0 0; border: 0; height: 1px;
  background: linear-gradient(90deg, rgba(129,140,248,.55), rgba(103,232,249,.35), transparent);
}

[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
  letter-spacing: -0.01em;
}
[data-testid="stSubheader"], [data-testid="stHeading"] h3 {
  font-size: 0.9rem !important; font-weight: 700 !important; letter-spacing: 0.1em;
  text-transform: uppercase; color: #93c5fd !important;
}

[data-testid="stMetric"] {
  background: linear-gradient(165deg, rgba(148,163,184,0.10), rgba(148,163,184,0.02));
  border: 1px solid rgba(148,163,184,0.22); border-radius: 14px;
  padding: 0.7rem 1rem 0.85rem;
}
[data-testid="stMetric"] label {
  color: #94a3b8; font-size: 0.76rem; letter-spacing: 0.05em; text-transform: uppercase;
}
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; }

[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(148, 163, 184, 0.045);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 16px;
  padding: 1rem 1.1rem !important;
}

[data-testid="stBaseButton-secondary"], .stButton > button, .stDownloadButton button {
  border-radius: 10px; font-weight: 600; padding: 0.42rem 1.05rem; min-height: 2.4rem;
  border: 1px solid rgba(148, 163, 184, 0.28); background: rgba(148, 163, 184, 0.08);
  color: #e2e8f0; transition: all 0.15s ease;
}
[data-testid="stBaseButton-secondary"]:hover, .stButton > button:hover {
  border-color: rgba(129, 140, 248, 0.7); background: rgba(129, 140, 248, 0.14); color: #fff;
}
[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {
  background: linear-gradient(92deg, #6366f1 0%, #06b6d4 100%); border: none; color: #fff;
  box-shadow: 0 10px 26px -10px rgba(99, 102, 241, 0.55);
}
[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover {
  background: linear-gradient(92deg, #545ef0 0%, #0891b2 100%);
  box-shadow: 0 14px 30px -10px rgba(99, 102, 241, 0.7);
}

[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stBaseInput-stTextInput"] input, [data-testid="stBaseInput-stNumberInput"] input {
  background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 10px; color: #e2e8f0; caret-color: #818cf8;
}
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: #818cf8; box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.16);
}
[data-testid="stTextInput"] label, [data-testid="stNumberInput"] label,
[data-testid="stTextArea"] label, [data-testid="stFileUploader"] label {
  color: #a9b7d0 !important; font-weight: 500;
}

[data-testid="stFileUploaderDropzone"] {
  background: rgba(15, 23, 42, 0.55);
  border: 1.5px dashed rgba(148, 163, 184, 0.35);
  border-radius: 14px; transition: border-color 0.15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(129, 140, 248, 0.8); }
[data-testid="stFileUploaderDropzone"] button {
  background: rgba(129, 140, 248, 0.16); border-radius: 8px;
}

[data-testid="stRadio"] > div { flex-wrap: wrap; gap: 0.45rem; }
[data-testid="stRadio"] label {
  background: rgba(148, 163, 184, 0.07); border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 999px; padding: 0.28rem 0.95rem; transition: all 0.15s ease; color: #cbd5e1;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(92deg, #6366f1, #06b6d4); border-color: transparent; color: #fff;
}

[data-testid="stDataFrame"] {
  border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; overflow: hidden;
}

[data-testid="stCaption"], .stCaption { color: #7c8db0; }

[data-testid="stExpander"] {
  border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 12px;
  background: rgba(148, 163, 184, 0.04);
}
[data-testid="stInfo"] {
  background: rgba(59, 130, 246, 0.10); border: 1px solid rgba(96, 165, 250, 0.25);
  border-radius: 12px; color: #bfdbfe;
}

.foot {
  margin: 2.6rem 0 1rem; padding-top: 1.1rem;
  border-top: 1px solid rgba(148, 163, 184, 0.15);
  color: #5b6b8c; font-size: 0.8rem; text-align: center; letter-spacing: 0.04em;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.25); border-radius: 8px; }

/* ---------- animated background ---------- */
[data-testid="stAppViewContainer"] { isolation: isolate; }
.bg-fx { position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; }
.bg-fx .orb { position: absolute; border-radius: 50%; filter: blur(80px); will-change: transform, opacity; }
.bg-fx .orb-a {
  width: 48vw; height: 48vw; left: -14vw; top: -16vw; opacity: 0.35;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.55), rgba(99, 102, 241, 0) 70%);
  animation: driftA 26s ease-in-out infinite alternate;
}
.bg-fx .orb-b {
  width: 40vw; height: 40vw; right: -12vw; top: 6vh; opacity: 0.30;
  background: radial-gradient(circle, rgba(34, 211, 238, 0.5), rgba(34, 211, 238, 0) 70%);
  animation: driftB 32s ease-in-out infinite alternate;
}
.bg-fx .orb-c {
  width: 46vw; height: 46vw; left: 22vw; bottom: -18vw; opacity: 0.28;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.45), rgba(168, 85, 247, 0) 70%);
  animation: driftC 38s ease-in-out infinite alternate;
}
.bg-fx .orb-d {
  width: 30vw; height: 30vw; right: 12vw; bottom: 4vh; opacity: 0.26;
  background: radial-gradient(circle, rgba(244, 63, 94, 0.4), rgba(244, 63, 94, 0) 70%);
  animation: driftD 28s ease-in-out infinite alternate;
}
.bg-fx .orb-e {
  width: 70vw; height: 70vw; left: 15vw; top: -30vw; opacity: 0.14;
  background: conic-gradient(from 0deg, transparent, rgba(129, 140, 248, 0.5), transparent 32%, rgba(103, 232, 249, 0.4) 55%, transparent);
  animation: spinSlow 70s linear infinite;
}
.bg-fx .grid {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(148, 163, 184, 0.07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, 0.07) 1px, transparent 1px);
  background-size: 52px 52px;
  -webkit-mask-image: radial-gradient(ellipse 90% 70% at 50% 0%, black 25%, transparent 78%);
  mask-image: radial-gradient(ellipse 90% 70% at 50% 0%, black 25%, transparent 78%);
  animation: gridScroll 75s linear infinite;
}
.bg-fx .slide { position: absolute; inset: 0; opacity: 0; animation: slideShow 45s ease-in-out infinite; }
.bg-fx .slide.s1 {
  background:
    radial-gradient(52% 40% at 18% 22%, rgba(129, 140, 248, 0.26), transparent 62%),
    radial-gradient(40% 34% at 86% 68%, rgba(34, 211, 238, 0.18), transparent 60%);
  animation-delay: 0s;
}
.bg-fx .slide.s2 {
  background:
    radial-gradient(50% 42% at 82% 18%, rgba(168, 85, 247, 0.24), transparent 62%),
    radial-gradient(44% 38% at 12% 74%, rgba(244, 63, 94, 0.15), transparent 60%);
  animation-delay: 15s;
}
.bg-fx .slide.s3 {
  background:
    radial-gradient(54% 44% at 50% 30%, rgba(34, 211, 238, 0.22), transparent 64%),
    radial-gradient(36% 30% at 8% 60%, rgba(99, 102, 241, 0.22), transparent 60%);
  animation-delay: 30s;
}

@keyframes driftA {
  from { transform: translate3d(0, 0, 0) scale(1); opacity: 0.25; }
  to   { transform: translate3d(9vw, 7vh, 0) scale(1.18); opacity: 0.5; }
}
@keyframes driftB {
  from { transform: translate3d(0, 0, 0) scale(1.1); opacity: 0.2; }
  to   { transform: translate3d(-8vw, 10vh, 0) scale(0.95); opacity: 0.45; }
}
@keyframes driftC {
  from { transform: translate3d(0, 0, 0) scale(1); opacity: 0.18; }
  to   { transform: translate3d(-10vw, -8vh, 0) scale(1.22); opacity: 0.42; }
}
@keyframes driftD {
  from { transform: translate3d(0, 0, 0) scale(1.05); opacity: 0.16; }
  to   { transform: translate3d(6vw, -9vh, 0) scale(0.9); opacity: 0.38; }
}
@keyframes spinSlow { to { transform: rotate(360deg); } }
@keyframes gridScroll { to { background-position: 52px 52px; } }
@keyframes slideShow {
  0% { opacity: 0; }
  6% { opacity: 1; }
  30% { opacity: 1; }
  36% { opacity: 0; }
  100% { opacity: 0; }
}

/* ---------- motion: title, cards, buttons ---------- */
.hero h1 {
  background-size: 220% auto;
  animation: textFlow 9s ease-in-out infinite;
}
@keyframes textFlow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.hero, [data-testid="stMetric"] { animation: riseIn 0.5s cubic-bezier(0.2, 0.7, 0.3, 1) both; }

[data-testid="stVerticalBlockBorderWrapper"] {
  animation: riseIn 0.5s cubic-bezier(0.2, 0.7, 0.3, 1) both;
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-3px);
  border-color: rgba(129, 140, 248, 0.35);
  box-shadow: 0 18px 40px -18px rgba(99, 102, 241, 0.35);
}
@keyframes riseIn {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stBaseButton-secondary"], .stButton > button, .stDownloadButton button {
  position: relative; overflow: hidden;
}
[data-testid="stBaseButton-secondary"]::after, .stButton > button::after, .stDownloadButton button::after {
  content: ""; position: absolute; top: 0; left: -130%; height: 100%; width: 55%;
  background: linear-gradient(100deg, transparent, rgba(255, 255, 255, 0.28), transparent);
  transform: skewX(-18deg); animation: shimmer 3.4s ease-in-out infinite;
}
@keyframes shimmer {
  0% { left: -130%; }
  55% { left: 130%; }
  100% { left: 130%; }
}

@keyframes csSlideBlur {
  0%   { filter: blur(0px); }
  35%  { filter: blur(7px); }
  75%  { filter: blur(3px); }
  100% { filter: blur(0px); }
}

html, body { scroll-behavior: smooth; }

@media (prefers-reduced-motion: reduce) {
  .bg-fx .orb, .bg-fx .grid, .bg-fx .slide, .hero h1,
  [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"],
  [data-testid="stBaseButton-secondary"]::after, .stButton button::after, .stDownloadButton button::after,
  div[data-testid="stColumn"]:has(div[data-testid="stFileUploader"]) {
    animation: none !important;
  }
  [data-testid="stVerticalBlockBorderWrapper"] { transform: none !important; }
}

@media (max-width: 720px) {
  [data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) {
    flex-direction: column;
  }
  [data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) > [data-testid="stColumn"] {
    width: 100% !important;
    max-width: 100% !important;
    flex-basis: auto !important;
  }
  [data-testid="stColumn"]:has(div[data-testid="stFileUploader"]) {
    transform: none !important;
    animation: none !important;
  }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
  }
  [data-testid="stHorizontalBlock"]:has(input[placeholder^="Note for marker"]) {
    flex-direction: column;
  }
  .hero h1 { font-size: 1.7rem !important; }
  .hero .badge { font-size: 0.62rem !important; }
}
</style>
"""

_HERO = """
<div class="hero">
  <span class="badge">EVIDENCE CAPTURE ASSISTANT</span>
  <h1>Crime Scene AI</h1>
  <p class="sub">Upload crime-scene photos — detections, stain candidates and the narrative are a
  <b>suggestion for triage</b>; nothing is logged until an officer confirms it (evidence boundary).</p>
  <hr class="divider">
</div>
"""

st.markdown(_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="bg-fx">'
    '<div class="orb orb-a"></div><div class="orb orb-b"></div><div class="orb orb-c"></div>'
    '<div class="orb orb-d"></div><div class="orb orb-e"></div>'
    '<div class="grid"></div>'
    '<div class="slide s1"></div><div class="slide s2"></div><div class="slide s3"></div>'
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(_HERO, unsafe_allow_html=True)


def _fmt_basis(basis):
    if not basis:
        return ""
    if isinstance(basis, str):
        return basis
    return "; ".join(str(b) for b in basis)


def _pct(v):
    if isinstance(v, (int, float)):
        return f"{v * 100:.0f}%"
    return v


def _df(rows, headers):
    if not rows:
        return pd.DataFrame(columns=headers)
    data = []
    for r in rows:
        row = {}
        for h in headers:
            v = r.get(h, "")
            row[h] = ", ".join(v) if isinstance(v, list) else v
            if h == "confidence":
                row[h] = _pct(row[h])
        data.append(row)
    return pd.DataFrame(data)


def _dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=10):
    for xs in range(x1, x2, dash * 2):
        cv2.line(img, (xs, y1), (min(xs + dash, x2), y1), color, thickness)
        cv2.line(img, (xs, y2), (min(xs + dash, x2), y2), color, thickness)
    for ys in range(y1, y2, dash * 2):
        cv2.line(img, (x1, ys), (x1, min(ys + dash, y2)), color, thickness)
        cv2.line(img, (x2, ys), (x2, min(ys + dash, y2)), color, thickness)


def _label_chip(img, x1, y1, x2, y2, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), color, -1)
    cv2.putText(img, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def _pin_chip(img, x, y, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tx = min(max(x - tw // 2, 0), img.shape[1] - tw - 5)
    ty = max(y - 22, 0)
    cv2.rectangle(img, (tx, ty), (tx + tw + 4, ty + th + 4), color, -1)
    cv2.putText(img, text, (tx + 2, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def annotate(img_bgr, analysis):
    """Draw detection boxes (solid) + stain candidates (dashed) on one image."""
    out = img_bgr.copy()
    colors = {
        "person": (0, 200, 0),
        "weapon": (0, 0, 255),
        "vehicle": (255, 128, 0),
        "container": (0, 200, 255),
        "personal item": (255, 0, 255),
        "electronic device": (200, 0, 200),
        "discarded item": (180, 180, 0),
    }
    for d in analysis.get("objects", []):
        b = d.get("bbox")
        if not b:
            continue
        x1, y1, x2, y2 = (int(round(b[k])) for k in ("x1", "y1", "x2", "y2"))
        color = colors.get(d.get("category"), (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _label_chip(out, x1, y1, x2, y2, f"{d.get('class', '?')} {_pct(d.get('confidence', 0))}", color)
    for s in analysis.get("stains", []):
        b = s.get("bbox")
        if not b:
            continue
        x1, y1, x2, y2 = (int(round(b[k])) for k in ("x1", "y1", "x2", "y2"))
        _dashed_rect(out, x1, y1, x2, y2, (0, 0, 220), thickness=2)
        _label_chip(out, x1, y1, x2, y2, f"blood stain {_pct(s.get('confidence', 0))}", (0, 0, 220))
    return out


def draw_pins(img, markers, photo):
    for m in markers:
        if m.get("photo") != photo:
            continue
        x, y = m["x"], m["y"]
        cv2.circle(img, (x, y), 10, (255, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
        _pin_chip(img, x, y, str(m["id"]), (255, 255, 0))


def canvas_bgr(photo):
    img = st.session_state.images[photo]["bgr"].copy()
    img = annotate(img, st.session_state.analyses[photo])
    draw_pins(img, st.session_state.markers, photo)
    return img


def merge_analyses(analyses):
    objs, stains, notes = [], [], []
    first = next(iter(analyses.values()))
    for name, a in analyses.items():
        for o in a.get("objects", []):
            o.setdefault("photo", name)
            objs.append(o)
        for s in a.get("stains", []):
            s.setdefault("photo", name)
            stains.append(s)
        notes.extend(a.get("processing_notes", []))
    merged = {
        "width": first.get("width"),
        "height": first.get("height"),
        "objects": objs,
        "stains": stains,
        "tamper": first.get("tamper", {}),
        "metadata": first.get("metadata", {}),
        "gps": first.get("gps"),
        "captured_at": first.get("captured_at"),
        "processing_notes": notes,
    }
    if len(analyses) > 1:
        notes.append(f"scene compiled from {len(analyses)} photos — candidates merged for triage")
    merged["processing_notes"] = notes
    return merged


def analyze_all(files, officer_id, lat, lng):
    analyses, images = {}, {}
    for name, raw in files:
        try:
            a = run_pipeline(raw)
        except ValueError as exc:
            st.error(f"{name}: {exc}")
            return False
        image_id = uuid.uuid4().hex[:12]
        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error(f"{name}: could not decode image")
            return False
        cv2.imwrite(str(config.IMAGES_DIR / f"{image_id}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        gps = a["metadata"].get("gps")
        if gps is None and lat and lng:
            gps = {"lat": float(lat), "lng": float(lng)}
        a["gps"] = gps
        a["captured_at"] = a["metadata"].get("captured_at") or datetime.now(timezone.utc).isoformat()
        a["image_id"] = image_id
        analyses[name] = a
        images[name] = {"image_id": image_id, "bgr": img_bgr}
    st.session_state.analyses = analyses
    st.session_state.images = images
    st.session_state.photo_view = next(iter(images))
    st.session_state.markers = []
    st.session_state.marker_seq = 0
    st.session_state.measurements = []
    st.session_state.measure_seq = 0
    st.session_state.merged = merge_analyses(analyses)
    st.session_state.merged["suggestions"] = reasoning.reason(st.session_state.merged)
    st.session_state.narrative = ""
    return True


def confirm(officer_id, narrative, lat, lng):
    merged = st.session_state.merged
    if not merged:
        st.error("Analyze photos first.")
        return
    images = st.session_state.images
    gps = merged.get("gps")
    if gps is None and lat and lng:
        gps = {"lat": float(lat), "lng": float(lng)}
    payload = {
        "officer_id": officer_id or "unknown",
        "image_id": st.session_state.photo_view or next(iter(images), ""),
        "gps": gps,
        "captured_at": merged.get("captured_at"),
        "narrative": narrative,
        "original_narrative": "",
        "next_steps": merged.get("suggestions", {}).get("next_steps", []),
        "anomaly_flags": merged.get("suggestions", {}).get("anomaly_flags", []),
        "objects": merged.get("objects", []),
        "stains": merged.get("stains", []),
        "ocr": [],
        "tamper": merged.get("tamper", {}),
        "metadata": merged.get("metadata", {}),
        "llm_source": merged.get("suggestions", {}).get("source", "manual"),
        "ai_report": merged.get("suggestions", {}),
        "processing_notes": merged.get("processing_notes", []),
        "evidence_markers": st.session_state.markers,
        "measurements": st.session_state.measurements,
        "photos": [{"name": n, "image_id": im["image_id"]} for n, im in images.items()],
    }
    log_entries = [
        {"actor": "system", "action": "case-opened",
         "detail": {"photos": len(payload["photos"]), "officer_id": officer_id}},
        {"actor": "ai", "action": "analysis-draft-generated",
         "detail": {"objects": len(payload["objects"]), "stains": len(payload["stains"])}},
        {"actor": "officer", "action": "confirmed-edited",
         "detail": {"narrative_written": bool(payload["narrative"]),
                    "objects_confirmed": len(payload["objects"]),
                    "evidence_markers": len(payload["evidence_markers"])}},
    ]
    case_id = db.insert_case(payload, log_entries)
    detail = db.get_case(case_id)
    detail["matches"] = db.find_matches(case_id)
    st.session_state.last_case = detail


def case_card(c):
    md = f"## Case `{c['id']}` — confirmed\n"
    md += f"- Officer: `{c['officer_id']}` · logged {c['created_at']}\n"
    photos = c.get("photos") or [{"name": "(archive)", "image_id": c.get("image_id")}]
    md += "- Photos: " + ", ".join(f"`{p['name']}` (`{p['image_id']}`)" for p in photos) + "\n"
    markers = c.get("evidence_markers", [])
    md += f"- Evidence markers: {len(markers)}\n"
    measurements = c.get("measurements", [])
    if measurements:
        md += "- Measured distances: " + ", ".join(
            f"{m['cm']} cm" if m.get("cm") else f"{m['px_len']} px" for m in measurements) + "\n"
    md += f"- Narrative: {c['narrative']}\n"
    ai = c.get("ai_report") or {}
    ai_flags = ai.get("anomaly_flags") or []
    if ai_flags:
        md += "\n**AI anomaly flags (triage draft):**\n" + "\n".join(f"- {f}" for f in ai_flags) + "\n"
    matches = c.get("matches", [])
    if matches:
        md += "\n**Pattern matches (triage aid):**\n"
        for m in matches:
            md += f"- case `{m['case_id']}` score {m['score']} · shared {', '.join(m['shared_categories'])}\n"
    else:
        md += "\nNo past cases share detected categories.\n"
    md += "\n**Audit trail:**\n"
    for e in c.get("log", []):
        md += f"- {e['ts']} · {e['actor']} · {e['action']}\n"
    return md


def _report_dict(detail):
    def clean(items):
        return [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in it.items()}
            for it in items
        ]

    return {
        "case_id": detail["id"],
        "officer_id": detail["officer_id"],
        "status": detail["status"],
        "created_at": detail["created_at"],
        "captured_at": detail["captured_at"],
        "gps": detail.get("gps"),
        "photos": detail.get("photos", []),
        "narrative": detail["narrative"],
        "ai_report": detail.get("ai_report", {}),
        "objects": clean(detail.get("objects", [])),
        "stains": clean(detail.get("stains", [])),
        "evidence_markers": clean(detail.get("evidence_markers", [])),
        "measurements": clean(detail.get("measurements", [])),
        "tamper": detail.get("tamper", {}),
        "processing_notes": detail.get("processing_notes", []),
        "pattern_matches": detail.get("matches", []),
        "audit_trail": detail.get("log", []),
    }


def _pdf_text(s):
    return str(s).encode("latin-1", "replace").decode("latin-1")


def _pdf_report(detail):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def sec(title):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, _pdf_text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)

    def line(text):
        pdf.multi_cell(0, 5, _pdf_text(text), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _pdf_text(f"Crime Scene AI — Case {detail['id']} (confirmed)"),
             new_x="LMARGIN", new_y="NEXT")

    sec("Case")
    line(f"Officer: {detail['officer_id']}  |  Logged: {detail['created_at']}  |  "
         f"Captured: {detail.get('captured_at')}")
    photos = detail.get("photos") or []
    if photos:
        line("Photos: " + ", ".join(p["name"] for p in photos))
    gps = detail.get("gps")
    line(f"GPS: {gps if gps else 'not recorded'}")

    sec("Narrative (confirmed)")
    line(detail["narrative"])

    ai = detail.get("ai_report") or {}
    if ai:
        sec("AI observation report (triage draft - not evidence)")
        line(f"Source: {ai.get('source', '?')} / {ai.get('model', '?')}")
        if ai.get("narrative"):
            line("Draft narrative: " + ai["narrative"])
        for f in ai.get("anomaly_flags", []):
            line("- FLAG: " + f)
        for s in ai.get("next_steps", []):
            line("- NEXT: " + s)

    matches = detail.get("matches", [])
    if matches:
        sec("Pattern matches (triage aid)")
        for m in matches:
            line(f"- case {m['case_id']} score {m['score']} — shared: {', '.join(m['shared_categories'])}")

    sec("Objects detected")
    pdf.cell(22, 5, "Photo", border=1)
    pdf.cell(36, 5, "Class", border=1)
    pdf.cell(30, 5, "Category", border=1)
    pdf.cell(16, 5, "Conf", border=1)
    pdf.cell(22, 5, "Source", border=1, new_x="LMARGIN", new_y="NEXT")
    for o in detail.get("objects", []):
        pdf.cell(22, 5, _pdf_text(o.get("photo", "")), border=1)
        pdf.cell(36, 5, _pdf_text(o.get("class", "?")), border=1)
        pdf.cell(30, 5, _pdf_text(o.get("category", "")), border=1)
        pdf.cell(16, 5, _pdf_text(_pct(o.get("confidence", ""))), border=1)
        pdf.cell(22, 5, _pdf_text(o.get("source", "")), border=1, new_x="LMARGIN", new_y="NEXT")

    sec("Blood-like stain candidates")
    pdf.cell(22, 5, "Photo", border=1)
    pdf.cell(60, 5, "Class", border=1)
    pdf.cell(16, 5, "Conf", border=1)
    pdf.cell(18, 5, "Area%", border=1, new_x="LMARGIN", new_y="NEXT")
    for s in detail.get("stains", []):
        pdf.cell(22, 5, _pdf_text(s.get("photo", "")), border=1)
        pdf.cell(60, 5, _pdf_text(s.get("class", "?")), border=1)
        pdf.cell(16, 5, _pdf_text(_pct(s.get("confidence", ""))), border=1)
        pdf.cell(18, 5, _pdf_text(s.get("area_pct", "")), border=1, new_x="LMARGIN", new_y="NEXT")

    if detail.get("evidence_markers"):
        sec("Evidence markers")
        for m in detail["evidence_markers"]:
            line(f"- #{m['id']} · {m.get('photo', '')} · px({m['x']}, {m['y']}) · {m.get('note', '')}")

    if detail.get("measurements"):
        sec("Measured distances")
        for m in detail["measurements"]:
            line(f"- #{m['id']} · {m.get('photo', '')} · "
                 f"{m.get('cm', '?')} cm ({m.get('px_len', '?')} px)" if m.get("cm")
                 else f"- #{m['id']} · {m.get('photo', '')} · {m.get('px_len', '?')} px")

    sec("Audit trail")
    for e in detail.get("log", []):
        line(f"- {e['ts']} · {e['actor']} · {e['action']}")

    return bytes(pdf.output())


def _reset_derived():
    for k in ("analyses", "images", "merged", "markers", "last_case", "photo_view"):
        st.session_state[k] = _DEFAULTS[k]
    st.session_state.marker_seq = 0
    st.session_state.measurements = []
    st.session_state.measure_seq = 0
    st.session_state.narrative = ""


def _on_canvas_click():
    """Fires once per canvas click (component's on_click) — no remount, no blink.

    The component reports x/y/width/height in *displayed* pixels (offsetX + CSS
    rendered img size), so coordinates are mapped to original-image pixels with
    the returned width/height as the scale.
    """
    photo = st.session_state.get("photo_view")
    if not photo:
        return
    click = st.session_state.get(f"cv_{photo}")
    if not isinstance(click, dict):
        return
    img = st.session_state.images[photo]["bgr"]
    h, w = img.shape[:2]
    cw, ch = click.get("width"), click.get("height")
    if cw and ch and cw > 0 and ch > 0:
        x = int(round(click.get("x", 0) * w / cw))
        y = int(round(click.get("y", 0) * h / ch))
    else:
        x, y = int(round(click.get("x", 0))), int(round(click.get("y", 0)))
    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)
    mode = st.session_state.get("canvas_mode", "View overlay")
    if mode == "Add evidence markers":
        st.session_state.marker_seq += 1
        st.session_state.markers.append({
            "id": st.session_state.marker_seq, "photo": photo,
            "x": x, "y": y, "note": "",
            "ts": datetime.now(timezone.utc).isoformat(),
        })


col1, col2 = st.columns([1, 2])

merged = st.session_state.merged
images = st.session_state.images
if merged:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Photos", len(images))
    m2.metric("Objects", len(merged.get("objects", [])))
    m3.metric("Stain candidates", len(merged.get("stains", [])))
    m4.metric("Evidence markers", len(st.session_state.markers))

with col1:
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Scene photo(s)", type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True, on_change=_reset_derived,
        )
        st.caption("JPG / PNG / WebP / BMP — analysis runs offline-safe.")
        files = [(f.name, f.getvalue()) for f in uploaded] if uploaded else []
        officer_id = st.text_input("Officer ID", placeholder="e.g. SUB-INSP-07")
        lat_in = st.number_input("Latitude (opt.)", value=None, format="%.6f")
        lng_in = st.number_input("Longitude (opt.)", value=None, format="%.6f")
        analyze_btn = st.button("Analyze all photos", type="primary")

    placed = "calc(100% + 0.5rem)" if not bool(uploaded) else "0px"
    slide = "0.55s ease-in-out" if bool(uploaded) else "none"
    blur = "csSlideBlur 0.55s ease-in-out" if bool(uploaded) else "none"
    st.markdown(f"""
    <style>
    div[data-testid="stColumn"]:has(div[data-testid="stFileUploader"]) {{
      transform: translateX({placed});
      transition: transform {slide};
      animation: {blur};
      will-change: transform, filter;
    }}
    </style>""", unsafe_allow_html=True)

if analyze_btn:
    if not files:
        st.error("Upload at least one photo first.")
    elif analyze_all(files, officer_id, lat_in, lng_in):
        st.success(f"Analyzed {len(files)} photo(s) — review the overlay, add markers, then confirm the case.")

with col2:
    merged = st.session_state.merged
    images = st.session_state.images
    if merged:
        with st.container(border=True):
            names = list(images)
            photo = st.session_state.photo_view
            if photo not in names:
                photo = names[0]
            if len(names) > 1:
                photo = st.radio("View photo", names, index=names.index(photo), horizontal=True, key="photo_sel")
            st.session_state.photo_view = photo
            mode = st.radio(
                "Canvas mode", ["View overlay", "Add evidence markers"],
                horizontal=True, key="canvas_mode",
            )
            canvas = canvas_bgr(photo)
            oh, ow = canvas.shape[:2]
            tw = min(CANVAS_W, ow)
            th = max(1, int(round(oh * tw / ow)))
            finger = "|".join((
                photo, mode,
                ";".join(f"{m['id']}:{m.get('photo', '')}:{m['x']}:{m['y']}" for m in st.session_state.markers),
                ";".join(f"{m['id']}:{m.get('photo', '')}:{m['px_len']}" for m in st.session_state.measurements),
            ))
            cache = st.session_state["_canvas_cache"]
            if cache.get("finger") != finger:
                thumb = canvas if tw == ow else cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                cache = {"finger": finger, "bytes": buf.tobytes() if ok else b"",
                         "rgb": cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)}
                st.session_state["_canvas_cache"] = cache
            if mode == "View overlay":
                st.caption("AI triage overlay — boxes are suggestions, not evidence")
                st.image(cache["bytes"], width="stretch")
            else:
                st.caption("Click anywhere on the photo to drop a numbered evidence marker.")
                streamlit_image_coordinates(
                    cache["rgb"], use_column_width="always", key=f"cv_{photo}",
                    on_click=_on_canvas_click, image_format="JPEG", jpeg_quality=85,
                )
            markers = st.session_state.markers
            if markers:
                st.markdown("### Evidence markers")
                for m in list(markers):
                    r1, r2, r3 = st.columns([4, 3, 1])
                    note = r1.text_input(
                        "note", value=m.get("note", ""), key=f"mk_note_{m['id']}",
                        label_visibility="collapsed", placeholder=f"Note for marker #{m['id']}...",
                    )
                    r2.caption(f"#{m['id']} · {m['photo']} · px({m['x']}, {m['y']})")
                    if r3.button(f"Remove #{m['id']}", key=f"mk_del_{m['id']}"):
                        st.session_state.markers = [x for x in st.session_state.markers if x["id"] != m["id"]]
                        st.rerun()
                    if note != m.get("note"):
                        m["note"] = note
    elif files:
        st.image(files[0][1], width="stretch")
        st.caption("Preview — click 'Analyze all photos' to run the triage pipeline.")

if merged:
    with st.container(border=True):
        st.caption(f"photos: {len(st.session_state.images)} · "
                   f"evidence markers: {len(st.session_state.markers)}")

        st.subheader("Objects detected")
        st.dataframe(_df(merged.get("objects", []),
                         ["class", "category", "confidence", "source", "photo"]),
                     width="stretch", hide_index=True)
        st.subheader("Blood-like stain candidates")
        st.dataframe(_df(merged.get("stains", []),
                         ["class", "confidence", "area_pct", "photo"]),
                     width="stretch", hide_index=True)

        tamper = merged.get("tamper", {})
        st.markdown(f"**Tamper check:** `{tamper.get('flag', 'inconclusive')}` "
                    f"(ELA score {tamper.get('ela_score', 'n/a')}) — heuristic, never a verdict")
        notes = merged.get("processing_notes", [])
        if notes:
            st.info("\n".join(f"- {n}" for n in dict.fromkeys(notes)))

        sug = merged.get("suggestions") or {}
        if sug:
            source, model = sug.get("source"), sug.get("model")
            if source == "mock":
                mode_txt = "offline rule-based draft (no API key configured)"
            else:
                mode_txt = f"LLM draft ({source} / {model})"
            st.markdown("## AI observations and suggestions (triage draft)")
            st.caption(f"Draft by: {mode_txt} — suggestions only, never evidence; an officer signs the final case.")
            flags = sug.get("anomaly_flags") or []
            steps = sug.get("next_steps") or []
            if flags:
                st.markdown("**Anomaly flags:**\n" + "\n".join(f"- {f}" for f in flags))
            if steps:
                st.markdown("**Suggested next steps:**\n" + "\n".join(f"- {s}" for s in steps))
            if sug.get("narrative"):
                with st.expander("AI-drafted observation report"):
                    st.write(sug["narrative"])
                    if st.button("Use as narrative draft", key="use_ai_narrative"):
                        st.session_state.narrative = sug["narrative"]
                        st.rerun()

        st.markdown("## Officer confirmation (human-in-the-loop)")
        st.caption(f"Marks {len(st.session_state.markers)} evidence markers")
        narrative = st.text_area(
            "Officer narrative (write your own account; nothing is logged until you confirm)",
            value=st.session_state.narrative, height=140,
            placeholder="Describe the scene, the markers and your triage decisions...",
        )
        st.session_state.narrative = narrative
        if st.button("Confirm & log case", type="primary"):
            confirm(officer_id, narrative, lat_in, lng_in)

if st.session_state.last_case:
    detail = st.session_state.last_case
    st.markdown(case_card(detail))
    report_json = json.dumps(_report_dict(detail), ensure_ascii=False, indent=2, default=str)
    st.download_button("Download JSON report", report_json,
                       file_name=f"case-{detail['id']}.json", mime="application/json",
                       key="dl_json_report")
    st.download_button("Download PDF report", _pdf_report(detail),
                       file_name=f"case-{detail['id']}.pdf", mime="application/pdf",
                       key="dl_pdf_report")

with st.container(border=True):
    st.markdown("## Past cases")
    rows = []
    for c in db.list_cases():
        rows.append({"id": c["id"], "officer": c["officer_id"], "created_at": c["created_at"],
                     "objects": c["object_count"], "stains": c["stain_count"],
                     "markers": len(c.get("evidence_markers", []))})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("No cases logged yet.")

st.markdown(
    '<div class="foot">Crime Scene AI · offline-safe triage draft · '
    "nothing is logged until an officer confirms · v0.1</div>",
    unsafe_allow_html=True,
)