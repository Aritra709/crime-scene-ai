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
from app.vision import detector as vision_detector
from app.vision import video as video_mod

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
    "video_analysis": None,
    "_canvas_cache": {},
}
CANVAS_W = 1100
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* Root variables for theme */
:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --primary-dark: #4f46e5;
    --secondary: #06b6d4;
    --secondary-light: #67e8f9;
    --secondary-dark: #0891b2;
    --accent: #06b6d4;
    --accent-light: #67e8f9;
    --accent-dark: #0891b2;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --dark-bg: #020617;
    --darker-bg: #010409;
    --card-bg: rgba(15, 23, 42, 0.6);
    --card-bg-light: rgba(15, 23, 42, 0.4);
    --border: rgba(148, 163, 184, 0.2);
    --border-light: rgba(148, 163, 184, 0.1);
    --text-primary: #e2e8f0;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global styles */
html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--dark-bg);
    color: var(--text-primary);
}

/* Main app container */
[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(1200px 600px at 20% -10%, rgba(99, 102, 241, 0.15), transparent 50%),
        radial-gradient(1000px 500px at 80% 0%, rgba(6, 182, 212, 0.1), transparent 50%),
        linear-gradient(180deg, #020617 0%, #010409 100%);
    position: relative;
    overflow-x: hidden;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.3);
}
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.4);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.6);
}

/* Hero section */
.hero {
    position: relative;
    text-align: center;
    padding: 4rem 2rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 30% 30%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 70% 70%, rgba(6, 182, 212, 0.1) 0%, transparent 50%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: var(--primary-light);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.4rem 0.9rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.hero-badge-icon {
    width: 16px;
    height: 16px;
}
.hero h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(135deg, var(--primary-light) 0%, var(--secondary-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
    position: relative;
    display: inline-block;
}
.hero h1::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: 2px;
}
.hero p.sub {
    font-size: 1.1rem;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 700px;
    margin: 0 auto 2rem;
    opacity: 0.9;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(129, 140, 248, 0.4), transparent);
    margin: 0 auto;
    width: 80%;
    max-width: 400px;
    position: relative;
}
.hero-divider::before, .hero-divider::after {
    content: '';
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 10px;
    height: 10px;
    background: var(--primary);
    border-radius: 50%;
}
.hero-divider::before { left: 0; }
.hero-divider::after { right: 0; }

/* Glassmorphism card */
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: var(--shadow-md);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    border-color: rgba(99, 102, 241, 0.3);
}
.glass-card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-light);
}
.glass-card-icon {
    width: 24px;
    height: 24px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    color: white;
}
.glass-card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.glass-card-description {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-container {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    text-align: center;
    transition: var(--transition);
    backdrop-filter: blur(10px);
}
.metric-container:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: rgba(99, 102, 241, 0.25);
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary-light), var(--secondary-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.metric-change {
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
}
.metric-change.positive { color: var(--success); }
.metric-change.negative { color: var(--error); }
.metric-change.neutral { color: var(--text-muted); }

/* Buttons */
.stButton > button {
    border-radius: var(--radius-md);
    font-weight: 600;
    transition: var(--transition);
    border: none;
    position: relative;
    overflow: hidden;
    font-size: 0.9rem;
    padding: 0.5rem 1.25rem;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.2),
        transparent
    );
    transition: 0.5s;
}
.stButton > button:hover::before {
    left: 100%;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    box-shadow: var(--shadow-md);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
}
.stButton > button[kind="secondary"] {
    background: var(--card-bg);
    color: var(--text-primary);
    border: 1px solid var(--border);
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(99, 102, 241, 0.05);
    transform: translateY(-1px);
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    font-size: 0.9rem;
    padding: 0.75rem;
    transition: var(--transition);
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary-light);
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
}
.stTextInput > label,
.stNumberInput > label,
.stTextArea > label {
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 0.25rem;
}

/* File uploader */
.stFileUploader > div {
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    background: var(--card-bg);
    transition: var(--transition);
}
.stFileUploader > div:hover {
    border-color: var(--primary-light);
    background: rgba(99, 102, 241, 0.05);
    transform: scale(1.02);
}
.stFileUploader > div > div > div > div {
    color: var(--text-muted);
}
.stFileUploader > div > div > div > div > div {
    color: var(--primary);
    font-weight: 600;
}

/* Radio buttons */
.stRadio > div {
    gap: 1rem;
}
.stRadio > div > label {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    font-weight: 500;
    transition: var(--transition);
    cursor: pointer;
}
.stRadio > div > label:hover {
    background: rgba(99, 102, 241, 0.05);
    border-color: var(--primary-light);
}
.stRadio > div > label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(6, 182, 212, 0.1));
    border: 1px solid var(--primary);
    color: var(--text-primary);
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    font-weight: 600;
    color: var(--text-primary);
    padding: 0.75rem 1rem;
}
.streamlit-expanderHeader:hover {
    background: rgba(99, 102, 241, 0.05);
}
.streamlit-expanderContent {
    background: var(--card-bg-light);
    border: 1px solid var(--border-light);
    border-radius: 0 0 var(--radius-md) var(--radius-md);
    margin-top: -1px;
    padding: 1rem;
}

/* Info/warning/error boxes */
.stAlert {
    border-radius: var(--radius-md);
    border: none;
    padding: 1rem;
    font-weight: 500;
}
.stAlert > div {
    padding: 0.5rem 1rem;
    border-radius: var(--radius-sm);
    font-weight: 500;
}
.stInfo {
    background: rgba(99, 102, 241, 0.1);
    border-left: 3px solid var(--primary);
    color: var(--primary-light);
}
.stWarning {
    background: rgba(245, 158, 11, 0.1);
    border-left: 3px solid var(--accent);
    color: var(--accent-light);
}
.stError {
    background: rgba(239, 68, 68, 0.1);
    border-left: 3px solid var(--error);
    color: var(--error-light);
}
.stSuccess {
    background: rgba(16, 185, 129, 0.1);
    border-left: 3px solid var(--success);
    color: var(--success-light);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}
[data-testid="stDataFrame"] > div {
    border: none !important;
}
[data-testid="stDataFrame"] thead th {
    background: var(--card-bg);
    color: var(--text-primary);
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    padding: 1rem;
    text-align: left;
    font-size: 0.9rem;
}
[data-testid="stDataFrame"] tbody td {
    background: var(--card-bg-light);
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: rgba(99, 102, 241, 0.05);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: var(--card-bg);
    border-radius: var(--radius-md);
    padding: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    font-weight: 500;
    padding: 0.5rem 1rem;
    transition: var(--transition);
}
.stTabs [aria-selected="true"] {
    background: var(--primary);
    color: white;
    box-shadow: var(--shadow-sm);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: var(--radius-sm);
}

/* Spinner */
.stSpinner > div {
    border-color: var(--primary) transparent var(--primary) transparent;
}
.stSpinner > div::after {
    content: '';
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--text-muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--border-light);
    margin-top: 3rem;
}
.footer a {
    color: var(--primary-light);
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-light);
}
.section-header-icon {
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    color: white;
}
.section-header-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: var(--radius-sm);
}
.badge-primary {
    background: rgba(99, 102, 241, 0.2);
    color: var(--primary-light);
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.badge-secondary {
    background: rgba(6, 182, 212, 0.2);
    color: var(--secondary-light);
    border: 1px solid rgba(6, 182, 212, 0.3);
}
.badge-accent {
    background: rgba(245, 158, 11, 0.2);
    color: var(--accent-light);
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.badge-success {
    background: rgba(16, 185, 129, 0.2);
    color: var(--success-light);
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-warning {
    background: rgba(245, 158, 11, 0.2);
    color: var(--accent-light);
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.badge-error {
    background: rgba(239, 68, 68, 0.2);
    color: var(--error-light);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Animated background elements */
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.15;
    pointer-events: none;
}
.orb-1 {
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.3), transparent 70%);
    top: -10%;
    left: -10%;
    animation: drift 15s ease-in-out infinite;
}
.orb-2 {
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.3), transparent 70%);
    bottom: -10%;
    right: -10%;
    animation: drift 20s ease-in-out infinite reverse;
}
@keyframes drift {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    33% { transform: translate(30px, -30px) rotate(120deg); }
    66% { transform: translate(-30px, 30px) rotate(240deg); }
}

/* Responsive design */
@media (max-width: 768px) {
    .hero h1 {
        font-size: 2.2rem;
    }
    .hero p.sub {
        font-size: 1rem;
    }
    .glass-card {
        padding: 1rem;
    }
    .section-header {
        margin: 1.5rem 0 1rem;
    }
    .section-header-title {
        font-size: 1.25rem;
    }
}
</style>
"""

_HERO = """
<div class="hero">
  <div class="hero-badge">
    <span class="hero-badge-icon">🔍</span>
    EVIDENCE CAPTURE ASSISTANT
  </div>
  <h1>Crime Scene AI</h1>
  <p class="sub">Upload crime-scene photos — detections, stain candidates and the narrative are a <b>suggestion for triage</b>; nothing is logged until an officer confirms it (evidence boundary).</p>
  <div class="hero-divider"></div>
</div>
"""

st.markdown(_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="bg-fx">'
    '<div class="orb orb-1"></div><div class="orb orb-2"></div>'
    '</div>',
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
        "person": (0, 255, 0),       # bright green
        "weapon": (0, 0, 255),       # red
        "vehicle": (255, 100, 0),    # blue
        "container": (0, 255, 255),  # cyan
        "personal item": (255, 0, 255), # magenta
        "electronic device": (200, 0, 200), # purple
        "discarded item": (0, 255, 200),  # teal
    }
    for d in analysis.get("objects", []):
        b = d.get("bbox")
        if not b:
            continue
        x1, y1, x2, y2 = (int(round(b[k])) for k in ("x1", "y1", "x2", "y2"))
        color = colors.get(d.get("category"), (0, 255, 128))  # green-cyan default
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


def _analyze_video(raw, filename=""):
    """Extract frames, detect objects, track persons (person1, person2, ...).

    Returns (video_analysis_dict, error_list). Frames are annotated in place
    with person ids so the same person keeps one name across the video.
    """
    import tempfile

    suffix = os.path.splitext(filename or ".mp4")[1].lower() or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(raw)
        tmp.close()
        tracked_dets, notes = video_mod.process_video(tmp.name)
    except Exception as exc:
        try: os.unlink(tmp.name)
        except OSError: pass
        return None, [f"video-unavailable: {exc}"]

    if not tracked_dets:
        try: os.unlink(tmp.name)
        except OSError: pass
        return None, notes or ["video-unavailable: no detections"]

    # Group detections by frame
    by_frame = {}
    for d in tracked_dets:
        by_frame.setdefault(d.get("frame_idx", 0), []).append(d)

    # Re-extract frames for annotation
    frames, frame_nums, note = video_mod.extract_frames(tmp.name)
    if note:
        notes = [note] + (notes or [])

    annotated = {}
    for fi, frame in zip(frame_nums, frames):
        annotated[fi] = annotate(frame, {"objects": by_frame.get(fi, []), "stains": []})

    try: os.unlink(tmp.name)
    except OSError: pass

    return {
        "frame_nums": frame_nums,
        "annotated": annotated,
        "detections": tracked_dets,
        "notes": notes,
    }, None


def _merge_video_into_case():
    """Merge tracked video detections into the scene analysis."""
    va = st.session_state.video_analysis
    if not va:
        return
    merged = st.session_state.merged
    if merged is None:
        merged = {
            "width": 0, "height": 0, "objects": [], "stains": [],
            "tamper": {}, "metadata": {}, "gps": None,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "processing_notes": [],
        }
    dets = []
    for d in va["detections"]:
        nd = dict(d)
        nd["source"] = "yolo-video"
        nd.setdefault("frame_idx", 0)
        nd["photo"] = f"video frame {nd['frame_idx']}"
        dets.append(nd)
    merged["objects"] = list(merged.get("objects", [])) + dets
    merged["processing_notes"] = list(merged.get("processing_notes", [])) + va["notes"]
    merged["processing_notes"].append(
        f"video analysis merged — {len(dets)} detections from {len(va['frame_nums'])} frames"
    )
    merged["suggestions"] = reasoning.reason(merged)
    st.session_state.merged = merged
    st.session_state.narrative = merged["suggestions"].get("narrative", "")
    if merged["metadata"] is None:
        merged["metadata"] = {}


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
    md = f'<div class="section-header"><div class="section-header-icon">📄</div><div class="section-header-title">Case `{c["id"]}` — confirmed</div></div>\n'
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
    with m1:
        st.markdown('<div class="metric-container"><div class="metric-label">Photos</div><div class="metric-value">' + str(len(images)) + '</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-container"><div class="metric-label">Objects</div><div class="metric-value">' + str(len(merged.get("objects", []))) + '</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-container"><div class="metric-label">Stain candidates</div><div class="metric-value">' + str(len(merged.get("stains", []))) + '</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-container"><div class="metric-label">Evidence markers</div><div class="metric-value">' + str(len(st.session_state.markers)) + '</div></div>', unsafe_allow_html=True)

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
    if merged and images:
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

st.markdown('<div class="section-header"><div class="section-header-icon">🎬</div><div class="section-header-title">Video Detection</div></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.caption("Upload a scene video (MP4/WebM/AVI). Frames are sampled, objects detected, and each person is tracked across frames as person1, person2, … — the same person keeps one name.")
    vfile = st.file_uploader(
        "Scene video", type=["mp4", "webm", "avi", "mov"],
        on_change=lambda: st.session_state.update(video_analysis=None),
    )
    if vfile:
        st.video(vfile)
        if st.button("Analyze video", type="primary", key="analyze_video_btn"):
            va, verr = _analyze_video(vfile.getvalue(), vfile.name)
            if va is None:
                st.error("; ".join(verr or ["video analysis failed"]))
            else:
                st.session_state.video_analysis = va
                st.success(
                    f"Analyzed {len(va['frame_nums'])} frames — {len(va['detections'])} detections, "
                    f"{len({d.get('person_id') for d in va['detections'] if d.get('person_id')})} persons tracked."
                )

    va = st.session_state.video_analysis
    if va:
        frames = va["frame_nums"]
        persons = sorted(
            {d.get("person_id") for d in va["detections"] if d.get("person_id")},
            key=lambda s: (len(s), s),
        )
        if persons:
            st.markdown("**Persons tracked:** " + " · ".join(f"`{p}`" for p in persons))
        if len(frames) > 1:
            idx = st.slider("Frame", 0, len(frames) - 1, 0, key="video_frame_slider")
        else:
            st.caption("Single frame extracted.")
            idx = 0
        fi = frames[idx]
        st.image(va["annotated"][fi], width="stretch")
        frame_dets = [d for d in va["detections"] if d.get("frame_idx") == fi]
        if frame_dets:
            st.dataframe(
                _df(frame_dets, ["class", "category", "confidence", "source", "person_id", "frame_idx"]),
                width="stretch", hide_index=True,
            )
        else:
            st.caption("No detections at this frame.")
        st.caption(f"Detections across all frames: {len(va['detections'])}")
        with st.expander("All video detections"):
            st.dataframe(
                _df(va["detections"], ["class", "category", "confidence", "person_id", "frame_idx"]),
                width="stretch", hide_index=True,
            )
        if va["notes"]:
            st.info("\n".join(f"- {n}" for n in dict.fromkeys(va["notes"])))
        if st.button("Add video detections to case", key="merge_video_btn"):
            _merge_video_into_case()
            st.success("Video detections merged into the scene analysis — review below and confirm the case.")
            st.rerun()

if merged:
    with st.container(border=True):
        st.markdown('<div class="section-header"><div class="section-header-icon">📷</div><div class="section-header-title">Scene Overview</div></div>', unsafe_allow_html=True)
        st.caption(f"photos: {len(st.session_state.images)} · "
                   f"evidence markers: {len(st.session_state.markers)}")

        st.markdown('<div class="section-header"><div class="section-header-icon">🔍</div><div class="section-header-title">Objects Detected</div></div>', unsafe_allow_html=True)
        st.dataframe(_df(merged.get("objects", []),
                         ["class", "category", "confidence", "source", "photo"]),
                    width="stretch", hide_index=True)
        st.markdown('<div class="section-header"><div class="section-header-icon">🩸</div><div class="section-header-title">Blood-like Stain Candidates</div></div>', unsafe_allow_html=True)
        st.dataframe(_df(merged.get("stains", []),
                         ["class", "confidence", "area_pct", "photo"]),
                    width="stretch", hide_index=True)

        tamper = merged.get("tamper", {})
        st.markdown(f'<div class="section-header"><div class="section-header-icon">🔒</div><div class="section-header-title">Tamper Check</div></div>', unsafe_allow_html=True)
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
            st.markdown('<div class="section-header"><div class="section-header-icon">🤖</div><div class="section-header-title">AI Observations & Suggestions</div></div>', unsafe_allow_html=True)
            st.caption(f"Draft by: {mode_txt} — suggestions only, never evidence; an officer signs the final case.")
            flags = sug.get("anomaly_flags") or []
            steps = sug.get("next_steps") or []
            if flags:
                st.markdown('<div class="section-header"><div class="section-header-icon">⚠️</div><div class="section-header-title">Anomaly Flags</div></div>', unsafe_allow_html=True)
                st.markdown("**Anomaly flags:**\n" + "\n".join(f"- {f}" for f in flags))
            if steps:
                st.markdown('<div class="section-header"><div class="section-header-icon">➡️</div><div class="section-header-title">Suggested Next Steps</div></div>', unsafe_allow_html=True)
                st.markdown("**Suggested next steps:**\n" + "\n".join(f"- {s}" for s in steps))
            if sug.get("narrative"):
                with st.expander("AI-drafted observation report"):
                    st.write(sug["narrative"])
                    if st.button("Use as narrative draft", key="use_ai_narrative"):
                        st.session_state.narrative = sug["narrative"]
                        st.rerun()

        st.markdown('<div class="section-header"><div class="section-header-icon">✅</div><div class="section-header-title">Officer Confirmation</div></div>', unsafe_allow_html=True)
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

st.markdown('<div class="section-header"><div class="section-header-icon">📁</div><div class="section-header-title">Past Cases</div></div>', unsafe_allow_html=True)
with st.container(border=True):
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