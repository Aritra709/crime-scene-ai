"""Streamlit Community Cloud front-end for Crime Scene AI.

Human-in-the-loop demo: upload photo → explainable analysis draft → officer
edits/confirms narrative → confirmed report logged to the case file with
audit trail → pattern matches against past cases.
"""

import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app import config, db
from app.pipeline import run_pipeline

db.init_db()

st.set_page_config(page_title="Crime Scene AI — Evidence Capture Assistant", layout="wide")

if "analysis" not in st.session_state:
    st.session_state.analysis = None
    st.session_state.image_id = None
    st.session_state.narrative = ""
    st.session_state.overlay_bytes = None


def _fmt_basis(basis):
    if not basis:
        return ""
    if isinstance(basis, str):
        return basis
    return "; ".join(str(b) for b in basis)


def _df(rows, headers):
    if not rows:
        return pd.DataFrame(columns=headers)
    data = []
    for r in rows:
        row = {}
        for h in headers:
            v = r.get(h, "")
            row[h] = ", ".join(v) if isinstance(v, list) else v
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
        _label_chip(out, x1, y1, x2, y2, f"{d.get('class', '?')} {d.get('confidence', 0):.2f}", color)
    for s in analysis.get("stains", []):
        b = s.get("bbox")
        if not b:
            continue
        x1, y1, x2, y2 = (int(round(b[k])) for k in ("x1", "y1", "x2", "y2"))
        _dashed_rect(out, x1, y1, x2, y2, (0, 0, 220), thickness=2)
        _label_chip(out, x1, y1, x2, y2, f"stain {s.get('confidence', 0):.2f}", (0, 0, 220))
    return out


def analyze(raw: bytes, officer_id, lat, lng):
    try:
        analysis = run_pipeline(raw)
    except ValueError as exc:
        st.error(str(exc))
        return None

    image_id = uuid.uuid4().hex[:12]
    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(config.IMAGES_DIR / f"{image_id}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated = annotate(img_bgr, analysis)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    st.session_state.overlay_bytes = buf.tobytes() if ok else None

    gps = analysis["metadata"].get("gps")
    if gps is None and lat and lng:
        gps = {"lat": float(lat), "lng": float(lng)}
    captured_at = analysis["metadata"].get("captured_at") or datetime.now(timezone.utc).isoformat()
    analysis["image_id"] = image_id
    analysis["gps"] = gps
    analysis["captured_at"] = captured_at

    st.session_state.analysis = analysis
    st.session_state.image_id = image_id
    st.session_state.narrative = analysis.get("llm", {}).get("narrative", "")
    return analysis


def confirm(officer_id, narrative, lat, lng):
    a = st.session_state.analysis
    if not a:
        st.error("Analyze an image first.")
        return
    gps = a.get("gps")
    if gps is None and lat and lng:
        gps = {"lat": float(lat), "lng": float(lng)}
    llm = a.get("llm", {})
    payload = {
        "officer_id": officer_id or "unknown",
        "image_id": st.session_state.image_id,
        "gps": gps,
        "captured_at": a.get("captured_at"),
        "narrative": narrative,
        "original_narrative": llm.get("narrative", ""),
        "next_steps": llm.get("next_steps", []),
        "anomaly_flags": llm.get("anomaly_flags", []),
        "objects": a.get("objects", []),
        "stains": a.get("stains", []),
        "ocr": a.get("ocr", []),
        "tamper": a.get("tamper", {}),
        "metadata": a.get("metadata", {}),
        "llm_source": llm.get("source", "mock"),
        "processing_notes": a.get("processing_notes", []),
    }
    log_entries = [
        {"actor": "system", "action": "case-opened",
         "detail": {"image_id": st.session_state.image_id, "officer_id": officer_id}},
        {"actor": "ai", "action": "analysis-draft-generated",
         "detail": {"source": payload["llm_source"],
                    "objects": len(payload["objects"]), "stains": len(payload["stains"])}},
        {"actor": "officer", "action": "confirmed-edited",
         "detail": {"narrative_changed": payload["narrative"] != payload["original_narrative"],
                    "objects_confirmed": len(payload["objects"]),
                    "next_steps": len(payload["next_steps"])}},
    ]
    case_id = db.insert_case(payload, log_entries)
    detail = db.get_case(case_id)
    detail["matches"] = db.find_matches(case_id)
    st.session_state.last_case = detail


def case_card(c):
    md = f"## Case `{c['id']}` — confirmed\n"
    md += f"- Officer: `{c['officer_id']}` · logged {c['created_at']} · LLM source `{c['llm_source']}`\n"
    md += f"- Narrative: {c['narrative']}\n"
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


st.markdown(
    "# Crime Scene AI — Smart Evidence Capture Assistant\n"
    "Upload a crime-scene photo. The draft is a **suggestion for triage** — "
    "nothing is logged until you confirm it (evidence boundary)."
)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Scene photo", type=["jpg", "jpeg", "png", "webp", "bmp"])

with col2:
    if uploaded:
        if st.session_state.get("overlay_bytes"):
            st.caption("AI triage overlay — boxes are suggestions, not evidence")
            st.image(st.session_state.overlay_bytes, width="stretch")
        else:
            st.image(uploaded, width="stretch")

with col1:
    officer_id = st.text_input("Officer ID", placeholder="e.g. SUB-INSP-07")
    lat_in = st.number_input("Latitude (opt.)", value=None, format="%.6f")
    lng_in = st.number_input("Longitude (opt.)", value=None, format="%.6f")
    analyze_btn = st.button("Analyze scene", type="primary")

if analyze_btn:
    if not uploaded:
        st.error("Upload a photo first.")
    else:
        analyze(uploaded.getvalue(), officer_id, lat_in, lng_in)

a = st.session_state.analysis
if a:
    llm = a.get("llm", {})
    gps = a.get("gps")
    st.caption(f"image_id `{a.get('image_id')}` · GPS: {gps if gps else 'none (type lat/lng to attach)'} · "
               f"captured_at {a.get('captured_at', 'n/a')} · LLM source: {llm.get('source', 'mock')}")

    st.subheader("Objects detected")
    st.dataframe(_df(a["objects"], ["class", "category", "confidence", "source", "basis"]),
                 width="stretch", hide_index=True)
    st.subheader("Blood-like stain candidates")
    st.dataframe(_df(a["stains"], ["class", "confidence", "area_pct", "source", "basis"]),
                 width="stretch", hide_index=True)
    st.subheader("OCR-extracted text")
    st.dataframe(_df(a["ocr"], ["text", "confidence", "source"]),
                 width="stretch", hide_index=True)

    st.markdown(f"**Narrative (EN):** {llm.get('narrative', '—')}")
    if llm.get("narrative_hi"):
        st.markdown(f"**नैरेटिव (हिंदी):** {llm['narrative_hi']}")
    st.markdown("**Anomaly flags:**\n" + "\n".join(f"- {x}" for x in llm.get("anomaly_flags", []) or ["none"]))
    st.markdown("**Suggested next steps:**\n" + "\n".join(f"- {x}" for x in llm.get("next_steps", []) or ["none"]))
    tamper = a.get("tamper", {})
    st.markdown(f"**Tamper check:** `{tamper.get('flag', 'inconclusive')}` "
                f"(ELA score {tamper.get('ela_score', 'n/a')}) — heuristic, never a verdict")
    st.caption("**Processing notes:**\n" + "\n".join(f"- {n}" for n in a.get("processing_notes", []) or ["none"]))

    st.markdown("## Officer confirmation (human-in-the-loop)")
    narrative = st.text_area("Narrative (edit before confirming)", value=st.session_state.narrative, height=120)
    st.session_state.narrative = narrative
    if st.button("Confirm & log case", type="primary"):
        confirm(officer_id, narrative, lat_in, lng_in)

if st.session_state.get("last_case"):
    st.markdown(case_card(st.session_state.last_case))

st.markdown("## Past cases")
rows = []
for c in db.list_cases():
    rows.append({"id": c["id"], "officer": c["officer_id"], "created_at": c["created_at"],
                 "llm_source": c["llm_source"], "objects": c["object_count"],
                 "stains": c["stain_count"],
                 "next_steps": len(c["next_steps"]), "flags": str(c["anomaly_flags"])})
if rows:
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
else:
    st.caption("No cases logged yet.")