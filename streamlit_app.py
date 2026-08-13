"""Streamlit Community Cloud front-end for Crime Scene AI.

Human-in-the-loop demo: upload one or more photos -> explainable triage draft
(objects, stain candidates, AI observation & suggestions, offline-safe) ->
click-to-drop numbered evidence markers + a two-click reference scale for
approximate size estimates (cm) -> officer writes/confirms the narrative ->
confirmed report logged to the case file with audit trail and pattern
matches -> JSON/PDF report export.
"""

import json
import math
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
from streamlit_image_coordinates import streamlit_image_coordinates

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
    "scale": None,
    "scale_start": None,
    "measurements": [],
    "measure_seq": 0,
    "photo_view": None,
    "_canvas_cache": {},
}
CANVAS_W = 1100
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


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


def draw_scale(img, scale, photo):
    if not scale or scale.get("photo") != photo:
        return
    (x1, y1), (x2, y2) = scale["start"], scale["end"]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.circle(img, (x1, y1), 6, (255, 0, 255), -1)
    cv2.circle(img, (x2, y2), 6, (255, 0, 255), -1)
    cm = scale.get("known_cm")
    if cm:
        label = f"{cm:g} cm  /  {scale['px_len']} px"
    else:
        label = f"line: {scale['px_len']} px"
    _pin_chip(img, (x1 + x2) // 2, (y1 + y2) // 2, label, (255, 0, 255))


def draw_measurements(img, measurements, photo, pxcm):
    for m in measurements:
        if m.get("photo") != photo:
            continue
        (x1, y1), (x2, y2) = m["start"], m["end"]
        color = (255, 215, 0)
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (x1, y1), 5, color, -1)
        cv2.circle(img, (x2, y2), 5, color, -1)
        cm = m.get("cm")
        label = f"{cm:.1f} cm" if cm else f"{m['px_len']} px"
        _pin_chip(img, (x1 + x2) // 2, (y1 + y2) // 2, label, color)


def canvas_bgr(photo):
    img = st.session_state.images[photo]["bgr"].copy()
    img = annotate(img, st.session_state.analyses[photo])
    draw_pins(img, st.session_state.markers, photo)
    pxcm = (st.session_state.scale or {}).get("px_per_cm")
    draw_measurements(img, st.session_state.measurements, photo, pxcm)
    draw_scale(img, st.session_state.scale, photo)
    return img


def _auto_measure(photo, img_bgr):
    """Auto-detect straight lines (Hough) and return px-length measurements."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                           minLineLength=40, maxLineGap=12)
    if segs is None:
        return []
    segs = np.asarray(segs, dtype=float)
    if segs.ndim == 3:
        segs = segs.reshape(-1, 4)
    if segs.ndim != 2 or segs.shape[1] != 4 or not len(segs):
        return []

    def _dup(k, px, py):
        (ax, ay), (bx, by) = k["start"], k["end"]
        l = math.hypot(bx - ax, by - ay) or 1
        dist = abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / l
        ang = abs(k["angle"] - math.degrees(math.atan2(by - ay, bx - ax)) % 180)
        return min(ang, 180 - ang) <= 8 and dist <= 30

    kept = []
    for (x1, y1, x2, y2) in segs:
        px_len = int(round(math.hypot(x2 - x1, y2 - y1)))
        if px_len < 40:
            continue
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if any(_dup(k, mx, my) for k in kept):
            continue
        kept.append({
            "photo": photo, "start": (int(x1), int(y1)), "end": (int(x2), int(y2)),
            "px_len": px_len,
            "angle": float(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180),
        })
    kept.sort(key=lambda k: k["px_len"], reverse=True)
    for m in kept:
        m.pop("angle", None)
    return kept[:10]


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
    st.session_state.scale = None
    st.session_state.scale_start = None
    st.session_state.measurements = []
    st.session_state.measure_seq = 0
    for name, im in images.items():
        for m in _auto_measure(name, im["bgr"]):
            st.session_state.measure_seq += 1
            m["id"] = st.session_state.measure_seq
            m["ts"] = datetime.now(timezone.utc).isoformat()
            st.session_state.measurements.append(m)
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
        "scale": st.session_state.scale,
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
                    "evidence_markers": len(payload["evidence_markers"]),
                    "scale_set": bool(payload.get("scale") and payload["scale"].get("known_cm"))}},
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
    scale = c.get("scale")
    if scale and scale.get("known_cm"):
        md += (f"- Scale reference on `{scale['photo']}`: {scale['px_len']} px = "
               f"{scale['known_cm']} cm — size estimates approximate\n")
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


def _sized(items, scale):
    pxcm = (scale or {}).get("px_per_cm")
    if not pxcm:
        return items
    out = []
    for it in items:
        r = dict(it)
        b = r.get("bbox")
        if b:
            r["w_cm"] = round((b["x2"] - b["x1"]) / pxcm, 1)
            r["h_cm"] = round((b["y2"] - b["y1"]) / pxcm, 1)
        out.append(r)
    return out


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
        "scale": detail.get("scale"),
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

    scale = detail.get("scale")
    if scale and scale.get("known_cm"):
        sec("Scale reference")
        line(f"Photo: {scale['photo']}  |  {scale['px_len']} px = {scale['known_cm']} cm  |  "
             f"{scale['px_per_cm']:.2f} px/cm (approximate, plane-dependent)")

    sec("Objects detected")
    pdf.cell(22, 5, "Photo", border=1)
    pdf.cell(36, 5, "Class", border=1)
    pdf.cell(30, 5, "Category", border=1)
    pdf.cell(16, 5, "Conf", border=1)
    pdf.cell(18, 5, "w (cm)", border=1)
    pdf.cell(18, 5, "h (cm)", border=1, new_x="LMARGIN", new_y="NEXT")
    for o in detail.get("objects", []):
        pdf.cell(22, 5, _pdf_text(o.get("photo", "")), border=1)
        pdf.cell(36, 5, _pdf_text(o.get("class", "?")), border=1)
        pdf.cell(30, 5, _pdf_text(o.get("category", "")), border=1)
        pdf.cell(16, 5, _pdf_text(_pct(o.get("confidence", ""))), border=1)
        pdf.cell(18, 5, _pdf_text(o.get("w_cm", "")), border=1)
        pdf.cell(18, 5, _pdf_text(o.get("h_cm", "")), border=1, new_x="LMARGIN", new_y="NEXT")

    sec("Blood-like stain candidates")
    pdf.cell(22, 5, "Photo", border=1)
    pdf.cell(60, 5, "Class", border=1)
    pdf.cell(16, 5, "Conf", border=1)
    pdf.cell(18, 5, "Area%", border=1)
    pdf.cell(18, 5, "w (cm)", border=1)
    pdf.cell(18, 5, "h (cm)", border=1, new_x="LMARGIN", new_y="NEXT")
    for s in detail.get("stains", []):
        pdf.cell(22, 5, _pdf_text(s.get("photo", "")), border=1)
        pdf.cell(60, 5, _pdf_text(s.get("class", "?")), border=1)
        pdf.cell(16, 5, _pdf_text(_pct(s.get("confidence", ""))), border=1)
        pdf.cell(18, 5, _pdf_text(s.get("area_pct", "")), border=1)
        pdf.cell(18, 5, _pdf_text(s.get("w_cm", "")), border=1)
        pdf.cell(18, 5, _pdf_text(s.get("h_cm", "")), border=1, new_x="LMARGIN", new_y="NEXT")

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
    st.session_state.scale = None
    st.session_state.scale_start = None
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
    elif mode == "Set scale reference":
        ss = st.session_state.get("scale_start")
        if ss is None or ss.get("photo") != photo:
            st.session_state.scale_start = {"photo": photo, "x": x, "y": y}
        else:
            px_len = int(round(math.hypot(x - ss["x"], y - ss["y"])))
            if px_len < 10:
                st.warning("Pick the two ends at least ~10 px apart, then re-click the first end.")
                st.session_state.scale_start = None
            else:
                st.session_state.scale = {
                    "photo": photo, "start": (ss["x"], ss["y"]), "end": (x, y),
                    "px_len": px_len, "known_cm": None, "px_per_cm": None,
                }
                st.session_state.scale_start = None


st.markdown(
    "# Crime Scene AI — Smart Evidence Capture Assistant\n"
    "Upload **one or more** crime-scene photos. Detections, stain candidates and "
    "the narrative are a **suggestion for triage** — nothing is logged until you "
    "confirm it (evidence boundary)."
)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader(
        "Scene photo(s)", type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True, on_change=_reset_derived,
    )
    files = [(f.name, f.getvalue()) for f in uploaded] if uploaded else []
    officer_id = st.text_input("Officer ID", placeholder="e.g. SUB-INSP-07")
    lat_in = st.number_input("Latitude (opt.)", value=None, format="%.6f")
    lng_in = st.number_input("Longitude (opt.)", value=None, format="%.6f")
    analyze_btn = st.button("Analyze all photos", type="primary")

if analyze_btn:
    if not files:
        st.error("Upload at least one photo first.")
    elif analyze_all(files, officer_id, lat_in, lng_in):
        st.success(f"Analyzed {len(files)} photo(s) — review the overlay, add markers/scale, then confirm the case.")

with col2:
    merged = st.session_state.merged
    images = st.session_state.images
    if merged:
        names = list(images)
        photo = st.session_state.photo_view
        if photo not in names:
            photo = names[0]
        if len(names) > 1:
            photo = st.radio("View photo", names, index=names.index(photo), horizontal=True, key="photo_sel")
        st.session_state.photo_view = photo
        mode = st.radio(
            "Canvas mode", ["View overlay", "Add evidence markers", "Set scale reference"],
            horizontal=True, key="canvas_mode",
        )
        canvas = canvas_bgr(photo)
        _s = st.session_state.scale
        if _s and _s.get("photo") == photo:
            _k = st.session_state.get(f"known_cm_{photo}", 10.0) or 10.0
            _s["known_cm"] = float(_k)
            _s["px_per_cm"] = _s["px_len"] / float(_k)
        oh, ow = canvas.shape[:2]
        tw = min(CANVAS_W, ow)
        th = max(1, int(round(oh * tw / ow)))
        finger = "|".join((
            photo, mode,
            ";".join(f"{m['id']}:{m.get('photo', '')}:{m['x']}:{m['y']}" for m in st.session_state.markers),
            ";".join(f"{m['id']}:{m.get('photo', '')}:{m['px_len']}" for m in st.session_state.measurements),
            repr(st.session_state.scale),
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
            st.image(cache["bytes"])
        else:
            if mode == "Add evidence markers":
                st.caption("Click anywhere on the photo to drop a numbered evidence marker.")
            elif mode == "Set scale reference":
                if st.session_state.scale_start is None:
                    st.caption(f"Step 1/3: click the first end of a known-length line on '{photo}'.")
                else:
                    st.caption("Step 2/3: click the second end of the line (must be a different point).")
            streamlit_image_coordinates(
                cache["rgb"], width=tw, height=th, key=f"cv_{photo}",
                on_click=_on_canvas_click, image_format="JPEG", jpeg_quality=85,
            )
        scale = st.session_state.scale
        if scale and scale.get("photo") == photo:
            st.markdown(f"**Scale reference** on `{photo}`: the selected line is "
                        f"{scale['px_len']} px long (magenta, drawn on the photo).")
            known_cm = st.number_input(
                "Known length of the reference line (cm)", min_value=0.1,
                value=10.0, step=1.0, key=f"known_cm_{photo}",
            )
            scale["known_cm"] = float(known_cm)
            scale["px_per_cm"] = scale["px_len"] / float(known_cm)
            pxcm = scale["px_per_cm"]
            for m in st.session_state.measurements:
                if m.get("photo") == photo:
                    m["cm"] = round(m["px_len"] / pxcm, 1)
            st.caption(f"Selected line = **{known_cm:g} cm** -> {scale['px_per_cm']:.2f} px/cm. "
                       "Sizes assume objects lie near the calibration plane (approximate). "
                       "Detected lines are measured automatically (gold, drawn on the photo).")
            if st.button("Clear scale reference", key=f"clr_scale_{photo}"):
                st.session_state.scale = None
                st.session_state.scale_start = None
                for m in st.session_state.measurements:
                    if m.get("photo") == photo:
                        m["cm"] = None
                st.rerun()
        measurements = st.session_state.measurements
        if measurements:
            st.markdown("### Auto-measured lines")
            st.caption("Straight lines detected automatically — lengths in cm once a scale reference is set.")
            for m in list(measurements):
                r1, r2 = st.columns([4, 1])
                r1.caption(f"#{m['id']} · {m['photo']} · **{m['cm']:.1f} cm** ({m['px_len']} px)")
                if r2.button(f"Remove #{m['id']}", key=f"ms_del_{m['id']}"):
                    st.session_state.measurements = [
                        x for x in st.session_state.measurements if x["id"] != m["id"]
                    ]
                    st.rerun()
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
    scale = st.session_state.scale
    pxcm = (scale or {}).get("px_per_cm")
    size_headers = ["w_cm", "h_cm"] if pxcm else []
    st.caption(f"photos: {len(st.session_state.images)} · "
               f"evidence markers: {len(st.session_state.markers)}")

    st.subheader("Objects detected")
    st.dataframe(_df(_sized(merged.get("objects", []), scale),
                     ["class", "category", "confidence", "source", "photo"] + size_headers),
                 width="stretch", hide_index=True)
    st.subheader("Blood-like stain candidates")
    st.dataframe(_df(_sized(merged.get("stains", []), scale),
                     ["class", "confidence", "area_pct", "photo"] + size_headers),
                 width="stretch", hide_index=True)
    if pxcm:
        st.caption("w_cm / h_cm estimated from the reference scale — approximate, plane-dependent.")

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
    st.caption(f"Marks {len(st.session_state.markers)} evidence markers · "
               f"scale {'set' if scale and scale.get('known_cm') else 'not set'}")
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