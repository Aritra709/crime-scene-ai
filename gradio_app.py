"""Gradio Space front-end for Crime Scene AI.

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
import gradio as gr
import numpy as np

from app import config, db
from app.pipeline import run_pipeline

db.init_db()


def _table(rows, headers):
    if not rows:
        return [[]], headers
    data = []
    for r in rows:
        data.append([r.get(h, "") for h in headers])
    return data, headers


def analyze(image_path, officer_id, lat, lng):
    if not image_path:
        raise gr.Error("Upload a photo first.")
    with open(image_path, "rb") as f:
        raw = f.read()
    try:
        analysis = run_pipeline(raw)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    image_id = uuid.uuid4().hex[:12]
    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(config.IMAGES_DIR / f"{image_id}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    gps = analysis["metadata"].get("gps")
    if gps is None and lat is not None and lng is not None:
        gps = {"lat": float(lat), "lng": float(lng)}
    captured_at = analysis["metadata"].get("captured_at") or datetime.now(timezone.utc).isoformat()
    analysis["image_id"] = image_id
    analysis["gps"] = gps
    analysis["captured_at"] = captured_at

    llm = analysis.get("llm", {})
    obj_rows, _ = _table(analysis["objects"], ["class", "category", "confidence", "source", "basis"])
    stn_rows, _ = _table(analysis["stains"], ["class", "confidence", "area_pct", "source", "basis"])
    ocr_rows, _ = _table(analysis["ocr"], ["text", "confidence", "source"])

    nar_md = f"**Narrative (EN):** {llm.get('narrative', '—')}\n\n"
    if llm.get("narrative_hi"):
        nar_md += f"**नैरेटिव (हिंदी):** {llm['narrative_hi']}\n\n"
    flags_md = "**Anomaly flags:**\n" + "\n".join(f"- {x}" for x in llm.get("anomaly_flags", []) or ["none"]) + "\n\n"
    flags_md += "**Suggested next steps:**\n" + "\n".join(f"- {x}" for x in llm.get("next_steps", []) or ["none"])
    tamper = analysis.get("tamper", {})
    tamper_md = f"**Tamper check:** `{tamper.get('flag', 'inconclusive')}` (ELA score {tamper.get('ela_score', 'n/a')}) — heuristic, never a verdict"
    notes_md = "\n".join(f"- {n}" for n in analysis.get("processing_notes", [])) or "- none"
    gps_md = f"GPS: {gps}" if gps else "GPS: none (type lat/lng to attach)"

    return (
        analysis,
        analysis["image_id"],
        obj_rows,
        stn_rows,
        ocr_rows,
        nar_md,
        flags_md,
        tamper_md,
        notes_md,
        gps_md,
        llm.get("narrative", ""),
        f"LLM source: {llm.get('source', 'mock')}",
    )


def confirm(state, image_id, officer_id, narrative, lat, lng):
    if not state:
        raise gr.Error("Analyze an image first.")
    a = state
    gps = a.get("gps")
    if gps is None and lat is not None and lng is not None:
        gps = {"lat": float(lat), "lng": float(lng)}
    llm = a.get("llm", {})
    payload = {
        "officer_id": officer_id or "unknown",
        "image_id": image_id,
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
         "detail": {"image_id": image_id, "officer_id": officer_id}},
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
    return _case_card(detail), _case_list_table()


def _case_card(c):
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


def _case_list_table():
    rows = []
    for c in db.list_cases():
        rows.append([c["id"], c["officer_id"], c["created_at"], c["llm_source"],
                     c["object_count"], c["stain_count"], len(c["next_steps"]), c["anomaly_flags"]])
    headers = ["id", "officer", "created_at", "llm_source", "objects", "stains", "next_steps", "flags"]
    return (rows or [[]]), headers


with gr.Blocks(title="Crime Scene AI — Evidence Capture Assistant") as demo:
    state = gr.State(None)
    gr.Markdown(
        "# Crime Scene AI — Smart Evidence Capture Assistant\n"
        "Upload a crime-scene photo. The draft is a **suggestion for triage** — "
        "nothing is logged until you confirm it (evidence boundary)."
    )
    with gr.Row():
        with gr.Column(scale=2):
            img_in = gr.Image(type="filepath", label="Scene photo", height=320)
            officer_id = gr.Textbox(label="Officer ID", placeholder="e.g. SUB-INSP-07", value="")
            with gr.Row():
                lat_in = gr.Number(label="Latitude (opt.)", precision=6)
                lng_in = gr.Number(label="Longitude (opt.)", precision=6)
            analyze_btn = gr.Button("Analyze scene", variant="primary")
        with gr.Column(scale=3):
            image_id_out = gr.Textbox(label="image_id", interactive=False, visible=False)
            gps_out = gr.Textbox(label="Capture metadata", interactive=False)
            obj_tbl = gr.Dataframe(headers=["class", "category", "confidence", "source", "basis"],
                                   label="Objects detected", interactive=False, wrap=True)
            stn_tbl = gr.Dataframe(headers=["class", "confidence", "area_pct", "source", "basis"],
                                   label="Blood-like stain candidates", interactive=False, wrap=True)
            ocr_tbl = gr.Dataframe(headers=["text", "confidence", "source"],
                                   label="OCR-extracted text", interactive=False, wrap=True)
    nar_md = gr.Markdown()
    flags_md = gr.Markdown()
    tamper_md = gr.Markdown()
    notes_md = gr.Markdown("**Processing notes:**")
    llm_source_out = gr.Textbox(label="Reasoning source", interactive=False)

    gr.Markdown("## Officer confirmation (human-in-the-loop)")
    with gr.Row():
        narrative_in = gr.Textbox(label="Narrative (edit before confirming)", lines=4)
        confirm_btn = gr.Button("Confirm & log case", variant="primary")

    gr.Markdown("## Past cases")
    with gr.Row():
        cases_tbl = gr.Dataframe(label="Case log", interactive=False, wrap=True)
        refresh_btn = gr.Button("Refresh")
    case_card = gr.Markdown()

    analyze_btn.click(
        analyze,
        [img_in, officer_id, lat_in, lng_in],
        [state, image_id_out, obj_tbl, stn_tbl, ocr_tbl, nar_md, flags_md, tamper_md, notes_md, gps_out, narrative_in, llm_source_out],
    )
    confirm_btn.click(
        confirm,
        [state, image_id_out, officer_id, narrative_in, lat_in, lng_in],
        [case_card, cases_tbl],
    )
    refresh_btn.click(_case_list_table, None, cases_tbl)
    demo.load(_case_list_table, None, cases_tbl)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))