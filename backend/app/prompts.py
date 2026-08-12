"""Prompt construction for the reasoning layer.

The system prompt is the explainability contract: the LLM must never
fabricate, must ground every statement in the supplied detections, must flag
uncertainty, and must return ONLY the fixed JSON schema. The canonical
narrative is English (with a short Hindi rendering) so the DB stores one
canonical form while clients translate at render time.
"""

SYSTEM_PROMPT = """You are a forensic scene-analysis assistant used by a police field officer. You receive structured computer-vision detections from a crime-scene photograph. Your job is to draft a PRELIMINARY OBSERVATION — not a verdict, not evidence.

Rules — follow strictly:
1. Ground every statement in the detections provided. Never invent objects, text, or locations not in the input.
2. If the evidence is sparse or ambiguous, SAY SO explicitly ("insufficient evidence to ..."). Hedging is mandatory, not optional.
3. Confidence is already attached to each detection — do not restate every number; use it only to qualify ("high-confidence knife detection" vs "possible red stain").
4. Do not mention the model, YOLO, or any AI internals in the narrative. Write like a cautious investigator's note.
5. Output ONLY a JSON object, no markdown fences, no commentary. Schema — exact keys only:
{
  "narrative": "string, 2-6 sentences, factual and cautious",
  "narrative_hi": "string, the same narrative translated to Hindi",
  "anomaly_flags": ["string", ...],   // 0-4 short flags, e.g. "weapon + stain co-located in same frame"
  "next_steps": ["string", ...]        // 2-5 concrete actions for the investigator
}
6. If any detection is categorised 'stain', note that color-based detection cannot distinguish blood from paint/dye. If tamper flag is 'edit-likelihood', the top anomaly flag must warn about image integrity."""

USER_PROMPT_TEMPLATE = """Scene analysis request. Evidence-handling context: a field officer photographed a potential crime scene with a standard phone camera.

Detected objects (source: yolo object detection; source 'hsv-heuristic' = color-based stain candidates only, NOT blood confirmation):
{objects}

OCR text regions:
{ocr}

Image-integrity check:
{tamper}

Capture metadata:
{metadata}

Produce the preliminary observation report per your system rules. Remember: this is a draft for a human officer to review and sign — err on the side of caution and explicit uncertainty."""


def build_user_prompt(analysis: dict) -> str:
    def fmt_bbox(b): return f"at ({b.get('x1'):.0f},{b.get('y1'):.0f})-({b.get('x2'):.0f},{b.get('y2'):.0f})"  # noqa: E731

    objects_txt = "\n".join(
        f"- {d.get('class')} [{d.get('category')}] conf={d.get('confidence')} {fmt_bbox(d.get('bbox', {}))}" +
        (" (candidate only, colour-based)" if d.get("source") == "hsv-heuristic" else "") +
        (" ocr-sourced" if d.get("source") == "ocr" else "")
        for d in analysis.get("objects", []) + analysis.get("stains", []) + analysis.get("ocr", [])
    ) or "- none detected"

    tamper = analysis.get("tamper", {})
    tamper_txt = (
        f"flag={tamper.get('flag')}, ela_score={tamper.get('ela_score')} vs threshold {tamper.get('threshold')}\n"
        + "\n".join(f"  note: {n}" for n in tamper.get("notes", []))
    )

    meta = analysis.get("metadata", {})
    meta_txt = (
        f"has_exif={meta.get('has_exif')}, gps={meta.get('gps')}, captured_at={meta.get('captured_at')}\n"
        + "\n".join(f"  note: {n}" for n in meta.get("notes", []))
    )

    return USER_PROMPT_TEMPLATE.format(
        objects=objects_txt,
        ocr="\n".join(f"- '{o.get('text')}' (conf {o.get('confidence')})" for o in analysis.get("ocr", [])) or "- none",
        tamper=tamper_txt,
        metadata=meta_txt,
    )