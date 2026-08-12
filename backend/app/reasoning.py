"""Reasoning layer: structured detections → human-readable draft report.

Two interchangeable backends:
- 'openai' — any OpenAI-compatible /v1/chat/completions endpoint (works with
  OpenAI, Azure, DeepSeek, OpenRouter, local Ollama...). Set OPENAI_API_KEY.
- 'mock'   — deterministic rule-driven template. Fully offline; the demo
  story for low-bandwidth field conditions. Labeled 'mock' and never
  disguised as a real LLM.

Output JSON schema is fixed (narrative, narrative_hi, anomaly_flags,
next_steps) — same schema from both backends, so the UI is backend-agnostic.
"""

import json
import urllib.request

from . import config
from . import prompts

EXPECTED_KEYS = {"narrative", "narrative_hi", "anomaly_flags", "next_steps"}


def _mock_reason(analysis: dict) -> dict:
    objs = analysis.get("objects", [])
    stains = analysis.get("stains", [])
    ocr_items = analysis.get("ocr", [])
    tamper = analysis.get("tamper", {})
    meta = analysis.get("metadata", {})

    weapons = [o for o in objs if o.get("category") == "weapon"]
    vehicles = [o for o in objs if o.get("category") == "vehicle"]
    flags: list[str] = []
    steps: list[str] = []
    narr_parts: list[str] = []
    narr_hi: list[str] = []

    if weapons:
        names = ", ".join(w["class"] for w in weapons)
        narr_parts.append(f"{len(weapons)} weapon-class object(s) detected in frame: {names}.")
        narr_hi.append(f"फ्रेम में {len(weapons)} वस्तु(एँ) शस्त्र-श्रेणी में पाई गईं: {names}।")
    if stains:
        biggest = stains[0]
        narr_parts.append(
            f"A red-dominant stain candidate covers ~{biggest['area_pct']:.1f}% of the frame "
            f"(confidence {biggest['confidence']:.2f}). This is a COLOR-BASED CANDIDATE ONLY — "
            f"it cannot be confirmed as blood without lab analysis."
        )
        narr_hi.append(
            f"लाल-प्रधान दाग ~{biggest['area_pct']:.1f}% फ्रेम क्षेत्र में मिला। यह केवल रंग-आधारित संभावित नमूना है — प्रयोगशाला जाँच के बिना रक्त की पुष्टि नहीं होती।"
        )
        flags.append("red-dominant stain candidate present — lab confirmation required")
    if ocr_items:
        texts = ", ".join(f'"{o["text"]}"' for o in ocr_items[:3])
        narr_parts.append(f"Visible text region(s): {texts}.")
        narr_hi.append(f"दिखाई देने वाला पाठ: {texts}।")
    if not narr_parts:
        narr_parts.append("No objects, stain candidates or text detected above thresholds; scene appears unremarkable at triage level.")
        narr_hi.append("सीमा स्तर से ऊपर कोई वस्तु, दाग या पाठ नहीं मिला; प्रारंभिक परीक्षण में दृश्य सामान्य प्रतीत होता है।")

    if weapons and stains:
        flags.append("weapon-class object co-located with stain candidate in the SAME frame — prioritise this scene")
    if vehicles:
        flags.append("vehicle present — possible involvement, verify ownership/plates")
    if not meta.get("has_exif"):
        flags.append(f"no EXIF metadata ({meta.get('notes', [''])[0]}) — GPS/timestamp not verifiable from file")
    if tamper.get("flag") == "edit-likelihood":
        flags.insert(0, "image-integrity risk flagged by ELA — verify source before relying on this photo")

    if weapons or stains:
        steps.append("Secure and photograph the scene with a scale reference before any item is moved")
        if weapons:
            steps.append("Tag and bag each weapon-class object for fingerprint/DNA processing by the forensic team")
        if stains:
            steps.append("Collect swab samples of the stain candidate for serology — do not rely on colour alone")
    if ocr_items:
        steps.append("Verify visible text/plate content against records; photograph text at higher resolution")
    if vehicles:
        steps.append("Identify vehicle registration and check against stolen-vehicle records")
    steps.append("Record officer name, ID, GPS and timestamp in the case file (chain of custody)")
    if not flags:
        flags.append("no anomaly flagged above thresholds — routine documentation suggested")

    narr = " ".join(narr_parts) + " Overall: preliminary triage only; all findings require investigator confirmation before they enter the case record."
    hi = " ".join(narr_hi) + " समग्र: यह केवल प्रारंभिक वर्गीकरण है; सभी निष्कर्ष केस रिकॉर्ड में दर्ज करने से पहले जाँच अधिकारी की पुष्टि आवश्यक है।"
    return {"narrative": narr, "narrative_hi": hi, "anomaly_flags": flags, "next_steps": steps}


def _call_api(user_prompt: str) -> dict:
    url = config.OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    body = {
        "model": config.OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned non-object JSON")
    return {k: v for k, v in parsed.items() if k in EXPECTED_KEYS}


def reason(analysis: dict) -> dict:
    user_prompt = prompts.build_user_prompt(analysis)
    if config.OPENAI_API_KEY:
        try:
            result = _call_api(user_prompt)
            for key in EXPECTED_KEYS - set(result):
                result[key] = [] if key in ("anomaly_flags", "next_steps") else ""
            return {"source": "openai", "model": config.OPENAI_MODEL, **result}
        except Exception as exc:
            mock = _mock_reason(analysis)
            mock["source"] = "mock"
            mock["model"] = "mock-draft"
            mock["failure"] = f"openai-call-failed: {exc}"
            return mock
    mock = _mock_reason(analysis)
    mock["source"] = "mock"
    mock["model"] = "mock-draft"
    return mock