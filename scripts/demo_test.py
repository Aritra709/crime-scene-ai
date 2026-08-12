"""End-to-end API test: upload → analysis → officer-confirm → case log → pattern match.

Usage:  python scripts/demo_test.py [base] [image_path]
Defaults: localhost:8000, scripts/sample_scene.jpg.
Pass scripts/sample_real_photo.jpg as the image to exercise YOLO object detection
(your synthetic drawing is too abstract for COCO models — real photos aren't).
Requires the backend running (uvicorn app.main:app --port 8000).
"""

import json
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
IMG = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "sample_scene.jpg"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def req(method: str, url: str, body=None, headers=None):
    data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body else None)
    r = urllib.request.Request(
        url, data=data, method=method,
        headers=headers or {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main():
    if not IMG.exists():
        print(f"sample image missing — run: python scripts/make_sample_image.py")
        sys.exit(1)

    boundary = "demo-boundary-1234"
    with open(IMG, "rb") as f:
        blob = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"scene.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n".encode() + blob +
        b"\r\n--" + boundary.encode() + b"\r\n" +
        b"Content-Disposition: form-data; name=\"officer_id\"\r\n\r\nDEMO-OFFICER-01\r\n" +
        f"--{boundary}--\r\n".encode()
    )

    print("== 1. upload + analysis ==")
    a = req("POST", f"{BASE}/api/upload", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    print(f"image_id={a['image_id']} size={a['width']}x{a['height']}")
    print(f"objects={len(a['objects'])} stains={len(a['stains'])} ocr={len(a['ocr'])}")
    for s in a["stains"]:
        print(f"  stain: {s['class']} conf={s['confidence']} area={s.get('area_pct')}% source={s['source']}")
    for o in a["objects"]:
        print(f"  object: {o['class']} conf={o['confidence']} source={o['source']}")
    print(f"tamper: {a['tamper']['flag']} (ela {a['tamper']['ela_score']})")
    print(f"metadata: gps={a['metadata'].get('gps')} captured_at={a['metadata'].get('captured_at')}")
    print(f"llm source={a['llm']['source']}")
    print(f"narrative: {a['llm']['narrative'][:180]}...")
    print(f"narrative_hi: {a['llm'].get('narrative_hi', '')[:120]}...")
    print(f"notes: {a['processing_notes']}")
    assert a["stains"], "expected at least one stain candidate on the synthetic scene"

    print("\n== 2. officer confirms (edited report) ==")
    payload = {
        "officer_id": "DEMO-OFFICER-01",
        "image_id": a["image_id"],
        "gps": a["metadata"]["gps"],
        "captured_at": a["metadata"]["captured_at"],
        "narrative": "Officer-reviewed: stain candidate present; knife-like object noted; awaiting forensic team.",
        "original_narrative": a["llm"]["narrative"],
        "next_steps": a["llm"]["next_steps"][:3],
        "anomaly_flags": a["llm"]["anomaly_flags"][:2],
        "objects": a["objects"],
        "stains": a["stains"],
        "ocr": a["ocr"],
        "tamper": a["tamper"],
        "llm_source": a["llm"]["source"],
        "processing_notes": a["processing_notes"],
    }
    case = req("POST", f"{BASE}/api/cases", payload)
    print(f"case id={case['id']} status={case['status']} llm_source={case['llm_source']}")
    print(f"audit trail: {[l['actor'] + ':' + l['action'] for l in case['log']]}")

    print("\n== 3. case list + pattern matches ==")
    cases = req("GET", f"{BASE}/api/cases")
    print(f"{len(cases)} case(s) logged")
    for c in cases:
        print(f"  {c['id']} officer={c['officer_id']} objects={c['object_count']} stains={c['stain_count']} llm={c['llm_source']}")
    if cases:
        detail = req("GET", f"{BASE}/api/cases/{cases[0]['id']}")
        print(f"matches for {cases[0]['id']}: {detail['matches']}")
        print("annotated image:", f"{BASE}/api/images/{detail['image_id']}")

    print("\nOK — full loop passed.")


if __name__ == "__main__":
    main()