import os, cv2, numpy as np
from streamlit.testing.v1 import AppTest

root = os.path.dirname(os.path.abspath(__file__))
app = os.path.join(root, "..", "streamlit_app.py")

def make_img(path, color, w, h):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    cv2.circle(img, (w // 3, h // 3), 80, (0, 0, 255), 3)
    cv2.imwrite(path, img)

p_a = os.path.join(root, "test_a.jpg")  # 1000x620
p_b = os.path.join(root, "test_b.jpg")  # 900x700
make_img(p_a, (70, 70, 110), 1000, 620)
make_img(p_b, (110, 70, 70), 900, 700)

at = AppTest.from_file(app, default_timeout=30)
at.run()
assert not at.exception
with open(p_a, "rb") as f:
    data_a = f.read()
with open(p_b, "rb") as f:
    data_b = f.read()

# upload 2 photos -> analyze
at.get("file_uploader")[0].set_value([("test_a.jpg", data_a, "image/jpeg"),
                                      ("test_b.jpg", data_b, "image/jpeg")]).run()
assert not at.exception
at.get("button")[0].set_value(True).run(timeout=120)
assert not at.exception
assert at.session_state["merged"] is not None
# radios: [0]=photo selector, [1]=canvas mode
assert len(at.radio) == 2

# canvas cache exists for the current photo and changes with markers (fingerprint)
photo = at.radio[0].value
first = at.session_state["_canvas_cache"]["finger"]
assert first

at.session_state["markers"] = [{"id": 1, "photo": photo, "x": 10, "y": 10, "note": "t", "ts": "x"}]
at.run()
assert not at.exception
second = at.session_state["_canvas_cache"]["finger"]
assert second != first, "marker change must invalidate canvas cache"
at.session_state["markers"] = []
at.run()

# markers mode + markers -> component renders with fixed dims
at.radio[1].set_value("Add evidence markers").run()
at.session_state["markers"] = [{"id": 1, "photo": photo, "x": 10, "y": 10, "note": "t", "ts": "x"}]
at.run()
assert not at.exception
cache = at.session_state["_canvas_cache"]
assert cache["rgb"].shape[1] <= 1100 and cache["rgb"].shape[0] > 0
assert cache["bytes"], "JPEG bytes must be non-empty"

# view mode uses cached bytes
at.radio[1].set_value("View overlay").run()
assert not at.exception
assert at.session_state["_canvas_cache"]["bytes"]

# AI suggestions generated (offline rule-based draft, no key needed)
sug = at.session_state["merged"]["suggestions"]
assert sug.get("source") == "mock", sug
assert sug.get("narrative") and sug.get("anomaly_flags") and sug.get("next_steps"), sug
print("ai suggestions:", sug["source"], "| flags:", len(sug["anomaly_flags"]),
      "| steps:", len(sug["next_steps"]))

# confirm flow
for i, b in enumerate(at.get("button")):
    if b.label.startswith("Confirm"):
        at.get("button")[i].set_value(True)
        at.run()
        break
assert not at.exception
detail = at.session_state["last_case"]
assert detail and detail.get("evidence_markers"), detail
ai = detail.get("ai_report") or {}
assert ai.get("source") == "mock" and ai.get("anomaly_flags"), ai
assert detail.get("anomaly_flags"), "DB anomaly_flags must be populated from AI report"
print("thumbs:", cache["rgb"].shape, "| case markers:", len(detail["evidence_markers"]))
print("ALL OK")