import json
import sqlite3
import uuid
from datetime import datetime, timezone

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    officer_id TEXT,
    image_id TEXT,
    image_path TEXT,
    gps_lat REAL,
    gps_lng REAL,
    captured_at TEXT,
    status TEXT DEFAULT 'confirmed',
    created_at TEXT,
    narrative TEXT,
    original_narrative TEXT,
    next_steps TEXT,
    anomaly_flags TEXT,
    objects TEXT,
    stains TEXT,
    ocr TEXT,
    tamper TEXT,
    metadata TEXT,
    llm_source TEXT,
    processing_notes TEXT
);
CREATE TABLE IF NOT EXISTS case_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT,
    ts TEXT,
    actor TEXT,
    action TEXT,
    detail TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_case(payload: dict, log_entries: list) -> str:
    case_id = uuid.uuid4().hex[:12]
    created_at = _now()
    gps = payload.get("gps") or {}
    with _connect() as conn:
        conn.execute(
            """INSERT INTO cases (
                id, officer_id, image_id, image_path, gps_lat, gps_lng, captured_at,
                status, created_at, narrative, original_narrative, next_steps,
                anomaly_flags, objects, stains, ocr, tamper, metadata, llm_source, processing_notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                case_id,
                payload.get("officer_id") or "unknown",
                payload.get("image_id") or "",
                payload.get("image_path") or "",
                gps.get("lat"),
                gps.get("lng"),
                payload.get("captured_at"),
                payload.get("status", "confirmed"),
                created_at,
                json.dumps(payload.get("narrative", ""), ensure_ascii=False),
                json.dumps(payload.get("original_narrative", ""), ensure_ascii=False),
                json.dumps(payload.get("next_steps", []), ensure_ascii=False),
                json.dumps(payload.get("anomaly_flags", []), ensure_ascii=False),
                json.dumps(payload.get("objects", []), ensure_ascii=False),
                json.dumps(payload.get("stains", []), ensure_ascii=False),
                json.dumps(payload.get("ocr", []), ensure_ascii=False),
                json.dumps(payload.get("tamper", {}), ensure_ascii=False),
                json.dumps(payload.get("metadata", {}), ensure_ascii=False),
                payload.get("llm_source", "unknown"),
                json.dumps(payload.get("processing_notes", []), ensure_ascii=False),
            ),
        )
        for entry in log_entries:
            conn.execute(
                "INSERT INTO case_log (case_id, ts, actor, action, detail) VALUES (?,?,?,?,?)",
                (case_id, _now(), entry.get("actor", "system"), entry.get("action", ""),
                 json.dumps(entry.get("detail", {}), ensure_ascii=False)),
            )
    return case_id


def list_cases() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM cases ORDER BY created_at DESC"
        ).fetchall()
    return [_row_summary(r) for r in rows]


def get_case(case_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            return None
        log = conn.execute(
            "SELECT ts, actor, action, detail FROM case_log WHERE case_id = ? ORDER BY id",
            (case_id,),
        ).fetchall()
    detail = _row_detail(row)
    detail["log"] = [_decode(e) for e in log]
    return detail


def find_matches(case_id: str, limit: int = 5) -> list[dict]:
    """Category-overlap scoring vs all OTHER confirmed cases (triage aid only)."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM cases WHERE id != ? AND status = 'confirmed'", (case_id,)).fetchall()
    target = get_case(case_id) or {}
    target_cats = _categories(target.get("objects", [])) | _categories(target.get("stains", []))
    if not target_cats:
        return []
    results = []
    for r in rows:
        other = _row_detail(r)
        other_cats = _categories(other.get("objects", [])) | _categories(other.get("stains", []))
        shared = sorted(target_cats & other_cats)
        if not shared:
            continue
        score = sum(
            max(
                [d.get("confidence", 0.5) for d in (other.get("objects", []) + other.get("stains", []))
                 if d.get("category") == c] or [0.5]
            ) * 1.0
            for c in shared
        ) + 1.0 * len(shared)
        results.append({
            "case_id": other["id"],
            "officer_id": other["officer_id"],
            "created_at": other["created_at"],
            "score": round(score, 2),
            "shared_categories": shared,
        })
    results.sort(key=lambda m: -m["score"])
    return results[:limit]


def _categories(items: list) -> set:
    return {str(i.get("category", i.get("class", ""))).lower() for i in items if i}


def _row_summary(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "officer_id": r["officer_id"],
        "status": r["status"],
        "created_at": r["created_at"],
        "captured_at": r["captured_at"],
        "gps": {"lat": r["gps_lat"], "lng": r["gps_lng"]} if r["gps_lat"] is not None else None,
        "narrative": json.loads(r["narrative"] or '""'),
        "next_steps": json.loads(r["next_steps"] or "[]"),
        "anomaly_flags": json.loads(r["anomaly_flags"] or "[]"),
        "llm_source": r["llm_source"],
        "object_count": len(json.loads(r["objects"] or "[]")),
        "stain_count": len(json.loads(r["stains"] or "[]")),
    }


def _row_detail(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "officer_id": r["officer_id"],
        "image_id": r["image_id"],
        "status": r["status"],
        "created_at": r["created_at"],
        "captured_at": r["captured_at"],
        "gps": {"lat": r["gps_lat"], "lng": r["gps_lng"]} if r["gps_lat"] is not None else None,
        "narrative": json.loads(r["narrative"] or '""'),
        "original_narrative": json.loads(r["original_narrative"] or '""'),
        "next_steps": json.loads(r["next_steps"] or "[]"),
        "anomaly_flags": json.loads(r["anomaly_flags"] or "[]"),
        "objects": json.loads(r["objects"] or "[]"),
        "stains": json.loads(r["stains"] or "[]"),
        "ocr": json.loads(r["ocr"] or "[]"),
        "tamper": json.loads(r["tamper"] or "{}"),
        "metadata": json.loads(r["metadata"] or "{}"),
        "llm_source": r["llm_source"],
        "processing_notes": json.loads(r["processing_notes"] or "[]"),
    }


def _decode(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["detail"] = json.loads(d.get("detail") or "{}")
    return d
