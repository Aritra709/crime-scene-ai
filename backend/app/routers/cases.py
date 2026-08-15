from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import db

router = APIRouter(prefix="/api", tags=["cases"])


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    id: str = ""
    class_: str | None = Field(default=None, alias="class")
    category: str = ""
    confidence: float = 0.0
    bbox: BBox | None = None
    source: str = ""
    basis: list[str] = []
    text: str | None = None
    area_pct: float | None = None

    model_config = {"populate_by_name": True}


class CasePayload(BaseModel):
    officer_id: str = ""
    image_id: str = ""
    image_path: str = ""
    gps: dict | None = None
    captured_at: str | None = None
    narrative: str = ""
    original_narrative: str = ""
    next_steps: list[str] = []
    anomaly_flags: list[str] = []
    objects: list[Detection] = []
    stains: list[Detection] = []
    ocr: list[Detection] = []
    tamper: dict = {}
    metadata: dict = {}
    llm_source: str = ""
    processing_notes: list[str] = []
    evidence_markers: list[dict] = []
    scale: dict | None = None
    photos: list[dict] = []
    ai_report: dict = {}
    measurements: list[dict] = []


@router.post("/cases")
async def create_case(payload: CasePayload):
    """Officer-confirmed report → case log entry with audit trail."""
    log_entries = [
        {"actor": "system", "action": "case-opened",
         "detail": {"image_id": payload.image_id, "officer_id": payload.officer_id}},
        {"actor": "ai", "action": "analysis-draft-generated",
         "detail": {"source": payload.llm_source,
                    "objects": len(payload.objects), "stains": len(payload.stains)}},
        {"actor": "officer", "action": "confirmed-edited",
         "detail": {"narrative_changed": payload.narrative != payload.original_narrative,
                    "objects_confirmed": len(payload.objects),
                    "next_steps": len(payload.next_steps)}},
    ]
    case_id = db.insert_case(payload.model_dump(by_alias=True), log_entries)
    detail = db.get_case(case_id)
    detail["matches"] = db.find_matches(case_id)
    return detail


@router.get("/cases")
async def list_cases():
    return db.list_cases()


@router.get("/cases/{case_id}")
async def get_case(case_id: str):
    detail = db.get_case(case_id)
    if detail is None:
        raise HTTPException(404, "case not found")
    detail["matches"] = db.find_matches(case_id)
    return detail