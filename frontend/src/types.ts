export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  id: string;
  class: string;
  category: string;
  confidence: number;
  bbox: BBox;
  source: string;
  basis?: string[];
  area_pct?: number;
  text?: string;
}

export interface TamperReport {
  ela_score: number;
  threshold: number;
  flag: string;
  notes: string[];
}

export interface LlmResult {
  source: string;
  model?: string;
  failure?: string;
  narrative: string;
  narrative_hi?: string;
  anomaly_flags: string[];
  next_steps: string[];
}

export interface Analysis {
  image_id: string;
  image_url: string;
  width: number;
  height: number;
  objects: Detection[];
  stains: Detection[];
  ocr: Detection[];
  tamper: TamperReport;
  metadata: {
    gps?: { lat: number; lng: number } | null;
    captured_at?: string | null;
    has_exif: boolean;
    notes: string[];
  };
  llm: LlmResult;
  processing_notes: string[];
  officer_id?: string;
}

export interface CaseSummary {
  id: string;
  officer_id: string;
  status: string;
  created_at: string;
  captured_at?: string | null;
  gps?: { lat: number; lng: number } | null;
  narrative: string;
  next_steps: string[];
  anomaly_flags: string[];
  llm_source: string;
  object_count: number;
  stain_count: number;
}

export interface CaseMatch {
  case_id: string;
  officer_id: string;
  created_at: string;
  score: number;
  shared_categories: string[];
}

export interface LogEntry {
  ts: string;
  actor: string;
  action: string;
  detail: Record<string, unknown>;
}

export interface CaseDetail extends CaseSummary {
  image_id: string;
  original_narrative: string;
  objects: Detection[];
  stains: Detection[];
  ocr: Detection[];
  tamper: TamperReport;
  metadata: Analysis["metadata"];
  processing_notes: string[];
  log: LogEntry[];
  matches: CaseMatch[];
}

export interface CasePayload {
  officer_id: string;
  image_id: string;
  gps: { lat: number; lng: number } | null;
  captured_at: string | null;
  narrative: string;
  original_narrative: string;
  next_steps: string[];
  anomaly_flags: string[];
  objects: Detection[];
  stains: Detection[];
  ocr: Detection[];
  tamper: TamperReport;
  llm_source: string;
  processing_notes: string[];
}
