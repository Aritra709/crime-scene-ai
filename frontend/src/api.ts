import type { Analysis, CaseDetail, CasePayload, CaseSummary } from "./types";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${body.slice(0, 300)}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadImage(
  file: File,
  officerId: string,
  gps: { lat: number; lng: number } | null,
): Promise<Analysis> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("officer_id", officerId);
  if (gps) {
    fd.append("lat", String(gps.lat));
    fd.append("lng", String(gps.lng));
  }
  return j<Analysis>(await fetch("/api/upload", { method: "POST", body: fd }));
}

export async function createCase(payload: CasePayload): Promise<CaseDetail> {
  return j<CaseDetail>(
    await fetch("/api/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function listCases(): Promise<CaseSummary[]> {
  return j<CaseSummary[]>(await fetch("/api/cases"));
}

export async function getCase(id: string): Promise<CaseDetail> {
  return j<CaseDetail>(await fetch(`/api/cases/${id}`));
}
