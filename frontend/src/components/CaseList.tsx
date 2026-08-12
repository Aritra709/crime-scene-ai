import { useEffect, useState } from "react";
import type { CaseDetail, CaseMatch, CaseSummary, Detection } from "../types";
import { getCase, listCases } from "../api";
import ImageAnnotator from "./ImageAnnotator";

export default function CaseList() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selected, setSelected] = useState<CaseDetail | null>(null);
  const [error, setError] = useState("");

  const load = () => {
    listCases()
      .then(setCases)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };

  useEffect(load, []);

  const open = async (id: string) => {
    setError("");
    try {
      setSelected(await getCase(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="case-list">
      <div className="card">
        <h2>Case log</h2>
        <p className="hint">
          Confirmed case entries with chain-of-custody metadata. Selecting a case shows
          its audit trail and pattern matches against past cases.
        </p>
        <button type="button" className="btn" onClick={load}>
          Refresh
        </button>
        {error && <div className="error">{error}</div>}
        {cases.length === 0 && !error && (
          <p className="hint">No cases logged yet — run a new analysis and confirm it.</p>
        )}
        <table className="cases">
          <thead>
            <tr>
              <th>Case</th>
              <th>Officer</th>
              <th>Objects</th>
              <th>Stains</th>
              <th>LLM</th>
              <th>Logged</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr key={c.id} onClick={() => open(c.id)} className="clickable">
                <td className="mono">{c.id}</td>
                <td>{c.officer_id}</td>
                <td>{c.object_count}</td>
                <td>{c.stain_count}</td>
                <td>
                  <span className={`badge badge-${c.llm_source === "openai" ? "high" : "mid"}`}>
                    {c.llm_source}
                  </span>
                </td>
                <td>{new Date(c.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && <CaseDetailView detail={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

function CaseDetailView({ detail, onClose }: { detail: CaseDetail; onClose: () => void }) {
  const items: Detection[] = [
    ...detail.objects.map((o) => ({ ...o, category: o.category || "object" })),
    ...detail.stains.map((s) => ({ ...s, category: "stain" })),
  ];

  return (
    <div className="card">
      <div className="row-between">
        <h3 className="mono">Case {detail.id}</h3>
        <button type="button" className="btn" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="meta-grid">
        <span>Officer</span><b>{detail.officer_id}</b>
        <span>Logged</span><b>{new Date(detail.created_at).toLocaleString()}</b>
        <span>Captured</span><b>{detail.captured_at ?? "—"}</b>
        <span>GPS</span>
        <b>
          {detail.gps
            ? `${detail.gps.lat.toFixed(5)}, ${detail.gps.lng.toFixed(5)}`
            : "—"}
        </b>
        <span>LLM draft</span>
        <b>
          <span className={`badge badge-${detail.llm_source === "openai" ? "high" : "mid"}`}>
            {detail.llm_source}
          </span>
        </b>
      </div>

      {detail.metadata.has_exif && (
        <p className="hint">EXIF present in original file (GPS + DateTimeOriginal).</p>
      )}
      {detail.metadata.notes?.map((n, i) => (
        <p key={i} className="hint">
          ⚠ {n}
        </p>
      ))}

      {items.length > 0 && (
        <ImageAnnotator
          imageUrl={`/api/images/${detail.image_id}`}
          width={0}
          height={0}
          items={items}
          hiddenIds={new Set()}
          onToggle={() => void 0}
        />
      )}
      {items.length === 0 && (
        <p className="hint">No confirmed visual detections in this case.</p>
      )}

      <h4>Confirmed narrative</h4>
      <p>{detail.narrative}</p>
      <p className="hint">
        AI draft (unchanged copy, for transparency): {detail.original_narrative || "—"}
      </p>

      <h4>{detail.anomaly_flags.length > 0 ? "Anomaly flags" : ""}</h4>
      <ul className="chip-list">
        {detail.anomaly_flags.map((f, i) => (
          <li key={i}>
            <span className="chip flag-chip">{f}</span>
          </li>
        ))}
      </ul>

      <h4>Next steps on record</h4>
      <ul className="basis">
        {detail.next_steps.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>

      <h4>Audit trail</h4>
      <table className="log-table">
        <tbody>
          {detail.log.map((l, i) => (
            <tr key={i}>
              <td className="mono">{new Date(l.ts).toLocaleTimeString()}</td>
              <td>{l.actor}</td>
              <td>{l.action}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Pattern matches vs past cases</h4>
      {detail.matches.length === 0 ? (
        <p className="hint">No overlapping categories with other logged cases.</p>
      ) : (
        <div className="matches">
          {detail.matches.map((m: CaseMatch) => (
            <div key={m.case_id} className="match-card">
              <b className="mono">{m.case_id}</b>
              <span>
                score <b>{m.score}</b>
              </span>
              <span className="match-cats">
                {m.shared_categories.map((c) => (
                  <span key={c} className="chip">
                    {c}
                  </span>
                ))}
              </span>
              <span className="hint">{new Date(m.created_at).toLocaleString()}</span>
            </div>
          ))}
          <p className="hint">
            Triage aid only — shared object categories, not case-linking evidence.
          </p>
        </div>
      )}
    </div>
  );
}