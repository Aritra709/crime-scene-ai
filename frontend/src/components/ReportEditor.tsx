import { useMemo, useState } from "react";
import type { Analysis, CasePayload, Detection } from "../types";
import { createCase } from "../api";
import Badge from "./Badge";
import ImageAnnotator from "./ImageAnnotator";

interface Props {
  analysis: Analysis;
  onSubmitted: (payload: CasePayload) => void;
  onDiscard: () => void;
}

function mergedItems(analysis: Analysis): Detection[] {
  return [
    ...analysis.objects.map((o) => ({ ...o, category: o.category || "object" })),
    ...analysis.stains.map((s) => ({ ...s, category: "stain" })),
    ...analysis.ocr.map((o) => ({ ...o, class: `text: ${o.text}`, category: "ocr" })),
  ];
}

export default function ReportEditor({ analysis, onSubmitted, onDiscard }: Props) {
  const items = useMemo(() => mergedItems(analysis), [analysis]);

  const [included, setIncluded] = useState<Set<string>>(
    () => new Set(items.map((i) => i.id)),
  );
  const [labels, setLabels] = useState<Record<string, string>>(
    () => Object.fromEntries(items.map((i) => [i.id, i.class])),
  );
  const [narrative, setNarrative] = useState(analysis.llm.narrative);
  const [flags, setFlags] = useState<string[]>(analysis.llm.anomaly_flags);
  const [steps, setSteps] = useState<string[]>(analysis.llm.next_steps);
  const [newStep, setNewStep] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  const toggle = (id: string) =>
    setIncluded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const removeFlag = (idx: number) => setFlags((f) => f.filter((_, i) => i !== idx));
  const removeStep = (idx: number) => setSteps((s) => s.filter((_, i) => i !== idx));
  const addStep = () => {
    const t = newStep.trim();
    if (t) setSteps((s) => [...s, t]);
    setNewStep("");
  };

  const submit = async () => {
    if (!narrative.trim()) return;
    setBusy(true);
    const all = items.filter((i) => included.has(i.id));
    const payload: CasePayload = {
      officer_id: analysis.officer_id ?? "",
      image_id: analysis.image_id,
      gps: analysis.metadata.gps ?? null,
      captured_at: analysis.metadata.captured_at ?? null,
      narrative,
      original_narrative: analysis.llm.narrative,
      next_steps: steps,
      anomaly_flags: flags,
      objects: all.filter((i) => i.category !== "stain" && i.category !== "ocr").map((i) => ({ ...i, class: labels[i.id] ?? i.class })),
      stains: all.filter((i) => i.category === "stain"),
      ocr: all.filter((i) => i.category === "ocr").map((i) => ({ text: i.text ?? i.class.replace(/^text: /, ""), ...i })),
      tamper: analysis.tamper,
      llm_source: analysis.llm.source,
      processing_notes: analysis.processing_notes,
    };
    try {
      await createCase(payload);
      setDone(true);
      onSubmitted(payload);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (done) {
    return (
      <div className="card">
        <h2>Case logged ✔</h2>
        <p>
          The officer-confirmed report was written to the case file with GPS, timestamps
          and an audit trail. Your changes were recorded — the AI draft is preserved as
          <code> original_narrative </code> for transparency.
        </p>
        <button className="btn primary" onClick={onDiscard}>
          Review case log
        </button>
      </div>
    );
  }

  return (
    <div className="report">
      <div className="card">
        <h2>Officer review — nothing is logged until you confirm</h2>
        <p className="hint">
          AI draft by <strong>{analysis.llm.source}</strong>
          {analysis.llm.model ? ` (${analysis.llm.model})` : ""}. Every box below is a
          suggestion carrying a confidence score. Confirm, re-label or remove items; the
          final report is yours.
        </p>
        <ImageAnnotator
          imageUrl={analysis.image_url}
          width={analysis.width}
          height={analysis.height}
          items={items}
          hiddenIds={new Set()}
          onToggle={() => void 0}
        />
      </div>

      <div className="card">
        <h3>Detections ({items.filter((i) => included.has(i.id)).length}/{items.length})</h3>
        <ul className="det-list">
          {items.map((item) => {
            const on = included.has(item.id);
            return (
              <li key={item.id} className={on ? "" : "muted"}>
                <label className="det-row">
                  <input type="checkbox" checked={on} onChange={() => toggle(item.id)} />
                  <input
                    className="det-label"
                    value={labels[item.id] ?? ""}
                    onChange={(e) =>
                      setLabels((l) => ({ ...l, [item.id]: e.target.value }))
                    }
                    disabled={!on}
                  />
                  <span className="det-cat">{item.category}</span>
                  <Badge confidence={item.confidence} source={item.source} />
                  <span className="det-source">{item.source}</span>
                </label>
                {item.basis && (
                  <ul className="basis">
                    {item.basis.slice(0, 3).map((b, i) => (
                      <li key={i}>{b}</li>
                    ))}
                  </ul>
                )}
              </li>
            );
          })}
        </ul>
      </div>

      <div className="card">
        <h3>AI anomaly flags {flags.length > 0 && `(${flags.length})`}</h3>
        {flags.length === 0 ? (
          <p className="hint">No flags.</p>
        ) : (
          <ul className="chip-list">
            {flags.map((f, i) => (
              <li key={i}>
                <span className="chip flag-chip">{f}</span>
                <button type="button" className="chip-x" onClick={() => removeFlag(i)}>
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
        <h3>Tamper / integrity</h3>
        <p>
          <span className={`tamper ${analysis.tamper.flag}`}>{analysis.tamper.flag}</span>
          <em className="hint"> ELA score {analysis.tamper.ela_score} (threshold {analysis.tamper.threshold}) — heuristic only</em>
        </p>
        <ul className="basis">
          {analysis.tamper.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h3>Observation report <span className="hint">(edit freely — final text is yours)</span></h3>
        <textarea
          className="narrative"
          rows={8}
          value={narrative}
          onChange={(e) => setNarrative(e.target.value)}
        />
        {analysis.llm.narrative_hi && (
          <details>
            <summary>हिन्दी ड्राफ्ट (Hindi draft, for reference)</summary>
            <p className="hindi">{analysis.llm.narrative_hi}</p>
          </details>
        )}

        <h3>Suggested next steps</h3>
        <ul className="basis">
          {steps.map((s, i) => (
            <li key={i}>
              {s}
              <button type="button" className="chip-x" onClick={() => removeStep(i)}>
                ✕
              </button>
            </li>
          ))}
        </ul>
        <div className="add-row">
          <input
            value={newStep}
            onChange={(e) => setNewStep(e.target.value)}
            placeholder="Add a step…"
            onKeyDown={(e) => e.key === "Enter" && addStep()}
          />
          <button type="button" className="btn" onClick={addStep}>
            Add
          </button>
        </div>

        <div className="actions">
          <button type="button" className="btn" onClick={onDiscard}>
            Discard draft
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={busy || !narrative.trim()}
            onClick={submit}
          >
            {busy ? "Logging…" : "Confirm & log to case file"}
          </button>
        </div>
      </div>
    </div>
  );
}