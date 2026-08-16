import { useState } from "react";
import type { Analysis, VideoAnalysis } from "./types";
import UploadView from "./components/UploadView";
import ReportEditor from "./components/ReportEditor";
import CaseList from "./components/CaseList";

type View = "upload" | "report" | "cases";

export default function App() {
  const [view, setView] = useState<View>("upload");
  const [analysis, setAnalysis] = useState<Analysis | VideoAnalysis | null>(null);

  return (
    <div className="app">
      <header>
        <h1>Crime Scene AI</h1>
        <span className="tagline">Smart Evidence Capture Assistant — SIH demo</span>
        <nav>
          <button
            type="button"
            className={view === "upload" ? "tab active" : "tab"}
            onClick={() => setView("upload")}
          >
            New analysis
          </button>
          <button
            type="button"
            className={view === "cases" ? "tab active" : "tab"}
            onClick={() => setView("cases")}
          >
            Case log
          </button>
          {analysis && view === "report" && (
            <button type="button" className="tab active" disabled>
              Review draft
            </button>
          )}
        </nav>
      </header>

      <main>
        {view === "upload" && (
          <UploadView
            onAnalyzed={(a) => {
              setAnalysis(a);
              setView("report");
            }}
          />
        )}
        {view === "report" && analysis && (
          <ReportEditor
            analysis={analysis}
            onSubmitted={() => setView("cases")}
            onDiscard={() => setView("upload")}
          />
        )}
        {view === "cases" && <CaseList />}
      </main>

      <footer className="hint">
        AI suggestions are drafts for officer confirmation — never admissible evidence on
        their own. Runs offline; YOLO·OCR·LLM are plug-in upgrades.
      </footer>
    </div>
  );
}