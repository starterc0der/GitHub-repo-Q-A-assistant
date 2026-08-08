import { useEffect, useRef, useState } from "react";
import { RagErrorBoundary } from "./RagErrorBoundary.jsx";
import { RagVectorModal } from "./RagVectorModal.jsx";
import { useStageRunner } from "../hooks/useStageRunner.js";
import { ingestStagesFor, IngestSections, IngestStageList } from "../views/IngestView.jsx";
import { RETRIEVAL_STAGES, RetrievalSections, RetrievalStageList } from "../views/RetrievalView.jsx";

// One overlay, two modes: a query trace (retrieval + generation, one question) or an
// ingest trace (one source going in). Same stage-rail + scrollable-sections shell either
// way — reusing RetrievalView/IngestView's stage bodies rather than re-deriving them.
export function PipelineOverlay({ mode, data, title, onClose }) {
  const isQuery = mode === "query";
  const stages = isQuery ? RETRIEVAL_STAGES : ingestStagesFor(data.kind);
  const ids = stages.map((s) => s.id);
  const runner = useStageRunner(ids);
  const [current, setCurrent] = useState(ids[0]);
  const [showVspace, setShowVspace] = useState(false);
  const contentRef = useRef(null);
  const startedFor = useRef(null);

  useEffect(() => {
    if (startedFor.current === data) return;
    startedFor.current = data;
    setCurrent(ids[0]);
    runner.run((id) => setCurrent(id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // The app has no router, so without this the browser Back button doesn't see the
  // overlay at all — it just does whatever Back normally does one level up, which is
  // leave the app. Fix: give the overlay one history entry, and route every way of
  // closing it (X, Escape, browser Back) through history.back() -> a single popstate
  // listener that does the actual close. That "one path in, one path out" shape is
  // what keeps this correct under StrictMode's dev-only double-invoke of effects —
  // pushing conditionally on history.state (not a local ref) makes the push idempotent
  // across mount->cleanup->mount, and there is no history.back() in any cleanup to race
  // against it.
  useEffect(() => {
    if (!window.history.state?.ragOverlay) {
      window.history.pushState({ ragOverlay: true }, "");
    }
    function onPopState() {
      if (showVspace) setShowVspace(false);
      else onClose();
    }
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [showVspace, onClose]);

  function requestClose() {
    window.history.back();
  }

  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape" && !showVspace) requestClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showVspace]);

  function selectStage(id) {
    setCurrent(id);
    const el = contentRef.current?.querySelector(`[data-stage-id="${id}"]`);
    if (el) contentRef.current.scrollTo({ top: Math.max(0, el.offsetTop - 12), behavior: "smooth" });
  }

  return (
    <div className="rag-overlay" role="dialog" aria-modal="true">
      <header className="rag-overlay__head">
        <div>
          <span className="rag-overlay__eyebrow">{isQuery ? "Retrieval pipeline" : "Ingestion pipeline"}</span>
          <h2>{title}</h2>
        </div>
        <button className="rag-modal__close" onClick={requestClose} aria-label="Close">
          ×
        </button>
      </header>
      <div className="rag-overlay__body">
        <aside className="rag-overlay__rail">
          {isQuery ? (
            <RetrievalStageList data={data} statuses={runner.statuses} current={current} onSelect={selectStage} />
          ) : (
            <IngestStageList data={data} statuses={runner.statuses} current={current} onSelect={selectStage} />
          )}
          {isQuery && (
            <button className="rag-vspace-cta" onClick={() => setShowVspace(true)}>
              <span className="rag-vspace-cta__title">Visualize whole vector space</span>
              <span className="rag-vspace-cta__desc">
                Fullscreen walkthrough of every embedding in this space.
              </span>
            </button>
          )}
        </aside>
        <main className="rag-overlay__content" ref={contentRef}>
          <RagErrorBoundary key={mode} label={isQuery ? "Retrieval" : "Ingestion"}>
            {isQuery ? (
              <RetrievalSections data={data} visible={runner.visible} />
            ) : (
              <IngestSections data={data} visible={runner.visible} />
            )}
          </RagErrorBoundary>
        </main>
      </div>
      {showVspace && isQuery && <RagVectorModal data={data} onClose={() => setShowVspace(false)} />}
    </div>
  );
}
