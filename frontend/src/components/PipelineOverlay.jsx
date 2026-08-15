import { useEffect, useRef, useState } from "react";
import { RagErrorBoundary } from "./RagErrorBoundary.jsx";
import { RagVectorModal } from "./RagVectorModal.jsx";
import { useStageRunner } from "../hooks/useStageRunner.js";
import { ingestStagesFor, IngestSections, IngestStageList } from "../views/IngestView.jsx";
import { RETRIEVAL_STAGES, RetrievalSections, RetrievalStageList } from "../views/RetrievalView.jsx";

// A meta trace ({meta, question, history, answer_text}) has no retrieval stages at all —
// this is literally the whole request: the raw transcript sent to the bulk model, and
// what it answered. See Pipeline.answer_meta / META_CHAT_SYSTEM_PROMPT.
function MetaSection({ data }) {
  return (
    <div className="rag-callout">
      <span className="rag-callout__label">why no retrieval ran</span>
      <p>
        Classified upfront (before any retrieval stage) as a question about this
        conversation or the assistant itself, not the source material — see
        QueryDecomposer/StandaloneRewriter's shared routing grammar. The model answered
        directly from the conversation transcript below.
      </p>
      <span className="rag-callout__label" style={{ marginTop: 14, display: "block" }}>
        transcript sent to the model
      </span>
      <ul className="rag-ranked" style={{ marginTop: 6 }}>
        {(data.history || []).map((turn, i) => (
          <li key={i}>
            <span className="rag-mono rag-ranked__path">
              <strong>{turn.role}:</strong> {turn.content}
            </span>
          </li>
        ))}
        <li>
          <span className="rag-mono rag-ranked__path">
            <strong>user:</strong> {data.question}
          </span>
        </li>
      </ul>
      {data.answer_text && (
        <div className="rag-answer" style={{ marginTop: 14 }}>
          <span className="rag-callout__label">answer</span>
          <p className="rag-answer__text">{data.answer_text}</p>
        </div>
      )}
    </div>
  );
}

// One overlay, three shapes: a query trace (retrieval + generation, one question), an
// ingest trace (one source going in), or a meta trace (a conversation/greeting question
// that bypassed retrieval entirely — see Pipeline.answer_meta). Same stage-rail +
// scrollable-sections shell for the first two — reusing RetrievalView/IngestView's stage
// bodies rather than re-deriving them; meta has no stages at all, so it skips the rail.
export function PipelineOverlay({ mode, data, title, onClose }) {
  const isQuery = mode === "query";
  const isMeta = isQuery && !!data.meta;
  const stages = isMeta ? [] : isQuery ? RETRIEVAL_STAGES : ingestStagesFor(data.kind);
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
          <span className="rag-overlay__eyebrow">
            {isMeta ? "Conversational answer — no retrieval" : isQuery ? "Retrieval pipeline" : "Ingestion pipeline"}
          </span>
          <h2>{title}</h2>
        </div>
        <button className="rag-modal__close" onClick={requestClose} aria-label="Close">
          ×
        </button>
      </header>
      {isQuery && !!data.cache_hit && (
        <div className="rag-callout rag-overlay__cache-note">
          <span className="rag-callout__label">
            {data.cache_kind === "semantic"
              ? "served from cache — semantic match, not a fresh retrieval"
              : "served from cache — exact match, not a fresh retrieval"}
          </span>
          {data.cache_kind === "semantic" && (
            <p className="rag-mono">
              matched &ldquo;{data.cache_match_question}&rdquo; · similarity {data.cache_match_score?.toFixed(3)}
            </p>
          )}
          <p className="rag-hint">
            Everything below is the pipeline that produced the ORIGINAL cached answer, not
            something that ran again for this question.
          </p>
        </div>
      )}
      <div className="rag-overlay__body">
        {!isMeta && (
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
        )}
        <main className="rag-overlay__content" ref={contentRef}>
          <RagErrorBoundary key={mode} label={isMeta ? "Conversational answer" : isQuery ? "Retrieval" : "Ingestion"}>
            {isMeta ? (
              <MetaSection data={data} />
            ) : isQuery ? (
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
