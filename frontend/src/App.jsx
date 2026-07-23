import { useEffect, useRef, useState } from "react";
import { ingestTraceStream, listRepos, queryTrace } from "./api.js";
import { cls } from "./components/RagAtoms.jsx";
import { RagVectorModal } from "./components/RagVectorModal.jsx";
import { useStageRunner } from "./hooks/useStageRunner.js";
import { INGEST_STAGES, IngestSections, IngestStageList } from "./views/IngestView.jsx";
import { RETRIEVAL_STAGES, RetrievalSections, RetrievalStageList } from "./views/RetrievalView.jsx";
import "./App.css";

const INGEST_IDS = INGEST_STAGES.map((s) => s.id);
const RETRIEVAL_IDS = RETRIEVAL_STAGES.map((s) => s.id);

// files_total is 0 during clone/walk/summarize_repo (duration unknown upfront) — shown as
// an indeterminate sliding bar; process_files/embed have a real done/total count.
function IngestProgressBar({ progress }) {
  const determinate = progress.files_total > 0;
  const pct = determinate ? Math.round((progress.files_done / progress.files_total) * 100) : 0;
  return (
    <div className="rag-ingest-progress">
      <div className="rag-ingest-progress__bar">
        <div
          className={cls(
            "rag-ingest-progress__fill",
            !determinate && "rag-ingest-progress__fill--indeterminate"
          )}
          style={determinate ? { width: `${pct}%` } : undefined}
        />
      </div>
      <p className="rag-ingest-progress__label">
        {progress.message}
        {determinate && ` (${progress.files_done}/${progress.files_total})`}
      </p>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("ingest");
  const [showInfo, setShowInfo] = useState(false);
  const [showVspace, setShowVspace] = useState(false);
  const contentRef = useRef(null);

  const [repoUrl, setRepoUrl] = useState("");
  const [ingestData, setIngestData] = useState(null);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(null);
  const [ingestError, setIngestError] = useState(null);
  const [currentIngest, setCurrentIngest] = useState(INGEST_IDS[0]);
  const ingestRunner = useStageRunner(INGEST_IDS);
  const closeIngestStreamRef = useRef(null);

  const [repo, setRepo] = useState("");
  const [repoOptions, setRepoOptions] = useState([]);
  const [question, setQuestion] = useState("");
  const [retrievalData, setRetrievalData] = useState(null);
  const [retrievalLoading, setRetrievalLoading] = useState(false);
  const [retrievalError, setRetrievalError] = useState(null);
  const [currentRetrieval, setCurrentRetrieval] = useState(RETRIEVAL_IDS[0]);
  const retrievalRunner = useStageRunner(RETRIEVAL_IDS);

  function refreshRepoOptions() {
    listRepos()
      .then((data) => setRepoOptions(data.repos))
      .catch(() => {});
  }

  useEffect(() => {
    refreshRepoOptions();
  }, []);

  function scrollToStage(id) {
    const container = contentRef.current;
    if (!container) return;
    const el = container.querySelector(`[data-stage-id="${id}"]`);
    if (el) container.scrollTo({ top: Math.max(0, el.offsetTop - 12), behavior: "smooth" });
  }

  function selectStage(kind, id) {
    if (kind === "ingest") setCurrentIngest(id);
    else setCurrentRetrieval(id);
    scrollToStage(id);
  }

  function handleRunIngest(e) {
    e.preventDefault();
    if (ingestRunner.running || !repoUrl.trim()) return;
    closeIngestStreamRef.current?.();
    setIngestError(null);
    setIngestProgress(null);
    setIngestLoading(true);
    closeIngestStreamRef.current = ingestTraceStream(repoUrl.trim(), {
      onProgress: (progress) => setIngestProgress(progress),
      onComplete: (data) => {
        setIngestLoading(false);
        setIngestProgress(null);
        setIngestData(data);
        setRepo(data.repo);
        refreshRepoOptions();
        setCurrentIngest(INGEST_IDS[0]);
        contentRef.current?.scrollTo({ top: 0, behavior: "smooth" });
        ingestRunner.run((id) => setCurrentIngest(id));
      },
      onError: (err) => {
        setIngestLoading(false);
        setIngestProgress(null);
        setIngestError(err.message);
      },
    });
  }

  async function handleRunRetrieval(e) {
    e.preventDefault();
    if (retrievalRunner.running || !repo.trim() || !question.trim()) return;
    setRetrievalError(null);
    setRetrievalLoading(true);
    try {
      const data = await queryTrace(question.trim(), repo.trim());
      setRetrievalData(data);
      setCurrentRetrieval(RETRIEVAL_IDS[0]);
      contentRef.current?.scrollTo({ top: 0, behavior: "smooth" });
      retrievalRunner.run((id) => setCurrentRetrieval(id));
    } catch (err) {
      setRetrievalError(err.message);
    } finally {
      setRetrievalLoading(false);
    }
  }

  const isIngest = tab === "ingest";
  const ingestBusy = ingestLoading || ingestRunner.running;
  const retrievalBusy = retrievalLoading || retrievalRunner.running;

  return (
    <div className="rag-app">
      <header className="rag-header">
        <div className="rag-brand">
          <span className="rag-brand__mark" />
          <div>
            <h1>RAG Pipeline Visualizer</h1>
            <p>Repo Q&amp;A assistant — ingestion and retrieval, stage by stage</p>
          </div>
        </div>
        <div className="rag-tabs">
          <button className={cls("rag-tabs__btn", isIngest && "rag-tabs__btn--active")} onClick={() => setTab("ingest")}>
            Ingestion
          </button>
          <button className={cls("rag-tabs__btn", !isIngest && "rag-tabs__btn--active")} onClick={() => setTab("retrieval")}>
            Retrieval
          </button>
        </div>
        <button className="rag-info-btn" onClick={() => setShowInfo(!showInfo)} title="What am I looking at?">
          ?
        </button>
      </header>

      {showInfo && (
        <div className="rag-info-panel">
          <h3>New to RAG? Start here.</h3>
          <p>
            <strong>Embedding</strong> = turning text into a list of numbers that captures its meaning. Similar text
            gets similar numbers, so "closeness" between two pieces of text becomes something you can measure and
            plot — that's what the dot plots below show.
          </p>
          <p>
            <strong>Ingestion</strong> (left tab) happens once per repo: it reads the code, cuts it into pieces, and
            embeds everything into a searchable index.
          </p>
          <p>
            <strong>Retrieval</strong> (right tab) happens per question: it embeds the question, finds the closest
            pieces from the index, double-checks them, trims them, and assembles the final prompt that would be sent
            to an LLM — this tool stops there and shows you exactly what would be sent, without calling the LLM.
          </p>
          <p>Click any step on the left to jump to it, or hit Run to watch the whole flow play out.</p>
        </div>
      )}

      <div className="rag-body">
        <aside className="rag-sidebar">
          {isIngest ? (
            <form className="rag-input-card" onSubmit={handleRunIngest}>
              <label className="rag-input-card__label">Repository URL</label>
              <input
                className="rag-input"
                placeholder="https://github.com/owner/repo"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                disabled={ingestBusy}
              />
              <button className="rag-btn" type="submit" disabled={ingestBusy}>
                {ingestBusy ? "Ingesting…" : "Run ingest"}
              </button>
              {ingestProgress && <IngestProgressBar progress={ingestProgress} />}
              {ingestError && <p className="rag-error">{ingestError}</p>}
            </form>
          ) : (
            <form className="rag-input-card" onSubmit={handleRunRetrieval}>
              <label className="rag-input-card__label">Repo</label>
              <select
                className="rag-input rag-input--mono"
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                disabled={retrievalBusy || repoOptions.length === 0}
              >
                <option value="" disabled>
                  {repoOptions.length === 0 ? "No ingested repos yet" : "Select a repo…"}
                </option>
                {repoOptions.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
              <label className="rag-input-card__label">Question</label>
              <input
                className="rag-input"
                placeholder="Ask a question about the repo…"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={retrievalBusy}
              />
              <button className="rag-btn" type="submit" disabled={retrievalBusy}>
                {retrievalBusy ? "Retrieving…" : "Run retrieval"}
              </button>
              {retrievalError && <p className="rag-error">{retrievalError}</p>}
            </form>
          )}

          <div className="rag-sidebar__stages">
            {isIngest
              ? ingestData && (
                  <IngestStageList
                    data={ingestData}
                    statuses={ingestRunner.statuses}
                    current={currentIngest}
                    onSelect={(id) => selectStage("ingest", id)}
                  />
                )
              : retrievalData && (
                  <RetrievalStageList
                    data={retrievalData}
                    statuses={retrievalRunner.statuses}
                    current={currentRetrieval}
                    onSelect={(id) => selectStage("retrieval", id)}
                  />
                )}
          </div>

          {!isIngest && retrievalData && (
            <button className="rag-vspace-cta" onClick={() => setShowVspace(true)}>
              <span className="rag-vspace-cta__title">Visualize whole vector space</span>
              <span className="rag-vspace-cta__desc">
                Open a fullscreen walkthrough of every embedding, and watch the question find
                its neighbors at each step.
              </span>
            </button>
          )}
        </aside>

        <main className="rag-content" ref={contentRef}>
          {isIngest
            ? ingestData && <IngestSections data={ingestData} visible={ingestRunner.visible} />
            : retrievalData && <RetrievalSections data={retrievalData} visible={retrievalRunner.visible} />}
        </main>
      </div>

      {showVspace && retrievalData && (
        <RagVectorModal data={retrievalData} onClose={() => setShowVspace(false)} />
      )}
    </div>
  );
}
