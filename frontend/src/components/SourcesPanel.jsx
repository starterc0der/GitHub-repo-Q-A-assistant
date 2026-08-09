import { useEffect, useRef, useState } from "react";
import { createSource, deleteSource, sourceIngestStream, sourceTrace, uploadSource } from "../api.js";
import { cls, timeAgo } from "./RagAtoms.jsx";
import { PipelineOverlay } from "./PipelineOverlay.jsx";

const KIND_LABEL = { repo: "REPO", pdf: "PDF", docx: "DOC", text: "TXT", csv: "CSV" };
const KIND_TAG_CLASS = { repo: "rag-tag--accent", pdf: "rag-tag--warn", docx: "rag-tag--doc" };
const STATUS_LABEL = { ready: "Ready", ingesting: "Ingesting", failed: "Failed", pending: "Pending" };
const FILTERS = [
  { key: "all", label: "All sources" },
  { key: "repo", label: "Repos" },
  { key: "pdf", label: "PDFs" },
  { key: "docx", label: "Docs" },
  { key: "csv", label: "CSV" },
  { key: "text", label: "Text" },
];

function SourceBadge({ kind }) {
  return (
    <span className={`rag-tag ${KIND_TAG_CLASS[kind] || "rag-tag--neutral"}`}>
      {KIND_LABEL[kind] || kind.toUpperCase()}
    </span>
  );
}

function StatusPill({ status }) {
  return <span className={`rag-status-pill rag-status-pill--${status}`}>{STATUS_LABEL[status] || status}</span>;
}

const RING_R = 9;
const RING_C = 2 * Math.PI * RING_R;

// files_total is 0 during stages with no known count yet (clone/walk/summarize) — shown
// as a spinning ring; process_files/embed have a real done/total, shown as a filled arc.
function IngestRing({ progress }) {
  const determinate = progress.files_total > 0;
  const pct = determinate ? Math.round((progress.files_done / progress.files_total) * 100) : 0;
  const offset = determinate ? RING_C - (pct / 100) * RING_C : RING_C * 0.75;

  return (
    <span className="rag-ring" title={progress.message}>
      <svg viewBox="0 0 24 24" className={cls("rag-ring__svg", !determinate && "rag-ring__svg--spin")}>
        <g transform="rotate(-90 12 12)">
          <circle cx="12" cy="12" r={RING_R} className="rag-ring__track" />
          <circle
            cx="12" cy="12" r={RING_R} className="rag-ring__fill"
            strokeDasharray={RING_C} strokeDashoffset={offset}
          />
        </g>
      </svg>
      {determinate && <span className="rag-ring__pct">{pct}%</span>}
    </span>
  );
}

// Subscribes to one source's ingest SSE while it's ingesting; reports live progress and
// calls onSettled(sourceId) once it reaches ready/failed so the parent can refetch.
function useIngestProgress(source, onSettled) {
  const [progress, setProgress] = useState(null);
  const closeRef = useRef(null);

  useEffect(() => {
    if (source.status !== "ingesting") {
      setProgress(null);
      return;
    }
    closeRef.current = sourceIngestStream(source.id, {
      onProgress: setProgress,
      onComplete: () => onSettled(source.id),
      onError: () => onSettled(source.id),
    });
    return () => closeRef.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source.id, source.status]);

  return progress;
}

function SourceRow({ source, onRefresh, onOpenDetail }) {
  const progress = useIngestProgress(source, onRefresh);
  return (
    <div className="rag-source-row rag-source-row--clickable" onClick={() => onOpenDetail(source.id)}>
      <div className="rag-source-row__main">
        <SourceBadge kind={source.kind} />
        <span className="rag-source-row__name">{source.name}</span>
        <span className="rag-source-row__added">{timeAgo(source.created_at)}</span>
        {source.status === "ingesting" && (
          <>
            <IngestRing progress={progress || { message: "Ingesting…", files_done: 0, files_total: 0 }} />
            <span className="rag-dim rag-source-row__stat">{progress?.message || "Ingesting…"}</span>
          </>
        )}
      </div>
      <StatusPill status={source.status} />
    </div>
  );
}

function RepoModal({ spaceId, onClose, onAdded }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function submit(e) {
    e.preventDefault();
    if (!url.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await createSource(spaceId, { kind: "repo", uri: url.trim() });
      onAdded();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <form className="rag-modal-card" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <h2>Connect GitHub repo</h2>
        <label className="rag-input-card__label">Repository URL</label>
        <input
          className="rag-input"
          placeholder="https://github.com/owner/repo"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={busy}
          autoFocus
        />
        {error && <p className="rag-error">{error}</p>}
        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="rag-btn" disabled={busy || !url.trim()}>
            {busy ? "Adding…" : "Add"}
          </button>
        </div>
      </form>
    </div>
  );
}

function TextModal({ spaceId, onClose, onAdded }) {
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function submit(e) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await createSource(spaceId, { kind: "text", name: name.trim() || undefined, text });
      onAdded();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <form className="rag-modal-card" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <h2>Paste text</h2>
        <label className="rag-input-card__label">Name (optional)</label>
        <input className="rag-input" value={name} onChange={(e) => setName(e.target.value)} disabled={busy} autoFocus />
        <label className="rag-input-card__label">Text</label>
        <textarea
          className="rag-input rag-textarea"
          placeholder="Paste text to index…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
          rows={8}
        />
        {error && <p className="rag-error">{error}</p>}
        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="rag-btn" disabled={busy || !text.trim()}>
            {busy ? "Adding…" : "Add"}
          </button>
        </div>
      </form>
    </div>
  );
}

function SourceDetailModal({ source, onClose, onDelete, onViewBreakdown }) {
  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <div className="rag-modal-card" onClick={(e) => e.stopPropagation()}>
        <h2>{source.name}</h2>
        <div className="rag-source-detail__row">
          <span className="rag-source-detail__k">Type</span>
          <SourceBadge kind={source.kind} />
        </div>
        <div className="rag-source-detail__row">
          <span className="rag-source-detail__k">Added</span>
          <span className="rag-dim">{new Date(source.created_at).toLocaleString()}</span>
        </div>
        <div className="rag-source-detail__row">
          <span className="rag-source-detail__k">Status</span>
          <StatusPill status={source.status} />
        </div>
        {source.status === "ready" && (
          <div className="rag-source-detail__row">
            <span className="rag-source-detail__k">Indexed</span>
            <span className="rag-dim">
              {source.file_count} files · {source.chunk_count} chunks
            </span>
          </div>
        )}
        {source.status === "failed" && (
          <div className="rag-source-detail__row">
            <span className="rag-source-detail__k">Error</span>
            <span className="rag-error">{source.error}</span>
          </div>
        )}
        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" style={{ color: "var(--warn-ink)" }} onClick={() => onDelete(source.id)}>
            Remove source
          </button>
          {source.has_trace && (
            <button type="button" className="rag-btn rag-btn--ghost" onClick={onViewBreakdown}>
              View ingestion breakdown
            </button>
          )}
          <button type="button" className="rag-btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export function useSourcesController(spaceId, sources, onRefresh) {
  const [filter, setFilter] = useState("all");
  const [modal, setModal] = useState(null); // null | "repo" | "text"
  const [detailId, setDetailId] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [loadingBreakdown, setLoadingBreakdown] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileRef = useRef(null);

  const detailSource = sources.find((s) => s.id === detailId) || null;
  const counts = sources.reduce((acc, s) => {
    acc[s.kind] = (acc[s.kind] || 0) + 1;
    return acc;
  }, {});
  const filtered = filter === "all" ? sources : sources.filter((s) => s.kind === filter);

  async function handleFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    const kind = { pdf: "pdf", docx: "docx", csv: "csv" }[ext];
    if (!kind) {
      setUploadError("Only .pdf, .docx, and .csv files are supported.");
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      await uploadSource(spaceId, kind, file);
      onRefresh();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(sourceId) {
    if (!confirm("Delete this source? This removes its indexed data too.")) return;
    await deleteSource(sourceId);
    setDetailId(null);
    onRefresh();
  }

  async function openBreakdown() {
    const source = detailSource;
    setDetailId(null);
    setLoadingBreakdown(true);
    try {
      setBreakdown({ trace: await sourceTrace(source.id), name: source.name });
    } catch (err) {
      alert(err.message);
    } finally {
      setLoadingBreakdown(false);
    }
  }

  return {
    filter, setFilter, modal, setModal, detailSource, setDetailId, breakdown, setBreakdown, loadingBreakdown,
    uploading, uploadError, fileRef, handleFile, handleDelete, openBreakdown, counts, filtered, sources, spaceId, onRefresh,
  };
}

export function SourcesSidebar({ filter, setFilter, counts, sources }) {
  return (
    <>
      <div className="rag-sources__sidebar-label">Filter</div>
      {FILTERS.map((f) => (
        <button
          key={f.key}
          className={cls("rag-sources__filter", filter === f.key && "rag-sources__filter--active")}
          onClick={() => setFilter(f.key)}
        >
          <span>{f.label}</span>
          <span className="rag-sources__filter-count">{f.key === "all" ? sources.length : counts[f.key] || 0}</span>
        </button>
      ))}
    </>
  );
}

export function SourcesMain({
  spaceId, sources, filtered, modal, setModal, detailSource, setDetailId, breakdown, setBreakdown,
  loadingBreakdown, uploading, uploadError, fileRef, handleFile, handleDelete, openBreakdown, onRefresh,
}) {
  return (
    <div className="rag-sources__main">
      <div className="rag-sources__container">
        <div className="rag-sources__header">
          <div>
            <h2>Sources</h2>
            <p>
              {filtered.length} of {sources.length} source{sources.length === 1 ? "" : "s"}
            </p>
          </div>
          <div className="rag-sources__actions">
            <button className="rag-btn rag-btn--ghost" onClick={() => setModal("repo")}>
              Connect GitHub repo
            </button>
            <button className="rag-btn rag-btn--ghost" onClick={() => fileRef.current?.click()} disabled={uploading}>
              {uploading ? "Uploading…" : "Upload PDF/Doc/CSV"}
            </button>
            <input ref={fileRef} type="file" accept=".pdf,.docx,.csv" hidden onChange={handleFile} />
            <button className="rag-btn" onClick={() => setModal("text")}>
              Paste text
            </button>
          </div>
        </div>
        {uploadError && <p className="rag-error">{uploadError}</p>}

        {filtered.length === 0 ? (
          <p className="rag-hint">
            {sources.length === 0
              ? "No sources yet — connect a repo, upload a file, or paste text above."
              : "No sources match this filter."}
          </p>
        ) : (
          <div className="rag-source-list">
            {filtered.map((s) => (
              <SourceRow key={s.id} source={s} onRefresh={onRefresh} onOpenDetail={setDetailId} />
            ))}
          </div>
        )}
      </div>

      {modal === "repo" && (
        <RepoModal spaceId={spaceId} onClose={() => setModal(null)} onAdded={() => { setModal(null); onRefresh(); }} />
      )}
      {modal === "text" && (
        <TextModal spaceId={spaceId} onClose={() => setModal(null)} onAdded={() => { setModal(null); onRefresh(); }} />
      )}
      {detailSource && (
        <SourceDetailModal
          source={detailSource}
          onClose={() => setDetailId(null)}
          onDelete={handleDelete}
          onViewBreakdown={openBreakdown}
        />
      )}
      {loadingBreakdown && (
        <p className="rag-hint" style={{ position: "fixed", top: 16, right: 16 }}>
          Loading…
        </p>
      )}
      {breakdown && (
        <PipelineOverlay mode="ingest" data={breakdown.trace} title={breakdown.name} onClose={() => setBreakdown(null)} />
      )}
    </div>
  );
}
