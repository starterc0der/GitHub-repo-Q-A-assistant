import { useEffect, useRef, useState } from "react";
import { createSource, deleteSource, sourceIngestStream, sourceTrace, uploadSource } from "../api.js";
import { PipelineOverlay } from "./PipelineOverlay.jsx";

const KIND_LABEL = { repo: "REPO", pdf: "PDF", docx: "DOC", text: "TXT" };

function SourceBadge({ kind }) {
  return <span className="rag-tag rag-tag--neutral">{KIND_LABEL[kind] || kind.toUpperCase()}</span>;
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

function SourceRow({ source, onRefresh, onDelete }) {
  const progress = useIngestProgress(source, onRefresh);
  const [breakdown, setBreakdown] = useState(null);
  const [loadingBreakdown, setLoadingBreakdown] = useState(false);

  async function openBreakdown() {
    setLoadingBreakdown(true);
    try {
      setBreakdown(await sourceTrace(source.id));
    } catch (err) {
      alert(err.message);
    } finally {
      setLoadingBreakdown(false);
    }
  }

  return (
    <div className="rag-source-row">
      <div className="rag-source-row__main">
        <SourceBadge kind={source.kind} />
        <span className="rag-source-row__name">{source.name}</span>
        {source.status === "ready" && (
          <span className="rag-dim rag-source-row__stat">
            {source.file_count} files · {source.chunk_count} chunks
          </span>
        )}
        {source.status === "ingesting" && (
          <span className="rag-dim rag-source-row__stat">{progress?.message || "Ingesting…"}</span>
        )}
        {source.status === "failed" && <span className="rag-error">{source.error}</span>}
      </div>
      <div className="rag-source-row__actions">
        {source.has_trace && (
          <button className="rag-btn rag-btn--ghost" onClick={openBreakdown} disabled={loadingBreakdown}>
            {loadingBreakdown ? "Loading…" : "View ingestion breakdown"}
          </button>
        )}
        <button className="rag-icon-btn" title="Delete source" onClick={() => onDelete(source.id)}>
          ×
        </button>
      </div>
      {breakdown && (
        <PipelineOverlay mode="ingest" data={breakdown} title={source.name} onClose={() => setBreakdown(null)} />
      )}
    </div>
  );
}

function AddSourceForm({ spaceId, onAdded }) {
  const [mode, setMode] = useState(null); // null | "repo" | "text" | "file"
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [textName, setTextName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  async function submitRepo(e) {
    e.preventDefault();
    if (!url.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await createSource(spaceId, { kind: "repo", uri: url.trim() });
      setUrl("");
      setMode(null);
      onAdded();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function submitText(e) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await createSource(spaceId, { kind: "text", name: textName.trim() || undefined, text });
      setText("");
      setTextName("");
      setMode(null);
      onAdded();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    const kind = ext === "pdf" ? "pdf" : ext === "docx" ? "docx" : null;
    if (!kind) {
      setError("Only .pdf and .docx files are supported.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await uploadSource(spaceId, kind, file);
      onAdded();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rag-add-source">
      <div className="rag-add-source__buttons">
        <button className="rag-btn rag-btn--ghost" onClick={() => setMode(mode === "repo" ? null : "repo")}>
          Connect GitHub repo
        </button>
        <button className="rag-btn rag-btn--ghost" onClick={() => fileRef.current?.click()} disabled={busy}>
          Upload PDF/Doc
        </button>
        <input ref={fileRef} type="file" accept=".pdf,.docx" hidden onChange={handleFile} />
        <button className="rag-btn rag-btn--ghost" onClick={() => setMode(mode === "text" ? null : "text")}>
          Paste text
        </button>
      </div>

      {mode === "repo" && (
        <form className="rag-add-source__form" onSubmit={submitRepo}>
          <input
            className="rag-input"
            placeholder="https://github.com/owner/repo"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={busy}
            autoFocus
          />
          <button className="rag-btn" type="submit" disabled={busy}>
            {busy ? "Adding…" : "Add"}
          </button>
        </form>
      )}

      {mode === "text" && (
        <form className="rag-add-source__form rag-add-source__form--stacked" onSubmit={submitText}>
          <input
            className="rag-input"
            placeholder="Name (optional)"
            value={textName}
            onChange={(e) => setTextName(e.target.value)}
            disabled={busy}
          />
          <textarea
            className="rag-input rag-textarea"
            placeholder="Paste text to index…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={busy}
            rows={5}
          />
          <button className="rag-btn" type="submit" disabled={busy}>
            {busy ? "Adding…" : "Add"}
          </button>
        </form>
      )}

      {busy && mode === null && <p className="rag-hint">Uploading…</p>}
      {error && <p className="rag-error">{error}</p>}
    </div>
  );
}

export function SourcesPanel({ spaceId, sources, onRefresh }) {
  async function handleDelete(sourceId) {
    if (!confirm("Delete this source? This removes its indexed data too.")) return;
    await deleteSource(sourceId);
    onRefresh();
  }

  return (
    <div className="rag-sources-panel">
      <AddSourceForm spaceId={spaceId} onAdded={onRefresh} />
      {sources.length === 0 ? (
        <p className="rag-hint">No sources yet — connect a repo, upload a file, or paste text above.</p>
      ) : (
        <div className="rag-source-list">
          {sources.map((s) => (
            <SourceRow key={s.id} source={s} onRefresh={onRefresh} onDelete={handleDelete} />
          ))}
        </div>
      )}
    </div>
  );
}
