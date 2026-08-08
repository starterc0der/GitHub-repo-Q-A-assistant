import { useEffect, useState } from "react";
import { createSpace, deleteSpace, listSpaces } from "../api.js";

const COLORS = ["accent", "accent2", "warn"];

function timeAgo(iso) {
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function CreateSpaceModal({ onClose, onCreated }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [color, setColor] = useState(COLORS[0]);
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    try {
      await createSpace({ name: name.trim(), description: description.trim(), color });
      onCreated();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <form className="rag-modal-card" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <h2>New space</h2>
        <label className="rag-input-card__label">Name</label>
        <input className="rag-input" value={name} onChange={(e) => setName(e.target.value)} autoFocus />
        <label className="rag-input-card__label">Description</label>
        <input className="rag-input" value={description} onChange={(e) => setDescription(e.target.value)} />
        <label className="rag-input-card__label">Color</label>
        <div className="rag-swatches">
          {COLORS.map((c) => (
            <button
              type="button"
              key={c}
              className={`rag-swatch rag-dot--${c}${color === c ? " rag-swatch--selected" : ""}`}
              onClick={() => setColor(c)}
            />
          ))}
        </div>
        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="rag-btn" disabled={busy || !name.trim()}>
            {busy ? "Creating…" : "Create space"}
          </button>
        </div>
      </form>
    </div>
  );
}

export function SpacesView({ onOpen }) {
  const [spaces, setSpaces] = useState(null);
  const [error, setError] = useState(null);
  const [showCreate, setShowCreate] = useState(false);

  function refresh() {
    setError(null);
    listSpaces()
      .then((data) => setSpaces(data.spaces))
      .catch((err) => setError(err.message));
  }

  useEffect(refresh, []);

  async function handleDelete(e, id) {
    e.stopPropagation();
    if (!confirm("Delete this space and everything in it?")) return;
    await deleteSpace(id);
    refresh();
  }

  return (
    <div className="rag-spaces">
      <header className="rag-header">
        <div className="rag-brand">
          <span className="rag-brand__mark" />
          <div>
            <h1>Spaces</h1>
            <p>Each space keeps its own sources and chat history.</p>
          </div>
        </div>
        <button className="rag-btn" onClick={() => setShowCreate(true)}>
          + New space
        </button>
      </header>

      <div className="rag-spaces__list">
        {error ? (
          <div>
            <p className="rag-error">Could not reach the backend: {error}</p>
            <button className="rag-btn" onClick={refresh}>
              Retry
            </button>
          </div>
        ) : spaces === null ? (
          <p className="rag-hint">Loading…</p>
        ) : spaces.length === 0 ? (
          <p className="rag-hint">No spaces yet — create one to get started.</p>
        ) : (
          spaces.map((s) => (
            <div className="rag-space-row" key={s.id} onClick={() => onOpen(s.id)}>
              <span className={`rag-dot rag-dot--${s.color}`} />
              <div className="rag-space-row__body">
                <span className="rag-space-row__name">{s.name}</span>
                {s.description && <span className="rag-dim">{s.description}</span>}
              </div>
              <span className="rag-dim">{s.source_count} source{s.source_count === 1 ? "" : "s"}</span>
              <span className="rag-dim">{timeAgo(s.updated_at)}</span>
              <button className="rag-icon-btn" onClick={(e) => handleDelete(e, s.id)}>
                ×
              </button>
            </div>
          ))
        )}
      </div>

      {showCreate && (
        <CreateSpaceModal
          onClose={() => setShowCreate(false)}
          onCreated={() => {
            setShowCreate(false);
            refresh();
          }}
        />
      )}
    </div>
  );
}
