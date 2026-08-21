import { useEffect, useState } from "react";
import { createSpace, deleteSpace, listSpaces, logout } from "../api.js";
import { AVATAR_BG, AVATAR_INK, ConfirmDialog, timeAgo } from "../components/RagAtoms.jsx";

const COLORS = ["accent", "accent2", "warn", "doc"];

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

export function SpacesView({ currentUser, onOpen, onOpenUsers, onLoggedOut }) {
  const [spaces, setSpaces] = useState(null);
  const [error, setError] = useState(null);
  const [showCreate, setShowCreate] = useState(false);
  const [deleteId, setDeleteId] = useState(null);
  const isAdmin = currentUser?.role === "admin";

  function refresh() {
    setError(null);
    listSpaces()
      .then((data) => setSpaces(data.spaces))
      .catch((err) => setError(err.message));
  }

  useEffect(refresh, []);

  function handleDelete(e, id) {
    e.stopPropagation();
    setDeleteId(id);
  }

  async function confirmDelete() {
    await deleteSpace(deleteId);
    setDeleteId(null);
    refresh();
  }

  return (
    <div className="rag-spaces-shell">
      <aside className="rag-spaces__sidebar">
        <div className="rag-spaces__brand">
          <span className="rag-spaces__brand-mark">S</span>
          <div>
            <h1>Spaces</h1>
            <p>Ask your own data</p>
          </div>
        </div>
        {isAdmin && (
          <button className="rag-spaces__new-btn" onClick={() => setShowCreate(true)}>
            + New space
          </button>
        )}
        <div>
          <div className="rag-spaces__nav-label">Workspace</div>
          <div className="rag-spaces__nav-list">
            <button className="rag-spaces__nav-item rag-spaces__nav-item--active">
              <span className="rag-spaces__nav-dot" />
              All spaces
            </button>
            {isAdmin && (
              <button className="rag-spaces__nav-item" onClick={onOpenUsers}>
                <span className="rag-spaces__nav-dot" />
                Users
              </button>
            )}
          </div>
        </div>
        <div className="rag-spaces__footer">
          <div className="rag-spaces__footer-identity">
            <span className="rag-spaces__footer-avatar">{currentUser?.name?.trim().charAt(0).toUpperCase() || "?"}</span>
            <span className="rag-spaces__footer-name">{currentUser?.name}</span>
          </div>
          <button
            className="rag-spaces__logout-btn"
            onClick={() => logout().finally(onLoggedOut)}
          >
            Log out
          </button>
        </div>
      </aside>

      <div className="rag-spaces__main">
        <div className="rag-spaces__container">
          <div className="rag-spaces__header">
            <div>
              <h1>Your spaces</h1>
              <p>Each space keeps its own sources and chat history.</p>
            </div>
            {spaces && spaces.length > 0 && (
              <span className="rag-count-pill">
                {spaces.length} space{spaces.length === 1 ? "" : "s"}
              </span>
            )}
          </div>

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
            <div className="rag-spaces__list">
              {spaces.map((s) => (
                <div className="rag-space-row" key={s.id} onClick={() => onOpen(s.id)}>
                  <span
                    className="rag-avatar"
                    style={{ background: AVATAR_BG[s.color] || AVATAR_BG.accent, color: AVATAR_INK[s.color] || AVATAR_INK.accent }}
                  >
                    {s.name.trim().charAt(0).toUpperCase() || "?"}
                  </span>
                  <div className="rag-space-row__body">
                    <span className="rag-space-row__name">{s.name}</span>
                    {s.description && <span className="rag-dim">{s.description}</span>}
                  </div>
                  <span className="rag-space-row__pill">
                    {s.source_count} source{s.source_count === 1 ? "" : "s"}
                  </span>
                  <span className="rag-space-row__time">{timeAgo(s.updated_at)}</span>
                  {isAdmin && (
                    <button className="rag-icon-btn rag-space-row__delete" onClick={(e) => handleDelete(e, s.id)}>
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
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

      {deleteId && (
        <ConfirmDialog
          title="Delete this space?"
          message="This deletes the space and everything in it — sources, chats, and messages."
          onCancel={() => setDeleteId(null)}
          onConfirm={confirmDelete}
        />
      )}
    </div>
  );
}
