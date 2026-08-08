import { useEffect, useState } from "react";
import { getSpace } from "../api.js";
import { cls } from "../components/RagAtoms.jsx";
import { ChatPanel } from "../components/ChatPanel.jsx";
import { SourcesPanel } from "../components/SourcesPanel.jsx";

export function SpaceView({ spaceId, onBack }) {
  const [space, setSpace] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("chat");

  function refresh() {
    setError(null);
    getSpace(spaceId)
      .then(setSpace)
      .catch((err) => setError(err.message));
  }

  useEffect(refresh, [spaceId]);

  if (error) {
    return (
      <div className="rag-spaces">
        <header className="rag-header">
          <button className="rag-back-btn" onClick={onBack}>
            ← Spaces
          </button>
        </header>
        <div className="rag-spaces__list">
          <p className="rag-error">Could not load this space: {error}</p>
          <button className="rag-btn" onClick={refresh}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!space) return <p className="rag-hint" style={{ margin: 24 }}>Loading…</p>;

  return (
    <div className="rag-space">
      <header className="rag-header">
        <button className="rag-back-btn" onClick={onBack}>
          ← Spaces
        </button>
        <div className="rag-brand">
          <span className={`rag-dot rag-dot--${space.color}`} />
          <div>
            <h1>{space.name}</h1>
            {space.description && <p>{space.description}</p>}
          </div>
        </div>
        <div className="rag-tabs">
          <button className={cls("rag-tabs__btn", tab === "chat" && "rag-tabs__btn--active")} onClick={() => setTab("chat")}>
            Chat
          </button>
          <button
            className={cls("rag-tabs__btn", tab === "sources" && "rag-tabs__btn--active")}
            onClick={() => setTab("sources")}
          >
            Sources ({space.sources.length})
          </button>
        </div>
      </header>

      <div className="rag-space__body">
        {tab === "chat" ? (
          <ChatPanel spaceId={spaceId} />
        ) : (
          <SourcesPanel spaceId={spaceId} sources={space.sources} onRefresh={refresh} />
        )}
      </div>
    </div>
  );
}
