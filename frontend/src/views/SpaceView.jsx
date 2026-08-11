import { useEffect, useState } from "react";
import { getSpace } from "../api.js";
import { AVATAR_BG, AVATAR_INK, cls } from "../components/RagAtoms.jsx";
import { ChatMain, ChatSidebar, useChatController } from "../components/ChatPanel.jsx";
import { SourcesMain, SourcesSidebar, useSourcesController } from "../components/SourcesPanel.jsx";

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

  const chat = useChatController(spaceId);
  const sourcesCtl = useSourcesController(spaceId, space?.sources || [], refresh);

  if (error) {
    return (
      <div className="rag-spaces">
        <header className="rag-header">
          <button className="rag-back-btn" onClick={onBack}>
            ← Spaces
          </button>
        </header>
        <div className="rag-space-error">
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
      <aside className="rag-space-sidebar">
        <div className="rag-space-sidebar__head">
          <button className="rag-space-sidebar__back" onClick={onBack}>
            ← Spaces
          </button>
          <div className="rag-space-sidebar__info">
            <span
              className="rag-avatar rag-avatar--sm"
              style={{ background: AVATAR_BG[space.color] || AVATAR_BG.accent, color: AVATAR_INK[space.color] || AVATAR_INK.accent }}
            >
              {space.name.trim().charAt(0).toUpperCase() || "?"}
            </span>
            <div>
              <div className="rag-space-sidebar__info-name">{space.name}</div>
              <div className="rag-space-sidebar__info-count">
                {space.sources.length} source{space.sources.length === 1 ? "" : "s"}
              </div>
            </div>
          </div>
          <div className="rag-space-sidebar__tabs">
            <button className={cls("rag-space-sidebar__tab", tab === "chat" && "rag-space-sidebar__tab--active")} onClick={() => setTab("chat")}>
              Chat
            </button>
            <button
              className={cls("rag-space-sidebar__tab", tab === "sources" && "rag-space-sidebar__tab--active")}
              onClick={() => setTab("sources")}
            >
              Sources ({space.sources.length})
            </button>
          </div>
        </div>
        <div className="rag-space-sidebar__content">
          {tab === "chat" ? <ChatSidebar {...chat} /> : <SourcesSidebar {...sourcesCtl} />}
        </div>
      </aside>

      <div className="rag-space__main">
        {tab === "chat" ? <ChatMain {...chat} /> : <SourcesMain {...sourcesCtl} />}
      </div>
    </div>
  );
}
