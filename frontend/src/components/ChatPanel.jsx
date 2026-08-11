import { useEffect, useRef, useState } from "react";
import { createChat, deleteChat, listChats, listMessages, messageTrace, sendMessage } from "../api.js";
import { cls, ConfirmDialog } from "./RagAtoms.jsx";
import { BarChart } from "./BarChart.jsx";
import { PipelineOverlay } from "./PipelineOverlay.jsx";

// [file:Lstart-Lend] markers ground the answer, but they're noise in a casual chat
// bubble — strip them here. They're still fully visible (as resolved, clickable
// citations with snippets) in "View pipeline breakdown", which is where verifying a
// claim actually belongs. Mirrors src/generate/answer.py's BRACKET_PATTERN/GROUP_PATTERN:
// a bracket can hold grouped ranges for one file ([path:L1-L5, L9-L12]), a bare single
// line ([path:L5]), or several path:range groups joined with "; " for one claim
// ([pathA:L1-L5; pathB:L9-L12] — common once a PDF "path" is "name · p.N" and two pages
// support the same sentence). Checked group-by-group so a bracket only strips when
// every "; "-separated part is actually a citation — unrelated bracket text stays.
const CITATION_BRACKET_RE = /\s?\[([^\[\]]+)\]/g;
const CITATION_GROUP_RE = /^[^:;]+:L\d+(?:-L\d+)?(?:\s*,\s*L\d+(?:-L\d+)?)*$/;

function stripCitations(text) {
  return text.replace(CITATION_BRACKET_RE, (match, inner) => {
    const groups = inner.split(";").map((g) => g.trim());
    return groups.every((g) => CITATION_GROUP_RE.test(g)) ? "" : match;
  });
}

// **bold** and `code` spans only — not a full markdown parser, just the two things the
// answer model actually produces. Built as React elements (never dangerouslySetInnerHTML),
// so there's no HTML-injection surface to sanitize against.
const INLINE_RE = /\*\*(.+?)\*\*|`([^`]+)`/g;

function renderInline(text) {
  const nodes = [];
  let last = 0;
  let match;
  while ((match = INLINE_RE.exec(text))) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    if (match[1] !== undefined) nodes.push(<strong key={nodes.length}>{match[1]}</strong>);
    else nodes.push(<code className="rag-inline-code" key={nodes.length}>{match[2]}</code>);
    last = INLINE_RE.lastIndex;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

// Splits into paragraphs and "* "/"- " bullet lists — the two block-level shapes the
// answer model actually produces — then applies inline formatting within each.
function MessageText({ text }) {
  const lines = stripCitations(text).split("\n");
  const blocks = [];
  let listItems = null;
  let paraLines = [];

  const flushPara = () => {
    if (paraLines.length) {
      blocks.push(<p className="rag-bubble__text" key={blocks.length}>{renderInline(paraLines.join(" "))}</p>);
      paraLines = [];
    }
  };
  const flushList = () => {
    if (listItems) {
      blocks.push(<ul className="rag-bubble__list" key={blocks.length}>{listItems}</ul>);
      listItems = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    const bullet = line.match(/^[*-]\s+(.*)$/);
    if (bullet) {
      flushPara();
      listItems ??= [];
      listItems.push(<li key={listItems.length}>{renderInline(bullet[1])}</li>);
    } else if (line === "") {
      flushPara();
      flushList();
    } else {
      flushList();
      paraLines.push(line);
    }
  }
  flushPara();
  flushList();
  return <>{blocks}</>;
}

function MessageBubble({ message, onViewTrace }) {
  const isUser = message.role === "user";
  return (
    <div className={cls("rag-bubble", isUser ? "rag-bubble--user" : "rag-bubble--assistant")}>
      <MessageText text={message.content} />
      {message.chart && <BarChart chart={message.chart} />}
      {!isUser && (
        <div className="rag-bubble__foot">
          {message.cache_hit ? <span className="rag-tag rag-tag--accent2">served from cache</span> : null}
          {message.has_trace ? (
            <button className="rag-link-btn" onClick={() => onViewTrace(message.id)}>
              View pipeline breakdown
            </button>
          ) : null}
        </div>
      )}
    </div>
  );
}

export function useChatController(spaceId) {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const [trace, setTrace] = useState(null);
  const [deleteChatId, setDeleteChatId] = useState(null);
  const abortRef = useRef(null);
  const scrollRef = useRef(null);

  async function refreshChats(selectId) {
    const data = await listChats(spaceId);
    setChats(data.chats);
    if (selectId) setActiveChatId(selectId);
    else if (!activeChatId && data.chats.length > 0) setActiveChatId(data.chats[0].id);
  }

  useEffect(() => {
    setActiveChatId(null);
    setMessages([]);
    refreshChats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spaceId]);

  useEffect(() => {
    if (!activeChatId) return;
    listMessages(activeChatId).then((data) => setMessages(data.messages));
  }, [activeChatId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function handleNewChat() {
    const chat = await createChat(spaceId);
    await refreshChats(chat.id);
    setMessages([]);
  }

  async function confirmDeleteChat() {
    const chatId = deleteChatId;
    setDeleteChatId(null);
    await deleteChat(chatId);
    if (activeChatId === chatId) setActiveChatId(null);
    refreshChats();
  }

  async function handleSend(e) {
    e.preventDefault();
    const content = input.trim();
    if (!content || sending) return;

    let chatId = activeChatId;
    if (!chatId) {
      const chat = await createChat(spaceId);
      await refreshChats(chat.id);
      chatId = chat.id;
    }

    setInput("");
    setSending(true);
    setError(null);
    // Optimistic user bubble — replaced by the server's copy on the post-send refetch.
    setMessages((m) => [...m, { id: "pending-user", role: "user", content }]);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await sendMessage(chatId, content, controller.signal);
      const data = await listMessages(chatId);
      setMessages(data.messages);
      refreshChats(chatId);
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message);
      const data = await listMessages(chatId);
      setMessages(data.messages);
    } finally {
      setSending(false);
      abortRef.current = null;
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  async function handleViewTrace(messageId) {
    setTrace(await messageTrace(messageId));
  }

  return {
    chats, activeChatId, setActiveChatId, messages, input, setInput, sending, error, trace, setTrace,
    scrollRef, handleNewChat, deleteChatId, setDeleteChatId, confirmDeleteChat, handleSend, handleStop, handleViewTrace,
  };
}

export function ChatSidebar({ chats, activeChatId, setActiveChatId, handleNewChat, deleteChatId, setDeleteChatId, confirmDeleteChat }) {
  return (
    <>
      <button className="rag-chat__new-btn" onClick={handleNewChat}>
        <span style={{ fontSize: 15, lineHeight: 1 }}>+</span> New chat
      </button>
      <div className="rag-chat__list">
        {chats.map((c) => (
          <div
            key={c.id}
            className={cls("rag-chat__item", c.id === activeChatId && "rag-chat__item--active")}
            onClick={() => setActiveChatId(c.id)}
          >
            <span className="rag-chat__item-title">{c.title}</span>
            <button
              className="rag-icon-btn"
              onClick={(e) => {
                e.stopPropagation();
                setDeleteChatId(c.id);
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {deleteChatId && (
        <ConfirmDialog
          title="Delete this chat?"
          message="This permanently deletes the chat and its messages."
          onCancel={() => setDeleteChatId(null)}
          onConfirm={confirmDeleteChat}
        />
      )}
    </>
  );
}

// No retrieval ran for a meta answer ("what have we discussed") — this shows the actual
// transcript sent to the model instead of the PipelineOverlay's route/hybrid/rerank/
// compress rail, which has nothing to show for a plain history-grounded reply.
function MetaTraceModal({ data, onClose }) {
  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <div className="rag-modal-card rag-modal-card--wide" onClick={(e) => e.stopPropagation()}>
        <h2>{data.question}</h2>
        <p className="rag-dim">
          Conversation-only answer — no document retrieval ran. Answered from{" "}
          {data.history.length} prior message{data.history.length === 1 ? "" : "s"} in this chat.
        </p>
        <div className="rag-meta-trace">
          {data.history.map((m, i) => (
            <div key={i} className={cls("rag-meta-trace__turn", `rag-meta-trace__turn--${m.role}`)}>
              <span className="rag-meta-trace__role">{m.role}</span>
              <span className="rag-meta-trace__text">{m.content}</span>
            </div>
          ))}
        </div>
        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export function ChatMain({ messages, sending, error, input, setInput, handleSend, handleStop, handleViewTrace, scrollRef, trace, setTrace }) {
  return (
    <div className="rag-chat__main">
      <div className="rag-chat__messages-viewport" ref={scrollRef}>
        <div className="rag-chat__messages">
          {messages.length === 0 && <p className="rag-hint">Ask a question about this space's sources.</p>}
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} onViewTrace={handleViewTrace} />
          ))}
          {sending && <div className="rag-bubble rag-bubble--assistant rag-bubble--thinking">Thinking…</div>}
        </div>
      </div>
      {error && <p className="rag-error">{error}</p>}
      <form className="rag-chat__composer" onSubmit={handleSend}>
        <input
          className="rag-input"
          placeholder="Ask a question…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={sending}
        />
        {sending ? (
          <button className="rag-btn rag-btn--ghost" type="button" onClick={handleStop}>
            Stop
          </button>
        ) : (
          <button className="rag-btn" type="submit" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>

      {trace && (
        trace.meta ? (
          <MetaTraceModal data={trace} onClose={() => setTrace(null)} />
        ) : (
          <PipelineOverlay mode="query" data={trace} title={trace.question} onClose={() => setTrace(null)} />
        )
      )}
    </div>
  );
}
