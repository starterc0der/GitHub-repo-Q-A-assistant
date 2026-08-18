import { useEffect, useRef, useState } from "react";
import { createChat, deleteChat, listChats, listMessages, sendMessage } from "../api.js";
import { cls, ConfirmDialog } from "./RagAtoms.jsx";
import { BarChart } from "./BarChart.jsx";
import { DataTable } from "./DataTable.jsx";

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
  const lines = text.split("\n");
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

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={cls("rag-bubble", isUser ? "rag-bubble--user" : "rag-bubble--assistant")}>
      <MessageText text={message.content} />
      {message.chart && <BarChart chart={message.chart} />}
      {message.table && <DataTable table={message.table} />}
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
  const [deleteChatId, setDeleteChatId] = useState(null);
  const [streamingText, setStreamingText] = useState("");
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
  }, [messages, streamingText]);

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
    setStreamingText("");
    // Optimistic user bubble — replaced by the server's copy on the post-send refetch.
    setMessages((m) => [...m, { id: "pending-user", role: "user", content }]);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await sendMessage(chatId, content, controller.signal, (delta) =>
        setStreamingText((t) => t + delta)
      );
      const data = await listMessages(chatId);
      setMessages(data.messages);
      refreshChats(chatId);
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message);
      const data = await listMessages(chatId);
      setMessages(data.messages);
    } finally {
      setSending(false);
      setStreamingText("");
      abortRef.current = null;
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  return {
    chats, activeChatId, setActiveChatId, messages, input, setInput, sending, error,
    streamingText, scrollRef, handleNewChat, deleteChatId, setDeleteChatId, confirmDeleteChat,
    handleSend, handleStop,
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

export function ChatMain({ messages, sending, streamingText, error, input, setInput, handleSend, handleStop, scrollRef }) {
  return (
    <div className="rag-chat__main">
      <div className="rag-chat__messages-viewport" ref={scrollRef}>
        <div className="rag-chat__messages">
          {messages.length === 0 && <p className="rag-hint">Ask a question about this space's sources.</p>}
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {sending && (
            streamingText ? (
              <div className="rag-bubble rag-bubble--assistant">
                <MessageText text={streamingText} />
              </div>
            ) : (
              <div className="rag-bubble rag-bubble--assistant rag-bubble--thinking">Thinking…</div>
            )
          )}
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
    </div>
  );
}
