const API_BASE = "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

function post(path, body, options) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    ...options,
  });
}

function patch(path, body) {
  return request(path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function del(path) {
  return request(path, { method: "DELETE" });
}

// ------------------------------------------------------------------- spaces

export const listSpaces = () => request("/spaces");
export const createSpace = (body) => post("/spaces", body);
export const getSpace = (id) => request(`/spaces/${id}`);
export const updateSpace = (id, body) => patch(`/spaces/${id}`, body);
export const deleteSpace = (id) => del(`/spaces/${id}`);

// ------------------------------------------------------------------ sources

export const listSources = (spaceId) => request(`/spaces/${spaceId}/sources`);
export const createSource = (spaceId, body) => post(`/spaces/${spaceId}/sources`, body);
export const deleteSource = (id) => del(`/sources/${id}`);
export const sourceTrace = (id) => request(`/sources/${id}/trace`);

export async function uploadSource(spaceId, kind, file) {
  const form = new FormData();
  form.append("kind", kind);
  form.append("file", file);
  return request(`/spaces/${spaceId}/sources/upload`, { method: "POST", body: form });
}

// GET + EventSource: polls the source row on the server and streams whatever changed.
// Reconnect-safe by construction — a refresh just re-attaches to the row's current state.
export function sourceIngestStream(sourceId, { onProgress, onComplete, onError }) {
  const es = new EventSource(`${API_BASE}/sources/${sourceId}/ingest/stream`);
  es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === "progress") onProgress(data);
    else if (data.type === "complete") {
      es.close();
      onComplete(data.trace);
    } else if (data.type === "error") {
      es.close();
      onError(new Error(data.message));
    }
  };
  es.onerror = () => {
    es.close();
    onError(new Error("Lost connection to the ingest stream."));
  };
  return () => es.close();
}

// --------------------------------------------------------------------- chats

export const listChats = (spaceId) => request(`/spaces/${spaceId}/chats`);
export const createChat = (spaceId, title) => post(`/spaces/${spaceId}/chats`, { title });
export const deleteChat = (id) => del(`/chats/${id}`);
export const listMessages = (chatId) => request(`/chats/${chatId}/messages`);
export const messageTrace = (id) => request(`/messages/${id}/trace`);
export const messageVectors = (id) => request(`/messages/${id}/vectors`);

// ----------------------------------------------------------------- insights

export const spaceInsights = (spaceId, range) => {
  const qs = range?.start && range?.end ? `?start=${range.start}&end=${range.end}` : "";
  return request(`/spaces/${spaceId}/insights${qs}`);
};
export const chatInsights = (spaceId, chatId) => request(`/spaces/${spaceId}/chats/${chatId}/insights`);
export const chunkInsights = (spaceId) => request(`/spaces/${spaceId}/insights/chunks`);

// POST + a streamed fetch body: EventSource can't send a POST body, so SSE frames are
// parsed by hand here instead. signal: an AbortController signal — aborting closes the
// fetch, which the backend detects via request.is_disconnected() and uses to cancel
// in-flight LLM calls. onDelta fires per text chunk as the answer streams in; the
// resolved value is the final persisted message (or null if the stream ended without a
// "done" event, e.g. cancellation — the caller's own refetch is the source of truth then).
export async function sendMessage(chatId, content, signal, onDelta) {
  const res = await fetch(`${API_BASE}/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
    signal,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`send message failed (${res.status}): ${text}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let message = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop(); // last piece may be a partial frame — kept for the next read
    for (const frame of frames) {
      if (!frame.startsWith("data: ")) continue;
      const event = JSON.parse(frame.slice("data: ".length));
      if (event.type === "delta") onDelta?.(event.text);
      else if (event.type === "done") message = event.message;
      else if (event.type === "error") throw new Error(event.message);
    }
  }
  return message;
}
