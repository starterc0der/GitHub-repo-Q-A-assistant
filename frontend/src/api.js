const API_BASE = "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

function post(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function ingestTrace(url) {
  return post("/ingest/trace", { url });
}

// GET + EventSource, not POST: browsers' EventSource can't send a request body, so the
// repo URL travels as a query param. Returns a cleanup function that closes the stream —
// call it on unmount/retry to avoid leaking a connection.
export function ingestTraceStream(url, { onProgress, onComplete, onError }) {
  const es = new EventSource(`${API_BASE}/ingest/trace/stream?url=${encodeURIComponent(url)}`);

  es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === "progress") {
      onProgress(data);
    } else if (data.type === "complete") {
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

export function queryTrace(question, repo) {
  return post("/query/trace", { question, repo });
}

export function listRepos() {
  return request("/repos");
}
