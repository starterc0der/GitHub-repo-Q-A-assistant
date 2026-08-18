import { useEffect, useState } from "react";
import {
  createConnector, deleteConnector, listConnectors, replaceConnectorCredentials,
  retestConnector, testConnectorCredentials,
} from "../api.js";
import { cls, ConfirmDialog, timeAgo } from "./RagAtoms.jsx";

const KIND_LABEL = { redis: "Redis", postgres: "Postgres" };
const STATUS_LABEL = { connected: "Connected", error: "Error", untested: "Untested" };
const STATUS_TONE = { connected: "accent2", error: "warn", untested: "neutral" };

const EMPTY_FORM = { kind: "redis", name: "", host: "", port: 6379, database: "", username: "", password: "", db_index: 0, tls: false, ssl: false };

function subtitle(c) {
  return c.kind === "postgres" ? `${c.database || "—"}@${c.host}:${c.port}` : `${c.host}:${c.port}`;
}

function StatusPill({ status }) {
  return <span className={cls("rag-tag", `rag-tag--${STATUS_TONE[status]}`)}>{STATUS_LABEL[status] || status}</span>;
}

function ConnectorRow({ connector, onOpen }) {
  return (
    <div className="rag-source-row rag-source-row--clickable" onClick={() => onOpen(connector.id)}>
      <div className="rag-source-row__main">
        <span className="rag-tag rag-tag--doc">{KIND_LABEL[connector.kind]}</span>
        <span className="rag-source-row__name">{connector.name}</span>
        <span className="rag-dim rag-source-row__stat">{subtitle(connector)}</span>
        <span className="rag-source-row__added">
          {connector.last_tested_at ? `tested ${timeAgo(connector.last_tested_at)}` : "never tested"}
        </span>
      </div>
      <StatusPill status={connector.status} />
    </div>
  );
}

function TestResultBanner({ result }) {
  if (!result) return null;
  return (
    <p className={cls("rag-error", result.ok && "rag-error--ok")} style={!result.ok ? undefined : { color: "var(--accent2-ink)" }}>
      {result.ok ? "✓ " : "✗ "}{result.message}
    </p>
  );
}

function ConnectorForm({ spaceId, mode, kind, connector, onClose, onSaved }) {
  const [form, setForm] = useState(() =>
    mode === "replace"
      ? { ...EMPTY_FORM, kind: connector.kind, host: connector.host, port: connector.port, database: connector.database || "", username: connector.username || "" }
      : { ...EMPTY_FORM, kind, port: kind === "postgres" ? 5432 : 6379 }
  );
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [showPassword, setShowPassword] = useState(false);

  function set(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
    setTestResult(null); // any credential change invalidates the prior test
  }

  async function runTest() {
    setTesting(true);
    setError(null);
    try {
      const result = await testConnectorCredentials(spaceId, {
        kind: form.kind, host: form.host, port: Number(form.port), database: form.database || null,
        username: form.username || null, password: form.password, db_index: form.kind === "redis" ? Number(form.db_index) : null,
        tls: form.tls, ssl: form.ssl,
      });
      setTestResult(result);
    } catch (err) {
      setTestResult({ ok: false, message: err.message });
    } finally {
      setTesting(false);
    }
  }

  async function save() {
    setSaving(true);
    setError(null);
    try {
      if (mode === "replace") {
        await replaceConnectorCredentials(connector.id, { username: form.username || null, password: form.password });
      } else {
        await createConnector(spaceId, {
          kind: form.kind, name: form.name, host: form.host, port: Number(form.port), database: form.database || null,
          username: form.username || null, password: form.password, db_index: form.kind === "redis" ? Number(form.db_index) : null,
          tls: form.tls, ssl: form.ssl,
        });
      }
      onSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  const locked = mode === "replace";
  const busy = testing || saving;

  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <div className="rag-modal-card" onClick={(e) => e.stopPropagation()}>
        <h2>{mode === "replace" ? "Replace credentials" : `Add ${KIND_LABEL[form.kind]} connector`}</h2>

        {!locked && (
          <>
            <label className="rag-input-card__label">Name</label>
            <input className="rag-input" value={form.name} onChange={(e) => set("name", e.target.value)} disabled={busy} placeholder={`${KIND_LABEL[form.kind]} connector`} />
            <label className="rag-input-card__label">Host</label>
            <input className="rag-input" value={form.host} onChange={(e) => set("host", e.target.value)} disabled={busy} autoFocus />
            <label className="rag-input-card__label">Port</label>
            <input className="rag-input" type="number" value={form.port} onChange={(e) => set("port", e.target.value)} disabled={busy} />
            {form.kind === "postgres" && (
              <>
                <label className="rag-input-card__label">Database</label>
                <input className="rag-input" value={form.database} onChange={(e) => set("database", e.target.value)} disabled={busy} />
              </>
            )}
          </>
        )}
        {locked && (
          <p className="rag-dim" style={{ margin: "0 0 4px" }}>
            {KIND_LABEL[form.kind]} · {form.kind === "postgres" ? `${form.database}@${form.host}:${form.port}` : `${form.host}:${form.port}`}
          </p>
        )}

        <label className="rag-input-card__label">Username</label>
        <input className="rag-input" value={form.username} onChange={(e) => set("username", e.target.value)} disabled={busy} placeholder={form.kind === "redis" ? "optional" : ""} />
        <label className="rag-input-card__label">Password</label>
        <div className="rag-password-field">
          <input
            className="rag-input" type={showPassword ? "text" : "password"} value={form.password}
            onChange={(e) => set("password", e.target.value)} disabled={busy} autoFocus={locked}
          />
          <button
            type="button" className="rag-password-field__toggle" onClick={() => setShowPassword((s) => !s)}
            disabled={busy} aria-label={showPassword ? "Hide password" : "Show password"}
            title={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? "🙈" : "👁"}
          </button>
        </div>

        {!locked && form.kind === "redis" && (
          <>
            <label className="rag-input-card__label">DB index</label>
            <input className="rag-input" type="number" value={form.db_index} onChange={(e) => set("db_index", e.target.value)} disabled={busy} style={{ maxWidth: 100 }} />
            <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8, fontSize: 13 }}>
              <input type="checkbox" checked={form.tls} onChange={(e) => set("tls", e.target.checked)} disabled={busy} /> Use TLS
            </label>
          </>
        )}
        {!locked && form.kind === "postgres" && (
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8, fontSize: 13 }}>
            <input type="checkbox" checked={form.ssl} onChange={(e) => set("ssl", e.target.checked)} disabled={busy} /> Use SSL
          </label>
        )}

        <TestResultBanner result={testResult} />
        {error && <p className="rag-error">{error}</p>}

        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" onClick={onClose} disabled={busy}>Cancel</button>
          <button type="button" className="rag-btn rag-btn--ghost" onClick={runTest} disabled={busy || !form.host || !form.password}>
            {testing ? "Testing…" : "Test connection"}
          </button>
          <button type="button" className="rag-btn" onClick={save} disabled={busy || !testResult?.ok}>
            {saving ? "Saving…" : mode === "replace" ? "Save credentials" : "Save connector"}
          </button>
        </div>
      </div>
    </div>
  );
}

function TypePickerModal({ onClose, onChoose }) {
  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <div className="rag-modal-card" onClick={(e) => e.stopPropagation()}>
        <h2>Add connector</h2>
        <p className="rag-dim" style={{ margin: "0 0 8px" }}>What are you connecting to?</p>
        <div className="rag-modal-card__actions" style={{ justifyContent: "stretch" }}>
          <button type="button" className="rag-btn rag-btn--ghost" style={{ flex: 1 }} onClick={() => onChoose("redis")}>Redis</button>
          <button type="button" className="rag-btn rag-btn--ghost" style={{ flex: 1 }} onClick={() => onChoose("postgres")}>Postgres</button>
        </div>
        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

function ConnectorDetailModal({ connector, onClose, onReplace, onDelete, onRetest, retesting }) {
  return (
    <div className="rag-modal-backdrop" onClick={onClose}>
      <div className="rag-modal-card" onClick={(e) => e.stopPropagation()}>
        <h2>{connector.name}</h2>
        <div className="rag-source-detail__row">
          <span className="rag-source-detail__k">Type</span>
          <span className="rag-tag rag-tag--doc">{KIND_LABEL[connector.kind]}</span>
        </div>
        <div className="rag-source-detail__row">
          <span className="rag-source-detail__k">Address</span>
          <span className="rag-dim">{subtitle(connector)}</span>
        </div>
        <div className="rag-source-detail__row">
          <span className="rag-source-detail__k">Status</span>
          <StatusPill status={connector.status} />
        </div>
        <div className="rag-source-detail__row">
          <span className="rag-source-detail__k">Last tested</span>
          <span className="rag-dim">{connector.last_tested_at ? timeAgo(connector.last_tested_at) : "never"}</span>
        </div>

        {connector.history?.length > 0 && (
          <>
            <p className="rag-input-card__label" style={{ marginTop: 10 }}>Test history</p>
            <div className="rag-meta-trace">
              {connector.history.slice(0, 3).map((h, i) => (
                <div key={i} className="rag-meta-trace__turn">
                  <span className={h.ok ? "rag-dim" : "rag-error"} style={h.ok ? { color: "var(--accent2-ink)" } : undefined}>
                    {h.ok ? "✓" : "✗"}
                  </span>
                  <span className="rag-meta-trace__text">{h.message} · {timeAgo(h.tested_at)}</span>
                </div>
              ))}
            </div>
          </>
        )}

        <p className="rag-hint" style={{ marginTop: 10 }}>No tools use this connector yet.</p>

        <div className="rag-modal-card__actions">
          <button type="button" className="rag-btn rag-btn--ghost" style={{ color: "var(--warn-ink)" }} onClick={() => onDelete(connector.id)}>
            Remove
          </button>
          <button type="button" className="rag-btn rag-btn--ghost" onClick={() => onRetest(connector.id)} disabled={retesting}>
            {retesting ? "Testing…" : "Re-test"}
          </button>
          <button type="button" className="rag-btn rag-btn--ghost" onClick={() => onReplace(connector)}>
            Replace credentials
          </button>
          <button type="button" className="rag-btn" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

export function ConnectorsPanel({ spaceId }) {
  const [connectors, setConnectors] = useState(null);
  const [error, setError] = useState(null);
  const [typePicker, setTypePicker] = useState(false);
  const [formState, setFormState] = useState(null); // { mode: "add"|"replace", kind, connector? }
  const [detailId, setDetailId] = useState(null);
  const [deleteId, setDeleteId] = useState(null);
  const [retestingId, setRetestingId] = useState(null);

  function refresh() {
    listConnectors(spaceId).then((r) => setConnectors(r.connectors)).catch((err) => setError(err.message));
  }

  useEffect(refresh, [spaceId]);

  const detail = connectors?.find((c) => c.id === detailId) || null;

  async function retest(id) {
    setRetestingId(id);
    try {
      await retestConnector(id);
      refresh();
    } finally {
      setRetestingId(null);
    }
  }

  async function confirmDelete() {
    await deleteConnector(deleteId);
    setDeleteId(null);
    setDetailId(null);
    refresh();
  }

  if (error) return <p className="rag-error" style={{ margin: 24 }}>{error}</p>;
  if (!connectors) return <p className="rag-hint" style={{ margin: 24 }}>Loading…</p>;

  return (
    <div className="rag-insights">
      <div className="rag-sources__header" style={{ margin: "0 0 16px" }}>
        <div>
          <h2>Connectors</h2>
          <p className="rag-hint" style={{ margin: "4px 0 0" }}>Redis and Postgres connections available to this space's tools.</p>
        </div>
        <button className="rag-btn" onClick={() => setTypePicker(true)}>+ Add connector</button>
      </div>

      {connectors.length === 0 ? (
        <p className="rag-hint">No connectors yet — add a Redis or Postgres connection above.</p>
      ) : (
        <div className="rag-source-list">
          {connectors.map((c) => <ConnectorRow key={c.id} connector={c} onOpen={setDetailId} />)}
        </div>
      )}

      {typePicker && (
        <TypePickerModal
          onClose={() => setTypePicker(false)}
          onChoose={(kind) => { setTypePicker(false); setFormState({ mode: "add", kind }); }}
        />
      )}
      {formState && (
        <ConnectorForm
          spaceId={spaceId}
          mode={formState.mode}
          kind={formState.kind}
          connector={formState.connector}
          onClose={() => setFormState(null)}
          onSaved={() => { setFormState(null); refresh(); }}
        />
      )}
      {detail && !formState && !deleteId && (
        <ConnectorDetailModal
          connector={detail}
          onClose={() => setDetailId(null)}
          onReplace={(connector) => setFormState({ mode: "replace", kind: connector.kind, connector })}
          onDelete={setDeleteId}
          onRetest={retest}
          retesting={retestingId === detail.id}
        />
      )}
      {deleteId && (
        <ConfirmDialog
          title="Remove this connector?"
          message="Any tools configured to use it will stop working."
          onCancel={() => setDeleteId(null)}
          onConfirm={confirmDelete}
        />
      )}
    </div>
  );
}
