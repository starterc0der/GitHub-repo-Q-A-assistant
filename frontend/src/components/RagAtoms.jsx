import { useState } from "react";

export function cls(...parts) {
  return parts.filter(Boolean).join(" ");
}

export const AVATAR_BG = { accent: "var(--accent-soft)", accent2: "var(--accent2-soft)", warn: "var(--warn-soft)", doc: "var(--doc-soft)" };
export const AVATAR_INK = { accent: "var(--accent-ink)", accent2: "var(--accent2-ink)", warn: "var(--warn-ink)", doc: "var(--doc-ink)" };

export function timeAgo(iso) {
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function RagTag({ children, tone }) {
  return <span className={cls("rag-tag", tone && `rag-tag--${tone}`)}>{children}</span>;
}

export function RagScorePill({ label, value, tone }) {
  const display = typeof value === "number" ? value.toFixed(3) : value;
  return (
    <span className={cls("rag-score", tone && `rag-score--${tone}`)}>
      <span className="rag-score__label">{label}</span>
      <span className="rag-score__value">{display}</span>
    </span>
  );
}

export function RagEmbedding({ embedding, label }) {
  if (!embedding) return null;
  return (
    <div className="rag-embedding">
      {label && <span className="rag-embedding__label">{label}</span>}
      <span className="rag-embedding__meta">
        {embedding.dim}d · norm {embedding.norm}
      </span>
      <span className="rag-embedding__vals">
        [{embedding.preview.map((v) => v.toFixed(3)).join(", ")}, …]
      </span>
    </div>
  );
}

export function RagCode({ code, startLine }) {
  const lines = code.split("\n");
  return (
    <pre className="rag-code">
      {lines.map((line, i) => (
        <span className="rag-code__line" key={i}>
          <span className="rag-code__ln">{startLine != null ? startLine + i : ""}</span>
          <span className="rag-code__src">{line || " "}</span>
        </span>
      ))}
    </pre>
  );
}

// mode: "full" (chunk + code, default), "context" (context header emphasized, code collapsed),
// "embedOnly" (compact row: location + embedding, no code)
export function RagChunkCard({ chunk, embedding, scores, badge, mode = "full", defaultOpen }) {
  const [open, setOpen] = useState(!!defaultOpen);

  if (mode === "embedOnly") {
    return (
      <div className="rag-chunk-row">
        <span className="rag-chunk-row__loc">
          {chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}
        </span>
        {chunk.symbol_name && <span className="rag-chunk-row__sym">{chunk.symbol_name}</span>}
        <RagEmbedding embedding={embedding} />
      </div>
    );
  }

  return (
    <div className="rag-chunk">
      <button className="rag-chunk__head" onClick={() => setOpen(!open)}>
        <span className="rag-chunk__loc">
          {chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}
        </span>
        {chunk.symbol_name && <span className="rag-chunk__sym">{chunk.symbol_name}()</span>}
        {badge}
        <span className="rag-chunk__toggle">{open ? "−" : "+"}</span>
      </button>
      {scores && (
        <div className="rag-chunk__scores">
          {scores.map((s) => (
            <RagScorePill key={s[0]} label={s[0]} value={s[1]} tone={s[2]} />
          ))}
        </div>
      )}
      {mode === "context" && chunk.context_header && (
        <p className="rag-chunk__context">&ldquo;{chunk.context_header}&rdquo;</p>
      )}
      {open && (
        <div className="rag-chunk__body">
          {mode === "full" && chunk.context_header && (
            <p className="rag-chunk__context">&ldquo;{chunk.context_header}&rdquo;</p>
          )}
          <RagCode code={chunk.code} startLine={chunk.start_line} />
          {embedding && <RagEmbedding embedding={embedding} />}
        </div>
      )}
    </div>
  );
}

// status: "pending" | "active" | "done"
export function RagStageStep({ num, title, meta, status, current, onClick, isLast }) {
  return (
    <button
      className={cls("rag-step", current && "rag-step--current", `rag-step--${status}`)}
      onClick={onClick}
    >
      <span className="rag-step__rail">
        <span className="rag-step__dot">{status === "done" ? "✓" : num}</span>
        {!isLast && <span className="rag-step__line" />}
      </span>
      <span className="rag-step__body">
        <span className="rag-step__title">{title}</span>
        {meta && <span className="rag-step__meta">{meta}</span>}
      </span>
    </button>
  );
}

export function RagSection({ id, num, title, description, plain, children }) {
  return (
    <section className="rag-section" id={id} data-stage-id={id}>
      <div className="rag-section__head">
        <span className="rag-section__num">{num}</span>
        <div>
          <h2 className="rag-section__title">{title}</h2>
          {description && <p className="rag-section__desc">{description}</p>}
        </div>
      </div>
      {plain && (
        <div className="rag-plain">
          <span className="rag-plain__tag">in plain English</span>
          <p>{plain}</p>
        </div>
      )}
      <div className="rag-section__body">{children}</div>
    </section>
  );
}
