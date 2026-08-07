// Fullscreen "whole vector space" walkthrough: one persistent scatter of every chunk in
// the repo (src/trace.py's whole_chunk_xyz — a PCA space separate from the per-stage plots
// in RagVectorSpace, since it covers the full repo rather than just the routed-file pool)
// that re-tints/re-sizes/re-connects via CSS transitions as you step through the stages.
//
// Ends at compression, not "answer" — this tool never calls an LLM, so there's no
// generated answer/citation to point the last step at.

import { useEffect, useMemo, useState } from "react";
import { cls } from "./RagAtoms.jsx";
import { CubeFrame, TONE_COLOR, chunkLabelFromId, normalizePoints, useOrbit } from "./RagVectorSpace.jsx";

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const MODAL_STEPS = [
  {
    title: "Every chunk, embedded",
    caption:
      "Each dot is one chunk's embedding. Position reflects meaning — chunks about the same topic land near each other, whether or not they're in the same file.",
  },
  {
    title: "The question embeds in",
    caption:
      "The question drops into the exact same space, as the diamond marker — not a special case, just another embedding.",
  },
  {
    title: "Routing narrows to files",
    caption:
      "Only chunks belonging to the shortlisted files stay lit. Everything else is set aside before any chunk-level search runs.",
  },
  {
    title: "Hybrid search finds candidates",
    caption:
      "Dense cosine similarity + BM25 keyword overlap, fused together, pick the closest candidates out of the routed files.",
  },
  {
    title: "Cross-encoder reranks",
    caption:
      "A slower, more careful pass reads the question and each candidate together, and can reorder — relevance isn't always raw distance.",
  },
  {
    title: "Compression keeps the survivors",
    caption:
      "Only these chunks — trimmed to their relevant lines — continue on into the final prompt. Everything else, including near-misses, is dropped.",
  },
];

const LAST_STEP = MODAL_STEPS.length - 1;

function buildContext(data) {
  const routedMap = new Map(data.routed_files.map((f) => [f.file_path, f.score]));
  const candMap = new Map(data.candidates.map((c) => [c.chunk.id, c.fused_score]));
  const rerankMap = new Map(data.reranked.map((r) => [r.chunk.id, r.rerank_score]));
  const rerankScores = data.reranked.map((r) => r.rerank_score);
  const maxRerank = rerankScores.length ? Math.max(...rerankScores) : 1;
  const kept = new Set();
  const dropped = new Set();
  data.final_chunks.forEach((f) => {
    (f.dropped ? dropped : kept).add(f.chunk.id);
  });
  return { routedMap, candMap, rerankMap, maxRerank, kept, dropped };
}

// Radii are in viewBox units, and the modal renders that 100-unit box across ~1600px —
// so 1 unit is ~16px. The old base of 1.5 drew 45px circles that merged into a blob at
// 179 points; these keep the same *relative* sizing (which encodes score) at a third the
// area, so the cloud reads as individual chunks again.
function pointState(step, point, ctx) {
  const routed = ctx.routedMap.has(point.filePath);
  const isCand = ctx.candMap.has(point.id);
  const isTop = ctx.rerankMap.has(point.id);
  const base = 0.62;

  if (step <= 1) return { tone: "neutral", r: base, line: false, weight: 0 };

  if (step === 2) {
    return routed
      ? { tone: "accent2", r: base + 0.3, line: true, weight: ctx.routedMap.get(point.filePath) }
      : { tone: "dim", r: base * 0.78, line: false, weight: 0 };
  }

  if (step === 3) {
    if (isCand) {
      return { tone: "accent", r: base + 0.3, line: true, weight: ctx.candMap.get(point.id) };
    }
    return routed
      ? { tone: "neutral", r: base, line: false, weight: 0 }
      : { tone: "dim", r: base * 0.7, line: false, weight: 0 };
  }

  if (step === 4) {
    if (isTop) {
      const norm = ctx.rerankMap.get(point.id) / (ctx.maxRerank || 1);
      return { tone: "accent2", r: base + 0.18 + norm * 0.55, line: true, weight: Math.max(0.25, norm) };
    }
    return isCand
      ? { tone: "neutral", r: base, line: false, weight: 0 }
      : { tone: "dim", r: base * 0.7, line: false, weight: 0 };
  }

  // step 5: compression — the last step
  if (ctx.kept.has(point.id)) return { tone: "accent2", r: base + 0.42, line: true, weight: 0.85 };
  if (ctx.dropped.has(point.id)) return { tone: "warn", r: base + 0.13, line: false, weight: 0 };
  return { tone: "dim", r: base * 0.7, line: false, weight: 0 };
}

export function RagVectorModal({ data, onClose }) {
  const { points, query } = useMemo(() => {
    const raw = Object.entries(data.whole_chunk_xyz).map(([id, xyz]) => ({
      id,
      x: xyz?.[0] ?? 0,
      y: xyz?.[1] ?? 0,
      z: xyz?.[2] ?? 0,
      filePath: chunkLabelFromId(id),
      label: data.chunk_labels[id] || chunkLabelFromId(id),
    }));
    return normalizePoints(raw, data.query_whole_chunk_xyz);
  }, [data]);

  const ctx = useMemo(() => buildContext(data), [data]);

  const { project, cube, handlers } = useOrbit();
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  // Project + depth-sort once per render: state depends on `step`, so this can't be
  // memoized on `points` alone. Painter's algorithm — furthest drawn first.
  const projected = points
    .map((p) => ({ p: { ...p, proj: project(p.x, p.y, p.z) }, s: pointState(step, p, ctx) }))
    .sort((a, b) => b.p.proj.depth - a.p.proj.depth);
  const q = project(query[0], query[1], query[2]);

  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") setStep((s) => Math.min(LAST_STEP, s + 1));
      else if (e.key === "ArrowLeft") setStep((s) => Math.max(0, s - 1));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    if (!playing) return;
    if (step >= LAST_STEP) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setStep((s) => Math.min(LAST_STEP, s + 1)), 2200);
    return () => clearTimeout(t);
  }, [playing, step]);

  const showQuery = step >= 1;

  return (
    <div className="rag-modal" role="dialog" aria-modal="true">
      <div className="rag-modal__head">
        <div>
          <span className="rag-modal__eyebrow">
            Whole vector space · step {step + 1} of {MODAL_STEPS.length}
          </span>
          <h2>{MODAL_STEPS[step].title}</h2>
        </div>
        <button className="rag-modal__close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>

      <div className="rag-modal__stage">
        <svg
          className="rag-modal__svg rag-vspace__svg--3d"
          viewBox="0 0 100 60"
          preserveAspectRatio="xMidYMid meet"
          {...handlers}
        >
          <CubeFrame cube={cube} />
          {showQuery &&
            projected
              .filter(({ s }) => s.line)
              .map(({ p, s }) => (
                <line
                  key={"l-" + p.id}
                  x1={q.x}
                  y1={q.y}
                  x2={p.proj.x}
                  y2={p.proj.y}
                  stroke={TONE_COLOR[s.tone]}
                  strokeWidth={(0.2 + s.weight * 0.5) * p.proj.k}
                  opacity={(0.2 + s.weight * 0.55) * clamp(p.proj.k, 0.45, 1.15)}
                />
              ))}
          {projected.map(({ p, s }) => (
            <circle
              key={p.id}
              cx={p.proj.x}
              cy={p.proj.y}
              r={s.r * clamp(p.proj.k, 0.45, 1.55)}
              fill={TONE_COLOR[s.tone]}
              stroke="var(--panel)"
              strokeWidth={0.09}
              opacity={(s.tone === "dim" ? 0.3 : 0.85) * clamp(p.proj.k * 0.85, 0.32, 1)}
            >
              <title>{p.label}</title>
            </circle>
          ))}
          {showQuery && (
            <g className="rag-vquery">
              <circle cx={q.x} cy={q.y} r={4.2 * clamp(q.k, 0.6, 1.4)} className="rag-vquery__ring" />
              <rect
                x={q.x - 1.7}
                y={q.y - 1.7}
                width="3.4"
                height="3.4"
                transform={`rotate(45 ${q.x} ${q.y})`}
                fill="var(--accent)"
              />
              <text x={q.x} y={q.y + 8} textAnchor="middle" className="rag-vspace__querylabel">
                your question
              </text>
            </g>
          )}
        </svg>
        <span className="rag-modal__hint">drag to rotate · ← → to step · Esc to close</span>
      </div>

      <div className="rag-modal__foot">
        <p className="rag-modal__caption">{MODAL_STEPS[step].caption}</p>
        <div className="rag-modal__controls">
          <button
            className="rag-btn rag-btn--ghost"
            onClick={() => setStep(Math.max(0, step - 1))}
            disabled={step === 0}
          >
            ← Back
          </button>
          <div className="rag-modal__dots">
            {MODAL_STEPS.map((s, i) => (
              <button
                key={i}
                className={cls("rag-modal__dot", i === step && "rag-modal__dot--active")}
                onClick={() => setStep(i)}
                aria-label={s.title}
              />
            ))}
          </div>
          <button
            className="rag-btn"
            onClick={() => {
              if (step >= LAST_STEP) {
                setStep(0);
                setPlaying(true);
              } else {
                setPlaying(!playing);
              }
            }}
          >
            {playing ? "Pause" : step >= LAST_STEP ? "Replay" : "Play"}
          </button>
          <button
            className="rag-btn rag-btn--ghost"
            onClick={() => setStep(Math.min(LAST_STEP, step + 1))}
            disabled={step === LAST_STEP}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
