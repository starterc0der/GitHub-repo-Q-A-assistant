// Fullscreen "whole vector space" walkthrough: one persistent scatter of every chunk in
// the repo (src/trace.py's whole_chunk_xy — a PCA space separate from the per-stage plots
// in RagVectorSpace, since it covers the full repo rather than just the routed-file pool)
// that re-tints/re-sizes/re-connects via CSS transitions as you step through the stages.
//
// Ends at compression, not "answer" — this tool never calls an LLM, so there's no
// generated answer/citation to point the last step at.

import { useEffect, useMemo, useState } from "react";
import { cls } from "./RagAtoms.jsx";
import { TONE_COLOR, chunkLabelFromId, normalizePoints } from "./RagVectorSpace.jsx";

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

function pointState(step, point, ctx) {
  const routed = ctx.routedMap.has(point.filePath);
  const isCand = ctx.candMap.has(point.id);
  const isTop = ctx.rerankMap.has(point.id);
  const base = 1.5;

  if (step <= 1) return { tone: "neutral", r: base, line: false, weight: 0 };

  if (step === 2) {
    return routed
      ? { tone: "accent2", r: base + 0.7, line: true, weight: ctx.routedMap.get(point.filePath) }
      : { tone: "dim", r: base * 0.8, line: false, weight: 0 };
  }

  if (step === 3) {
    if (isCand) {
      return { tone: "accent", r: base + 0.7, line: true, weight: ctx.candMap.get(point.id) };
    }
    return routed
      ? { tone: "neutral", r: base, line: false, weight: 0 }
      : { tone: "dim", r: base * 0.7, line: false, weight: 0 };
  }

  if (step === 4) {
    if (isTop) {
      const norm = ctx.rerankMap.get(point.id) / (ctx.maxRerank || 1);
      return { tone: "accent2", r: base + 0.4 + norm * 1.3, line: true, weight: Math.max(0.25, norm) };
    }
    return isCand
      ? { tone: "neutral", r: base, line: false, weight: 0 }
      : { tone: "dim", r: base * 0.7, line: false, weight: 0 };
  }

  // step 5: compression — the last step
  if (ctx.kept.has(point.id)) return { tone: "accent2", r: base + 1, line: true, weight: 0.85 };
  if (ctx.dropped.has(point.id)) return { tone: "warn", r: base + 0.3, line: false, weight: 0 };
  return { tone: "dim", r: base * 0.7, line: false, weight: 0 };
}

export function RagVectorModal({ data, onClose }) {
  const { points, query } = useMemo(() => {
    const raw = Object.entries(data.whole_chunk_xy).map(([id, xy]) => ({
      id,
      x: xy[0],
      y: xy[1],
      filePath: chunkLabelFromId(id),
      label: data.chunk_labels[id] || chunkLabelFromId(id),
    }));
    return normalizePoints(raw, data.query_whole_chunk_xy);
  }, [data]);

  const ctx = useMemo(() => buildContext(data), [data]);

  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

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
        <svg className="rag-modal__svg" viewBox="0 0 100 60" preserveAspectRatio="xMidYMid meet">
          {showQuery &&
            points
              .map((p) => ({ p, s: pointState(step, p, ctx) }))
              .filter(({ s }) => s.line)
              .map(({ p, s }) => (
                <line
                  key={"l-" + p.id}
                  x1={query[0]}
                  y1={query[1]}
                  x2={p.x}
                  y2={p.y}
                  stroke={TONE_COLOR[s.tone]}
                  strokeWidth={0.2 + s.weight * 0.5}
                  opacity={0.2 + s.weight * 0.55}
                />
              ))}
          {points.map((p) => {
            const s = pointState(step, p, ctx);
            return (
              <circle
                key={p.id}
                cx={p.x}
                cy={p.y}
                r={s.r}
                fill={TONE_COLOR[s.tone]}
                opacity={s.tone === "dim" ? 0.35 : 0.92}
              >
                <title>{p.label}</title>
              </circle>
            );
          })}
          {showQuery && (
            <g className="rag-vquery">
              <circle cx={query[0]} cy={query[1]} r="4.2" className="rag-vquery__ring" />
              <rect
                x={query[0] - 1.7}
                y={query[1] - 1.7}
                width="3.4"
                height="3.4"
                transform={`rotate(45 ${query[0]} ${query[1]})`}
                fill="var(--accent)"
              />
              <text x={query[0]} y={query[1] + 8} textAnchor="middle" className="rag-vspace__querylabel">
                your question
              </text>
            </g>
          )}
        </svg>
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
