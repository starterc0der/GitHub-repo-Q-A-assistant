// Vector-space scatter visualization shared by ingest (stage 5) and retrieval (stages 1-5).
// Points come from the backend's PCA projection of real embeddings (src/trace.py's
// project_2d) — this renders relative distance in that projection, not literal
// coordinates, and the raw values are normalized to fit the viewBox below.

import { useState } from "react";

export const TONE_COLOR = {
  dim: "var(--border-strong)",
  neutral: "var(--ink-faint)",
  accent: "var(--accent)",
  accent2: "var(--accent2)",
  warn: "var(--warn)",
};

const VIEW_W = 100;
const VIEW_H = 60;
const PAD = 7;

function extent(values) {
  if (!values.length) return [0, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  return min === max ? [min - 1, max + 1] : [min, max];
}

function scale(v, inMin, inMax, outMin, outMax) {
  return outMin + ((v - inMin) / (inMax - inMin)) * (outMax - outMin);
}

// Rescales raw PCA coordinates (arbitrary range) into the SVG viewBox, since — unlike
// the design's hand-placed mock layout — real projected coordinates aren't pre-fit to
// any particular range.
export function normalizePoints(rawPoints, rawQuery) {
  const xs = rawPoints.map((p) => p.x).concat(rawQuery ? [rawQuery[0]] : []);
  const ys = rawPoints.map((p) => p.y).concat(rawQuery ? [rawQuery[1]] : []);
  const [xMin, xMax] = extent(xs);
  const [yMin, yMax] = extent(ys);
  const sx = (x) => scale(x, xMin, xMax, PAD, VIEW_W - PAD);
  const sy = (y) => scale(y, yMin, yMax, PAD, VIEW_H - PAD);
  return {
    points: rawPoints.map((p) => ({ ...p, x: sx(p.x), y: sy(p.y) })),
    query: rawQuery ? [sx(rawQuery[0]), sy(rawQuery[1])] : null,
  };
}

export function RagVectorSpace({ points, query, legend, caption, height }) {
  // Native SVG <title> tooltips are unreliable (long hover delay, inconsistent across
  // browsers) and these dots are tiny — track hover explicitly and render the label as
  // real SVG text instead, via a larger invisible hit-area so small dots stay easy to hover.
  const [hoveredId, setHoveredId] = useState(null);

  return (
    <div className="rag-vspace">
      <svg
        className="rag-vspace__svg"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        style={{ height: height || 340 }}
        preserveAspectRatio="xMidYMid meet"
      >
        {query &&
          points
            .filter((p) => p.line)
            .map((p, i) => (
              <line
                key={"l-" + p.id}
                className="rag-vline-in"
                style={{ animationDelay: `${i * 45}ms` }}
                x1={query[0]}
                y1={query[1]}
                x2={p.x}
                y2={p.y}
                stroke={TONE_COLOR[p.tone] || TONE_COLOR.neutral}
                strokeWidth={0.25 + (p.weight || 0.3) * 0.55}
                opacity={0.25 + (p.weight || 0.3) * 0.55}
              />
            ))}
        {points.map((p, i) => {
          const isHovered = p.id === hoveredId;
          const r = p.r || 1.6;
          return (
            <g
              key={p.id}
              className="rag-vpoint rag-vpoint-in"
              style={{ animationDelay: `${Math.min(i * 12, 420)}ms` }}
            >
              {!p.hideDot && (
                <>
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={isHovered ? r + 0.6 : r}
                    fill={TONE_COLOR[p.tone] || TONE_COLOR.neutral}
                    opacity={isHovered ? 1 : p.tone === "dim" ? 0.45 : 0.92}
                  />
                  {/* Invisible, larger hit-area — the visible dot alone is too small to
                      reliably hover/tap, especially at r ~1-1.5 units in a 100-unit viewBox. */}
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={r + 2.2}
                    fill="transparent"
                    onMouseEnter={() => setHoveredId(p.id)}
                    onMouseLeave={() => setHoveredId((id) => (id === p.id ? null : id))}
                  />
                </>
              )}
              {(p.showLabel || isHovered) && (
                <text
                  x={p.x}
                  y={p.y - r - 1.4}
                  textAnchor="middle"
                  className={"rag-vspace__label" + (isHovered ? " rag-vspace__label--hover" : "")}
                >
                  {p.label}
                </text>
              )}
            </g>
          );
        })}
        {query && (
          <g className="rag-vquery">
            <circle cx={query[0]} cy={query[1]} r="4.2" className="rag-vquery__ring" />
            <rect
              x={query[0] - 1.6}
              y={query[1] - 1.6}
              width="3.2"
              height="3.2"
              transform={`rotate(45 ${query[0]} ${query[1]})`}
              fill="var(--accent)"
            >
              <title>query embedding</title>
            </rect>
            <text x={query[0]} y={query[1] + 7.5} textAnchor="middle" className="rag-vspace__querylabel">
              your question
            </text>
          </g>
        )}
      </svg>
      <div className="rag-vspace__footer">
        {legend && (
          <div className="rag-vspace__legend">
            {legend.map((l) => (
              <span key={l.label} className="rag-vspace__legend-item">
                <span className="rag-vspace__swatch" style={{ background: TONE_COLOR[l.tone] }} />
                {l.label}
              </span>
            ))}
          </div>
        )}
        {caption && <p className="rag-vspace__caption">{caption}</p>}
      </div>
    </div>
  );
}

// ---- stage-specific point builders, operating on real trace API responses ----------

export function chunkLabelFromId(id) {
  // CodeChunk.id is "{file_path}::{start_line}-{end_line}" (src/index/schema.py) — a
  // defensive fallback if a chunk_id is ever missing from chunk_labels; in practice the
  // backend covers every id in chunk_xy, so this should never actually trigger.
  return id.split("::")[0];
}

function labelFor(retrievalData, id) {
  return retrievalData.chunk_labels[id] || chunkLabelFromId(id);
}

export function pointsForIngestChunks(ingestData) {
  const raw = [];
  ingestData.files.forEach((f) => {
    f.chunks.forEach((c) => {
      raw.push({
        id: c.chunk.id,
        x: c.xy[0],
        y: c.xy[1],
        r: 1.5,
        tone: "neutral",
        label: `${f.file_path} · ${c.chunk.symbol_name || "block"}`,
      });
    });
  });
  return normalizePoints(raw, null);
}

export function pointsForFileSpace(retrievalData) {
  const raw = Object.entries(retrievalData.file_xy).map(([fp, xy]) => ({
    id: fp,
    x: xy[0],
    y: xy[1],
    r: 1.9,
    tone: "neutral",
    label: fp.split("/").pop(),
    showLabel: true,
  }));
  return normalizePoints(raw, retrievalData.query_file_xy);
}

export function pointsForRoute(retrievalData) {
  const routedMap = new Map(retrievalData.routed_files.map((f) => [f.file_path, f.score]));
  const raw = Object.entries(retrievalData.file_xy).map(([fp, xy]) => {
    const routed = routedMap.has(fp);
    return {
      id: fp,
      x: xy[0],
      y: xy[1],
      label: fp,
      showLabel: routed,
      tone: routed ? "accent2" : "dim",
      r: routed ? 2.4 : 1.4,
      line: routed,
      weight: routed ? routedMap.get(fp) : 0,
    };
  });
  return normalizePoints(raw, retrievalData.query_file_xy);
}

// These three plot the whole-repo chunk space (same space + same query position the
// "visualize whole vector space" modal uses) rather than a routed-only projection — so
// the query marker lands in the same spot here as it does in the modal, and non-routed
// chunks show up dim in the background instead of being invisible.
export function pointsForHybrid(retrievalData) {
  const candMap = new Map(retrievalData.candidates.map((c) => [c.chunk.id, c.fused_score]));
  const raw = Object.entries(retrievalData.whole_chunk_xy).map(([id, xy]) => {
    const isCand = candMap.has(id);
    return {
      id,
      x: xy[0],
      y: xy[1],
      label: labelFor(retrievalData, id),
      tone: isCand ? "accent" : "dim",
      r: isCand ? 2.1 : 1.2,
      line: isCand,
      weight: isCand ? candMap.get(id) : 0,
    };
  });
  return normalizePoints(raw, retrievalData.query_whole_chunk_xy);
}

export function pointsForRerank(retrievalData) {
  const rerankMap = new Map(retrievalData.reranked.map((r) => [r.chunk.id, r.rerank_score]));
  const scores = retrievalData.reranked.map((r) => r.rerank_score);
  const maxScore = scores.length ? Math.max(...scores) : 1;
  const raw = Object.entries(retrievalData.whole_chunk_xy).map(([id, xy]) => {
    const isTop = rerankMap.has(id);
    const norm = isTop ? Math.max(0.2, rerankMap.get(id) / (maxScore || 1)) : 0;
    return {
      id,
      x: xy[0],
      y: xy[1],
      label: labelFor(retrievalData, id),
      tone: isTop ? "accent2" : "dim",
      r: isTop ? 1.6 + norm * 1.4 : 1.1,
      line: isTop,
      weight: norm,
    };
  });
  return normalizePoints(raw, retrievalData.query_whole_chunk_xy);
}

export function pointsForCompress(retrievalData) {
  const kept = new Set();
  const dropped = new Set();
  retrievalData.final_chunks.forEach((f) => {
    (f.dropped ? dropped : kept).add(f.chunk.id);
  });
  const raw = Object.entries(retrievalData.whole_chunk_xy).map(([id, xy]) => {
    const tone = kept.has(id) ? "accent2" : dropped.has(id) ? "warn" : "dim";
    return {
      id,
      x: xy[0],
      y: xy[1],
      label: labelFor(retrievalData, id),
      tone,
      r: kept.has(id) ? 2.3 : dropped.has(id) ? 1.7 : 1,
      line: kept.has(id),
    };
  });
  return normalizePoints(raw, retrievalData.query_whole_chunk_xy);
}
