// Vector-space scatter visualization shared by ingest (stage 5) and retrieval (stages 1-5).
// Points come from the backend's PCA projection of real embeddings (src/trace.py's
// project_3d) — this renders relative distance in that projection, not literal
// coordinates, and the raw values are normalized to a unit cube below.
//
// Three components rather than two: on real ingests PC3 carries ~82% of PC2's variance,
// so points that sit on top of each other in a flat plot are often genuinely far apart.
// The rotation is the point — a static 3D scatter reads worse than 2D, so drag to orbit.

import { useRef, useState } from "react";

export const TONE_COLOR = {
  dim: "var(--border-strong)",
  neutral: "var(--ink-faint)",
  accent: "var(--accent)",
  accent2: "var(--accent2)",
  warn: "var(--warn)",
};

const VIEW_W = 100;
const VIEW_H = 60;
const CX = VIEW_W / 2;
const CY = VIEW_H / 2;
const SPREAD = 17; // half-width of the unit cube once on screen
// Perspective strength. At 4+ the near/far size ratio is only ~1.6x, which on a roughly
// symmetric point cloud is invisible — the plot just looks like a flat blob. 2.4 gives
// ~2.4x, enough for depth to read at a glance without the far side collapsing to nothing.
const FOCAL = 2.4;
const PITCH_LIMIT = 1.25; // stop just short of poles, where yaw becomes meaningless

// The 12 edges of the unit cube, as index pairs into CUBE_CORNERS. Drawing the bounding
// box is what actually sells the third dimension: a cloud of dots can be rotated without
// the eye registering it, but a box visibly turns.
const CUBE_CORNERS = [
  [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
];
const CUBE_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

function extent(values) {
  if (!values.length) return [0, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  return min === max ? [min - 1, max + 1] : [min, max];
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Rescales raw PCA coordinates (arbitrary range) into a centered unit cube. Centering
// matters more in 3D than it did in 2D: rotation happens about the origin, so an
// off-center cloud would swing around rather than spin in place.
export function normalizePoints(rawPoints, rawQuery) {
  const axisScaler = (key, queryIndex) => {
    const values = rawPoints
      .map((p) => p[key] ?? 0)
      .concat(rawQuery ? [rawQuery[queryIndex] ?? 0] : []);
    const [lo, hi] = extent(values);
    return (v) => ((v - lo) / (hi - lo)) * 2 - 1;
  };
  const sx = axisScaler("x", 0);
  const sy = axisScaler("y", 1);
  const sz = axisScaler("z", 2);
  return {
    points: rawPoints.map((p) => ({ ...p, x: sx(p.x), y: sy(p.y), z: sz(p.z ?? 0) })),
    query: rawQuery ? [sx(rawQuery[0]), sy(rawQuery[1]), sz(rawQuery[2] ?? 0)] : null,
  };
}

// Yaw about the vertical axis, then pitch about the horizontal one, then a weak
// perspective divide. Two rotations are enough — roll adds no information here and
// only makes the cloud harder to track while dragging.
function makeProjector(yaw, pitch) {
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  return (x, y, z) => {
    const rx = x * cosYaw + z * sinYaw;
    const rz1 = z * cosYaw - x * sinYaw;
    const ry = y * cosPitch - rz1 * sinPitch;
    const rz = y * sinPitch + rz1 * cosPitch;
    const k = FOCAL / (FOCAL + rz); // >1 nearer the viewer, <1 further away
    return { x: CX + rx * SPREAD * k, y: CY + ry * SPREAD * k, depth: rz, k };
  };
}

// Orbit state, projection and drag gesture in one place, so the fullscreen modal renders
// the same space the same way instead of keeping its own copy of the projection math —
// the last time those two diverged, the modal silently plotted unit-cube coordinates
// straight into the viewBox and piled every point into a corner.
export function useOrbit(initial = { yaw: 0.62, pitch: 0.3 }) {
  const [rot, setRot] = useState(initial);
  const dragRef = useRef(null);

  const project = makeProjector(rot.yaw, rot.pitch);
  const cube = CUBE_CORNERS.map(([x, y, z]) => project(x, y, z));

  const handlers = {
    onPointerDown(e) {
      dragRef.current = { px: e.clientX, py: e.clientY, ...rot };
      e.currentTarget.setPointerCapture?.(e.pointerId);
    },
    onPointerMove(e) {
      const d = dragRef.current;
      if (!d) return;
      setRot({
        yaw: d.yaw + (e.clientX - d.px) * 0.011,
        pitch: clamp(d.pitch + (e.clientY - d.py) * 0.011, -PITCH_LIMIT, PITCH_LIMIT),
      });
    },
    onPointerUp(e) {
      dragRef.current = null;
      e.currentTarget.releasePointerCapture?.(e.pointerId);
    },
    onPointerLeave(e) {
      dragRef.current = null;
      e.currentTarget.releasePointerCapture?.(e.pointerId);
    },
  };

  return { project, cube, handlers };
}

// Wireframe bounding box. Drawn behind the points, fading with depth — a point cloud can
// rotate without the eye registering it; a box visibly turns.
export function CubeFrame({ cube }) {
  return (
    <g className="rag-vcube">
      {CUBE_EDGES.map(([a, b]) => {
        const near = (cube[a].k + cube[b].k) / 2;
        return (
          <line
            key={`cube-${a}-${b}`}
            x1={cube[a].x}
            y1={cube[a].y}
            x2={cube[b].x}
            y2={cube[b].y}
            stroke="var(--border-strong)"
            strokeWidth={0.14 * near}
            opacity={clamp(near - 0.45, 0.08, 0.4)}
          />
        );
      })}
    </g>
  );
}

export function RagVectorSpace({ points, query, legend, caption, height }) {
  // Native SVG <title> tooltips are unreliable (long hover delay, inconsistent across
  // browsers) and these dots are tiny — track hover explicitly and render the label as
  // real SVG text instead, via a larger invisible hit-area so small dots stay easy to hover.
  const [hoveredId, setHoveredId] = useState(null);
  // A slight default tilt so the plot reads as 3D before anyone touches it; head-on it
  // would be indistinguishable from the old flat version.
  const { project, cube, handlers } = useOrbit();

  // Painter's algorithm: furthest first, so nearer points overlap them rather than the
  // other way round. Without this the depth cues fight the draw order and read as noise.
  const projected = points
    .map((p) => ({ ...p, proj: project(p.x, p.y, p.z) }))
    .sort((a, b) => b.proj.depth - a.proj.depth);

  const q = query ? project(query[0], query[1], query[2]) : null;

  return (
    <div className="rag-vspace">
      <svg
        className="rag-vspace__svg rag-vspace__svg--3d"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        style={{ height: height || 340 }}
        preserveAspectRatio="xMidYMid meet"
        {...handlers}
      >
        <CubeFrame cube={cube} />
        {q &&
          projected
            .filter((p) => p.line)
            .map((p, i) => (
              <line
                key={"l-" + p.id}
                className="rag-vline-in"
                style={{ animationDelay: `${i * 45}ms` }}
                x1={q.x}
                y1={q.y}
                x2={p.proj.x}
                y2={p.proj.y}
                stroke={TONE_COLOR[p.tone] || TONE_COLOR.neutral}
                strokeWidth={(0.25 + (p.weight || 0.3) * 0.55) * p.proj.k}
                opacity={(0.25 + (p.weight || 0.3) * 0.55) * clamp(p.proj.k, 0.45, 1.15)}
              />
            ))}
        {projected.map((p, i) => {
          const isHovered = p.id === hoveredId;
          // Radius and opacity both track the perspective factor, so depth stays legible
          // even when the cloud is viewed near head-on.
          const r = (p.r || 1.0) * clamp(p.proj.k, 0.45, 1.55);
          const baseOpacity = p.tone === "dim" ? 0.4 : 0.9;
          return (
            <g
              key={p.id}
              className="rag-vpoint rag-vpoint-in"
              style={{ animationDelay: `${Math.min(i * 12, 420)}ms` }}
            >
              {!p.hideDot && (
                <>
                  <circle
                    cx={p.proj.x}
                    cy={p.proj.y}
                    r={isHovered ? r + 0.6 : r}
                    fill={TONE_COLOR[p.tone] || TONE_COLOR.neutral}
                    stroke="var(--panel)"
                    strokeWidth={0.09}
                    opacity={isHovered ? 1 : baseOpacity * clamp(p.proj.k * 0.85, 0.32, 1)}
                  />
                  {/* Invisible, larger hit-area — the visible dot alone is too small to
                      reliably hover/tap, especially at r ~1-1.5 units in a 100-unit viewBox. */}
                  <circle
                    cx={p.proj.x}
                    cy={p.proj.y}
                    r={r + 2.2}
                    fill="transparent"
                    onMouseEnter={() => setHoveredId(p.id)}
                    onMouseLeave={() => setHoveredId((id) => (id === p.id ? null : id))}
                  />
                </>
              )}
              {(p.showLabel || isHovered) && (
                <text
                  x={p.proj.x}
                  y={p.proj.y - r - 1.4}
                  textAnchor="middle"
                  className={"rag-vspace__label" + (isHovered ? " rag-vspace__label--hover" : "")}
                >
                  {p.label}
                </text>
              )}
            </g>
          );
        })}
        {q && (
          <g className="rag-vquery">
            <circle cx={q.x} cy={q.y} r={4.2 * clamp(q.k, 0.6, 1.4)} className="rag-vquery__ring" />
            <rect
              x={q.x - 1.6}
              y={q.y - 1.6}
              width="3.2"
              height="3.2"
              transform={`rotate(45 ${q.x} ${q.y})`}
              fill="var(--accent)"
            >
              <title>query embedding</title>
            </rect>
            <text x={q.x} y={q.y + 7.5} textAnchor="middle" className="rag-vspace__querylabel">
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
            <span className="rag-vspace__legend-item rag-vspace__hint">drag to rotate</span>
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
  // backend covers every id in chunk_xyz, so this should never actually trigger.
  return id.split("::")[0];
}

// Cached ingest traces are a persisted on-disk format, and a trace written before the
// 2D->3D change (src/trace.py's project_2d -> project_3d) carries `xy`, not `xyz`. Read
// either shape and default the missing axis to the mid-plane, rather than letting one
// stale cache file throw and take the whole view down.
function coords(point) {
  const v = point.xyz || point.xy;
  if (!v) return [0, 0, 0];
  return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];
}

function labelFor(retrievalData, id) {
  return retrievalData.chunk_labels[id] || chunkLabelFromId(id);
}

export function pointsForIngestChunks(ingestData) {
  const raw = [];
  ingestData.files.forEach((f) => {
    f.chunks.forEach((c) => {
      const [x, y, z] = coords(c);
      raw.push({
        id: c.chunk.id,
        x,
        y,
        z,
        r: 0.9,
        tone: "neutral",
        label: `${f.file_path} · ${c.chunk.symbol_name || "block"}`,
      });
    });
  });
  return normalizePoints(raw, null);
}

export function pointsForFileSpace(retrievalData) {
  const raw = Object.entries(retrievalData.file_xyz).map(([fp, xyz]) => ({
    id: fp,
    x: xyz?.[0] ?? 0,
    y: xyz?.[1] ?? 0,
    z: xyz?.[2] ?? 0,
    r: 1.15,
    tone: "neutral",
    label: fp.split("/").pop(),
    showLabel: true,
  }));
  return normalizePoints(raw, retrievalData.query_file_xyz);
}

export function pointsForRoute(retrievalData) {
  const routedMap = new Map(retrievalData.routed_files.map((f) => [f.file_path, f.score]));
  const raw = Object.entries(retrievalData.file_xyz).map(([fp, xyz]) => {
    const routed = routedMap.has(fp);
    return {
      id: fp,
      x: xyz?.[0] ?? 0,
      y: xyz?.[1] ?? 0,
      z: xyz?.[2] ?? 0,
      label: fp,
      showLabel: routed,
      tone: routed ? "accent2" : "dim",
      r: routed ? 1.45 : 0.85,
      line: routed,
      weight: routed ? routedMap.get(fp) : 0,
    };
  });
  return normalizePoints(raw, retrievalData.query_file_xyz);
}

// These three plot the whole-repo chunk space (same space + same query position the
// "visualize whole vector space" modal uses) rather than a routed-only projection — so
// the query marker lands in the same spot here as it does in the modal, and non-routed
// chunks show up dim in the background instead of being invisible.
export function pointsForHybrid(retrievalData) {
  const candMap = new Map(retrievalData.candidates.map((c) => [c.chunk.id, c.fused_score]));
  const raw = Object.entries(retrievalData.whole_chunk_xyz).map(([id, xyz]) => {
    const isCand = candMap.has(id);
    return {
      id,
      x: xyz?.[0] ?? 0,
      y: xyz?.[1] ?? 0,
      z: xyz?.[2] ?? 0,
      label: labelFor(retrievalData, id),
      tone: isCand ? "accent" : "dim",
      r: isCand ? 1.25 : 0.75,
      line: isCand,
      weight: isCand ? candMap.get(id) : 0,
    };
  });
  return normalizePoints(raw, retrievalData.query_whole_chunk_xyz);
}

export function pointsForRerank(retrievalData) {
  const rerankMap = new Map(retrievalData.reranked.map((r) => [r.chunk.id, r.rerank_score]));
  const scores = retrievalData.reranked.map((r) => r.rerank_score);
  const maxScore = scores.length ? Math.max(...scores) : 1;
  const raw = Object.entries(retrievalData.whole_chunk_xyz).map(([id, xyz]) => {
    const isTop = rerankMap.has(id);
    const norm = isTop ? Math.max(0.2, rerankMap.get(id) / (maxScore || 1)) : 0;
    return {
      id,
      x: xyz?.[0] ?? 0,
      y: xyz?.[1] ?? 0,
      z: xyz?.[2] ?? 0,
      label: labelFor(retrievalData, id),
      tone: isTop ? "accent2" : "dim",
      r: isTop ? 0.95 + norm * 0.85 : 0.65,
      line: isTop,
      weight: norm,
    };
  });
  return normalizePoints(raw, retrievalData.query_whole_chunk_xyz);
}

export function pointsForCompress(retrievalData) {
  const kept = new Set();
  const dropped = new Set();
  retrievalData.final_chunks.forEach((f) => {
    (f.dropped ? dropped : kept).add(f.chunk.id);
  });
  const raw = Object.entries(retrievalData.whole_chunk_xyz).map(([id, xyz]) => {
    const tone = kept.has(id) ? "accent2" : dropped.has(id) ? "warn" : "dim";
    return {
      id,
      x: xyz?.[0] ?? 0,
      y: xyz?.[1] ?? 0,
      z: xyz?.[2] ?? 0,
      label: labelFor(retrievalData, id),
      tone,
      r: kept.has(id) ? 1.4 : dropped.has(id) ? 1.0 : 0.6,
      line: kept.has(id),
    };
  });
  return normalizePoints(raw, retrievalData.query_whole_chunk_xyz);
}
