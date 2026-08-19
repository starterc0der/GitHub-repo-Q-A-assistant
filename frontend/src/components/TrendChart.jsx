import { useId, useState } from "react";

// Same {title, categories, series} spec as BarChart (see src/generate/answer.py's
// ChartParser) but rendered as a gradient area/line — one fill+stroke per series, same
// "Token usage" chart style used in Insights — because a time trend (or a comparison of
// two places' trends) reads as a line, not discrete bars. Selected via chart.kind
// ("trend"), same fenced ```chart block either way.

// A multi-place comparison routinely has 5+ series (e.g. one per sub-place across two
// places) — 3 colors cycling meant the 4th series silently reused the 1st's color,
// making two different lines indistinguishable. 6 distinct hues, spaced around the
// wheel, still built from the existing design tokens where available.
const COLORS = [
  "var(--accent)", "var(--accent2)", "var(--warn)", "var(--doc)",
  "oklch(62% 0.16 300)", "oklch(55% 0.13 125)",
];
// A fixed 480 width squeezes wide date ranges into unreadably tight buckets and hides
// data behind hover alone — grow with the category count instead (capped so a short
// series still renders at a normal, non-stretched size), and let the container scroll
// horizontally rather than the SVG itself shrinking everything back down.
const MIN_WIDTH = 480;
const PX_PER_CATEGORY = 46;
const HEIGHT = 190;
const X0 = 12, X1_PAD = 12, Y_TOP = 14, Y_BOTTOM = 150;

function fmtValue(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return value;
  return (Math.round(value * 100) / 100).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function truncate(text, max) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

const ISO_DATE_RE = /^\d{4}-(\d{2})-(\d{2})$/;

// "2026-08-05" truncated at a fixed character count all collapse to the same "2026-08-"
// prefix (the actual distinguishing part, the day, is what gets cut) — show month-day
// for a recognized date instead of blindly chopping from the front.
function fmtCategory(cat) {
  const s = String(cat);
  const m = ISO_DATE_RE.exec(s);
  return m ? `${m[1]}-${m[2]}` : truncate(s, 8);
}

export function TrendChart({ chart }) {
  const { title, categories, series } = chart;
  const gradientBase = useId();
  const [hoverIdx, setHoverIdx] = useState(null);
  const n = categories.length;
  const WIDTH = Math.max(MIN_WIDTH, n * PX_PER_CATEGORY);
  const X1 = WIDTH - X1_PAD;

  const allValues = series.flatMap((s) => s.values);
  const dataMin = Math.min(...allValues);
  const dataMax = Math.max(...allValues);
  // A trend chart's whole point is showing how a value MOVES, not its distance from
  // zero — pressure/flow sitting in a tight band far from 0 (e.g. 2.7-2.85) needs the
  // axis scaled to that band, not stretched down to a 0 baseline that flattens every
  // real difference into an unreadable sliver near the top (see git history for the
  // actual screenshot this was reported against).
  const range = dataMax - dataMin;
  const pad = range > 0 ? range * 0.15 : Math.abs(dataMax || 1) * 0.1;
  const maxValue = dataMax + pad;
  const minValue = dataMin - pad;
  const span = maxValue - minValue || 1;

  const xAt = (i) => (n > 1 ? X0 + (i / (n - 1)) * (X1 - X0) : (X0 + X1) / 2);
  const yAt = (v) => Y_BOTTOM - ((v - minValue) / span) * (Y_BOTTOM - Y_TOP);
  const step = n > 1 ? (X1 - X0) / (n - 1) : X1 - X0;

  const plotted = series.map((s, si) => {
    const points = s.values.map((v, i) => ({ x: xAt(i), y: yAt(v) }));
    const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    const area = points.length
      ? `${line} L${points[n - 1].x.toFixed(1)},${Y_BOTTOM} L${points[0].x.toFixed(1)},${Y_BOTTOM} Z`
      : "";
    return { name: s.name, values: s.values, line, area, color: COLORS[si % COLORS.length], gradientId: `${gradientBase}-${si}` };
  });

  // Target one label roughly every 55px of actual plotted width, instead of a fixed
  // "show at most 6" cap that left most of a wide, scrollable chart unlabeled.
  const labelEvery = Math.max(1, Math.ceil(n / Math.max(1, Math.floor((X1 - X0) / 55))));

  return (
    <div className="rag-chart">
      {title && <div className="rag-chart__title">{title}</div>}
      <div className={`rag-chart-hover-wrap${WIDTH > MIN_WIDTH ? " rag-chart__hscroll" : ""}`}>
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          width={WIDTH}
          className="rag-chart__svg"
          onMouseLeave={() => setHoverIdx(null)}
        >
          <defs>
            {plotted.map((s) => (
              <linearGradient key={s.gradientId} id={s.gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity="0.28" />
                <stop offset="100%" stopColor={s.color} stopOpacity="0.02" />
              </linearGradient>
            ))}
          </defs>
          <line x1={X0} y1={Y_BOTTOM} x2={X1} y2={Y_BOTTOM} className="rag-chart__axis" />
          <line x1={X0} y1={Y_TOP} x2={X1} y2={Y_TOP} className="rag-chart__axis" />
          <text x={X0} y={Y_BOTTOM + 12} className="rag-chart__value">{fmtValue(minValue)}</text>
          <text x={X0} y={Y_TOP - 4} className="rag-chart__value">{fmtValue(maxValue)}</text>
          {plotted.map((s) => (
            <g key={s.name}>
              <path d={s.area} fill={`url(#${s.gradientId})`} />
              <path d={s.line} fill="none" stroke={s.color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </g>
          ))}
          {categories.map((cat, i) =>
            i % labelEvery === 0 || i === n - 1 ? (
              <text key={`${cat}-${i}`} x={xAt(i)} y={HEIGHT - 4} textAnchor="middle" className="rag-chart__cat">
                {fmtCategory(cat)}
              </text>
            ) : null
          )}
          {hoverIdx != null && (
            <line x1={xAt(hoverIdx)} y1={Y_TOP} x2={xAt(hoverIdx)} y2={Y_BOTTOM} className="rag-chart__guide" />
          )}
          {hoverIdx != null &&
            plotted.map((s) => (
              <circle
                key={s.name} cx={xAt(hoverIdx)} cy={yAt(s.values[hoverIdx])} r="4"
                fill={s.color} stroke="var(--panel)" strokeWidth="1.5"
              />
            ))}
          {categories.map((_cat, i) => (
            <rect
              key={i}
              x={Math.max(X0, xAt(i) - step / 2)} y={Y_TOP}
              width={Math.min(X1, xAt(i) + step / 2) - Math.max(X0, xAt(i) - step / 2)}
              height={Y_BOTTOM - Y_TOP}
              fill="transparent"
              onMouseEnter={() => setHoverIdx(i)}
            />
          ))}
        </svg>
        {hoverIdx != null && (
          <div className="rag-chart-tooltip" style={{ left: `${(xAt(hoverIdx) / WIDTH) * 100}%`, top: "6%" }}>
            <div className="rag-chart-tooltip__title">{categories[hoverIdx]}</div>
            {plotted.map((s) => (
              <div key={s.name} className="rag-chart-tooltip__row">
                <span className="rag-chart-tooltip__dot" style={{ background: s.color }} />
                {s.name} · {fmtValue(s.values[hoverIdx])}
              </div>
            ))}
          </div>
        )}
      </div>
      {series.length > 1 && (
        <div className="rag-chart__legend">
          {plotted.map((s) => (
            <span key={s.name} className="rag-chart__legend-item">
              <span className="rag-chart__swatch" style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
