import { useEffect, useId, useState } from "react";
import { chatInsights, chunkInsights, messageTrace, messageVectors, spaceInsights } from "../api.js";
import { cls } from "../components/RagAtoms.jsx";
import { ConnectorsPanel } from "../components/ConnectorsPanel.jsx";
import { PipelineOverlay } from "../components/PipelineOverlay.jsx";

// "partial": a decomposed question where some (not all) sub-questions found no
// evidence — a real answer was generated, but not a full success, so it gets its own
// color rather than being folded into "Answered". See Pipeline._retrieve.
const GATE_LABEL = { answered: "Answered", "no-match": "No match", "wide-fallback": "Wide fallback", partial: "Partial" };
const GATE_TONE = { answered: "accent2", "no-match": "warn", "wide-fallback": "accent", partial: "doc" };
const STAGE_LABEL = {
  cache: "Cache check", decompose: "Decomposition", embed: "Embed", route: "Route",
  hybrid: "Hybrid search", rerank: "Rerank", sufficiency_check: "Sufficiency check", gate: "Gate",
  retry: "Retry", compress: "Compression", generate: "Generation",
};

function fmtMs(ms) {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function fmtPct(x) {
  return `${Math.round((x || 0) * 100)}%`;
}

function fmtDateShort(iso) {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function avg(vals) {
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

// First-half vs second-half average, as a fraction change — same "which way is this
// trending" signal used for every KPI/gauge delta arrow on this page. Null means "no
// baseline to compare against" (first half averaged exactly 0, e.g. a quiet space with
// no questions yet) rather than a fake four-digit percentage — going from 0 to anything
// isn't expressible as a % change.
function trendDelta(vals) {
  if (vals.length < 2) return null;
  const half = Math.floor(vals.length / 2) || 1;
  const first = avg(vals.slice(0, half));
  const second = avg(vals.slice(-half));
  if (first === 0) return second === 0 ? 0 : null;
  return (second - first) / Math.abs(first);
}

function sparkPaths(vals, w, h, pad) {
  const series = vals.length >= 2 ? vals : [vals[0] ?? 0, vals[0] ?? 0];
  const max = Math.max(...series), min = Math.min(...series);
  const range = Math.max(max - min, 0.0001);
  const n = series.length;
  const xs = series.map((_, i) => pad + (i / (n - 1)) * (w - 2 * pad));
  const ys = series.map((v) => h - pad - ((v - min) / range) * (h - 2 * pad));
  const line = xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ");
  const area = `${line} L${xs[n - 1].toFixed(1)},${(h - pad).toFixed(1)} L${xs[0].toFixed(1)},${(h - pad).toFixed(1)} Z`;
  return { line, area };
}

function DeltaBadge({ delta }) {
  if (delta == null) return <span className="rag-delta rag-delta--flat">new</span>;
  const pct = Math.round(Math.abs(delta) * 100);
  const up = delta >= 0;
  return (
    <span className={cls("rag-delta", up ? "rag-delta--up" : "rag-delta--down")}>
      {up ? "▲" : "▼"} {pct}%
    </span>
  );
}

// Floating info box for chart hover state. Positioned by percentage within a
// `position: relative` wrapper — since the SVGs it floats over use
// preserveAspectRatio="none", a viewBox coordinate maps to that same percentage of the
// rendered width/height, so no pixel measurement of the actual DOM is needed.
function ChartTooltip({ leftPct, topPct, children }) {
  return (
    <div className="rag-chart-tooltip" style={{ left: `${leftPct}%`, top: `${topPct}%` }}>
      {children}
    </div>
  );
}

// KPI card with a trend sparkline — null days (no questions asked) are dropped before
// computing the spark/delta so a sparse history doesn't read as "trending toward zero".
function KpiCard({ label, value, series, color }) {
  const vals = series.filter((v) => v != null);
  const delta = trendDelta(vals);
  const { line, area } = sparkPaths(vals, 120, 34, 4);
  return (
    <div className="rag-kpi-card">
      <span className="rag-kpi-card__label">{label}</span>
      <span className="rag-kpi-card__value">{value}</span>
      <div className="rag-kpi-card__foot">
        <svg viewBox="0 0 120 34" className="rag-kpi-card__spark">
          <path d={area} fill={color} opacity="0.14" />
          <path d={line} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <DeltaBadge delta={delta} />
      </div>
    </div>
  );
}

// A "current level" companion to the KPI trend cards — a big percentage, a thin progress
// bar, and a row of mini trend bars so the day-to-day shape is visible at a glance.
// byDay carries the real dates alongside the values (not just a bare number array) so
// the per-bar hover tooltip can say *which* day, not just "0.62".
function GaugeCard({ label, valuePct, byDay, field, color, softColor, note, noDataLabel }) {
  const [hoverIdx, setHoverIdx] = useState(null);
  const series = byDay.map((d) => d[field]);
  const vals = series.filter((v) => v != null);
  const delta = trendDelta(vals);
  const hovered = hoverIdx != null ? byDay[hoverIdx] : null;

  return (
    <div className="rag-gauge-card">
      <span className="rag-gauge-card__label">{label}</span>
      <div className="rag-gauge-card__top">
        <span className="rag-gauge-card__value">{fmtPct(valuePct)}</span>
        <DeltaBadge delta={delta} />
      </div>
      <div className="rag-gauge-card__bar">
        <div className="rag-gauge-card__bar-fill" style={{ width: `${Math.round((valuePct || 0) * 100)}%`, background: color }} />
      </div>
      <div className="rag-chart-hover-wrap">
        <div className="rag-gauge-card__trend">
          {byDay.map((d, i) => {
            const v = d[field];
            return (
              <span
                key={d.date}
                className="rag-gauge-card__trend-bar"
                style={{
                  height: `${Math.max(2, Math.round((v || 0) * 100))}%`,
                  background: v == null ? "var(--panel-2)" : softColor,
                  outline: hoverIdx === i ? `1px solid ${color}` : "none",
                }}
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx((h) => (h === i ? null : h))}
              />
            );
          })}
        </div>
        {hovered && (
          <ChartTooltip leftPct={((hoverIdx + 0.5) / byDay.length) * 100} topPct={0}>
            <div className="rag-chart-tooltip__title">{fmtDateShort(hovered.date)}</div>
            {hovered[field] != null
              ? <div className="rag-chart-tooltip__row"><span className="rag-chart-tooltip__dot" style={{ background: color }} />{fmtPct(hovered[field])}</div>
              : <div className="rag-dim">{noDataLabel}</div>}
          </ChartTooltip>
        )}
      </div>
      <span className="rag-gauge-card__note">{note}</span>
    </div>
  );
}

// Manages which of the two dashboard levels is showing (space overview or one chat's
// question list) plus the question-detail overlay, which reuses the existing
// PipelineOverlay unchanged — same merge-cache-fields-onto-the-trace pattern the chat
// panel used to do before that view moved here.
export function useInsightsController(spaceId) {
  const [chatId, setChatId] = useState(null);
  const [spaceData, setSpaceData] = useState(null);
  const [chunkData, setChunkData] = useState(null);
  const [chatData, setChatData] = useState(null);
  const [error, setError] = useState(null);
  const [trace, setTrace] = useState(null);
  const [loadingTraceId, setLoadingTraceId] = useState(null);
  // {start, end} once resolved from the backend's own default (last 14 days) — null
  // means "not resolved yet", never a stale value from a previous space, since
  // refreshSpace always takes the range to use as an explicit argument rather than
  // reading this state itself.
  const [range, setRange] = useState(null);

  function refreshSpace(explicitRange) {
    setError(null);
    Promise.all([spaceInsights(spaceId, explicitRange), chunkInsights(spaceId)])
      .then(([space, chunks]) => {
        setSpaceData(space);
        setChunkData(chunks);
        setRange({ start: space.range_start, end: space.range_end });
      })
      .catch((err) => setError(err.message));
  }

  useEffect(() => {
    setChatId(null);
    setChatData(null);
    setRange(null);
    refreshSpace(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spaceId]);

  function setDateRange(start, end) {
    refreshSpace({ start, end });
  }

  function openChat(id) {
    setChatId(id);
    setError(null);
    chatInsights(spaceId, id)
      .then(setChatData)
      .catch((err) => setError(err.message));
  }

  function backToSpace() {
    setChatId(null);
    setChatData(null);
    refreshSpace(range);
  }

  async function openQuestion(row) {
    setLoadingTraceId(row.message_id);
    try {
      const t = await messageTrace(row.message_id);
      if (!t.meta) Object.assign(t, await messageVectors(row.message_id));
      t.cache_hit = row.cache_hit;
      t.cache_kind = row.cache_kind;
      t.cache_match_question = row.cache_match_question;
      t.cache_match_score = row.cache_match_score;
      setTrace(t);
    } finally {
      setLoadingTraceId(null);
    }
  }

  return {
    chatId, spaceData, chunkData, chatData, error, trace, setTrace, loadingTraceId, range,
    refreshSpace, setDateRange, openChat, backToSpace, openQuestion,
  };
}

function StatCard({ label, value, sub }) {
  return (
    <div className="rag-insight-card">
      <span className="rag-insight-card__label">{label}</span>
      <span className="rag-insight-card__value">{value}</span>
      {sub && <span className="rag-insight-card__sub">{sub}</span>}
    </div>
  );
}

const GATE_COLOR_VAR = { accent2: "var(--accent2)", warn: "var(--warn)", accent: "var(--accent)", doc: "var(--doc)" };
const GATE_KEYS = ["answered", "no-match", "wide-fallback", "partial"];
const GATE_PCT_FIELD = {
  answered: "answered_pct", "no-match": "no_match_pct", "wide-fallback": "wide_fallback_pct",
  partial: "partial_pct",
};

// Days with zero questions are dropped from the band paths entirely (rather than
// plotted as a 0% dip) — same "no data" vs. "asked, got nothing" distinction the rest
// of this page preserves, just expressed as a gap in the stack instead of a gap in a line.
function GateOutcomeTrend({ gateOutcomes, gateOutcomesByDay, rangeDayCount, rangeStartLabel, rangeEndLabel }) {
  const [hoverIdx, setHoverIdx] = useState(null);
  const total = Object.values(gateOutcomes).reduce((a, b) => a + b, 0);
  const X0 = 40, X1 = 770, Y_TOP = 16, Y_BOTTOM = 176;
  const n = gateOutcomesByDay.length;
  const xAt = (i) => (n > 1 ? X0 + (i / (n - 1)) * (X1 - X0) : (X0 + X1) / 2);

  const pts = gateOutcomesByDay
    .map((d, i) => {
      if (!d.total) return null;
      const cum1 = d.answered_pct || 0;
      const cum2 = cum1 + (d.partial_pct || 0);
      const cum3 = cum2 + (d.wide_fallback_pct || 0);
      const cum4 = cum3 + (d.no_match_pct || 0);
      return { x: xAt(i), cum1, cum2, cum3, cum4 };
    })
    .filter(Boolean);

  const toY = (f) => Y_BOTTOM - f * (Y_BOTTOM - Y_TOP);
  function band(topKey, botKey) {
    if (pts.length < 2) return "";
    let d = `M${pts[0].x.toFixed(1)},${toY(pts[0][topKey]).toFixed(1)}`;
    for (let i = 1; i < pts.length; i++) d += ` L${pts[i].x.toFixed(1)},${toY(pts[i][topKey]).toFixed(1)}`;
    for (let i = pts.length - 1; i >= 0; i--) d += ` L${pts[i].x.toFixed(1)},${toY(botKey ? pts[i][botKey] : 0).toFixed(1)}`;
    return `${d} Z`;
  }
  const bands = [
    { key: "answered", d: band("cum1", null), color: GATE_COLOR_VAR.accent2 },
    { key: "partial", d: band("cum2", "cum1"), color: GATE_COLOR_VAR.doc },
    { key: "wide-fallback", d: band("cum3", "cum2"), color: GATE_COLOR_VAR.accent },
    { key: "no-match", d: band("cum4", "cum3"), color: GATE_COLOR_VAR.warn },
  ];

  const step = n > 1 ? (X1 - X0) / (n - 1) : X1 - X0;
  const hovered = hoverIdx != null ? gateOutcomesByDay[hoverIdx] : null;

  return (
    <div className="rag-insight-panel">
      <div className="rag-insight-panel__title">Gate outcomes ({rangeDayCount}d)</div>
      <div className="rag-gate-trend__legend">
        {GATE_KEYS.map((k) => (
          <span key={k} className="rag-gate-bar__legend-item">
            <span className={cls("rag-gate-bar__dot", `rag-gate-bar__dot--${GATE_TONE[k]}`)} />
            {GATE_LABEL[k]}
            <span className="rag-mono rag-dim">{fmtPct(total ? gateOutcomes[k] / total : 0)}</span>
          </span>
        ))}
      </div>
      <div className="rag-chart-hover-wrap">
        <svg viewBox="0 0 780 202" preserveAspectRatio="none" className="rag-gate-trend__svg">
          <line x1={X0} y1={Y_BOTTOM} x2={X1} y2={Y_BOTTOM} className="rag-chart__axis" />
          <line x1={X0} y1="96" x2={X1} y2="96" className="rag-chart__axis" strokeDasharray="3 3" />
          <line x1={X0} y1={Y_TOP} x2={X1} y2={Y_TOP} className="rag-chart__axis" />
          <text x={X0 - 8} y={Y_BOTTOM + 3} textAnchor="end" className="rag-chart__value">0%</text>
          <text x={X0 - 8} y="99" textAnchor="end" className="rag-chart__value">50%</text>
          <text x={X0 - 8} y={Y_TOP + 3} textAnchor="end" className="rag-chart__value">100%</text>
          {pts.length >= 2
            ? bands.map((b) => <path key={b.key} d={b.d} fill={b.color} opacity="0.88" stroke="var(--panel)" strokeWidth="1" />)
            : <text x={(X0 + X1) / 2} y={(Y_TOP + Y_BOTTOM) / 2} textAnchor="middle" className="rag-chart__value">Not enough data yet</text>}
          <text x={X0} y="198" className="rag-chart__value">{rangeStartLabel}</text>
          <text x={X1} y="198" textAnchor="end" className="rag-chart__value">{rangeEndLabel}</text>
          {hoverIdx != null && (
            <line x1={xAt(hoverIdx)} y1={Y_TOP} x2={xAt(hoverIdx)} y2={Y_BOTTOM} className="rag-chart__guide" />
          )}
          {/* One invisible hit-area per day (including no-data days, so hovering a gap still
              explains itself) rather than continuous mouse tracking — cheaper and exact. */}
          {gateOutcomesByDay.map((d, i) => (
            <rect
              key={d.date}
              x={Math.max(X0, xAt(i) - step / 2)} y={Y_TOP}
              width={Math.min(X1, xAt(i) + step / 2) - Math.max(X0, xAt(i) - step / 2)} height={Y_BOTTOM - Y_TOP}
              fill="transparent"
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx((h) => (h === i ? null : h))}
            />
          ))}
        </svg>
        {hovered && (
          <ChartTooltip leftPct={(xAt(hoverIdx) / 780) * 100} topPct={4}>
            <div className="rag-chart-tooltip__title">{fmtDateShort(hovered.date)}</div>
            {hovered.total
              ? (
                <>
                  <div className="rag-dim" style={{ marginBottom: 4 }}>{hovered.total} question{hovered.total === 1 ? "" : "s"}</div>
                  {GATE_KEYS.map((k) => (
                    <div className="rag-chart-tooltip__row" key={k}>
                      <span className={cls("rag-chart-tooltip__dot", `rag-gate-bar__dot--${GATE_TONE[k]}`)} />
                      {GATE_LABEL[k]} · {fmtPct(hovered[GATE_PCT_FIELD[k]] || 0)}
                    </div>
                  ))}
                </>
              )
              : <div className="rag-dim">No questions this day</div>}
          </ChartTooltip>
        )}
      </div>
    </div>
  );
}

function StageLatencyPanel({ stageLatency }) {
  const maxMs = Math.max(...stageLatency.map((s) => s.avg_ms || 0), 1);
  return (
    <div className="rag-insight-panel">
      <div className="rag-insight-panel__title">
        Avg latency per stage
        <span className="rag-insight-panel__legend">
          <span><span className="rag-dot rag-dot--accent" /> LLM call</span>
          <span><span className="rag-dot rag-dot--local" /> local</span>
        </span>
      </div>
      <div className="rag-stage-bars">
        {stageLatency.map((s) => (
          <div className="rag-stage-bars__row" key={s.stage}>
            <span className="rag-stage-bars__label">{STAGE_LABEL[s.stage]}</span>
            <span className="rag-stage-bars__track">
              <span
                className={cls("rag-stage-bars__fill", s.is_api ? "rag-stage-bars__fill--accent" : "rag-stage-bars__fill--local")}
                style={{ width: s.avg_ms ? `${(s.avg_ms / maxMs) * 100}%` : "0%" }}
              />
            </span>
            <span className="rag-mono rag-stage-bars__ms">{fmtMs(s.avg_ms)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Single line, total tokens (prompt + completion) per day — real daily sums from stored
// traces, not an estimate. A day with zero questions is a genuine 0, not a gap, unlike
// the gate/cache-hit trends: token usage has no "no data" vs. "data but zero" distinction
// to preserve.
function TokenUsageTrend({ tokensByDay, rangeDayCount, rangeStartLabel, rangeEndLabel }) {
  const gradientId = useId();
  const [hoverIdx, setHoverIdx] = useState(null);
  const X0 = 40, X1 = 770, Y_TOP = 16, Y_BOTTOM = 176;
  const totals = tokensByDay.map((d) => d.prompt_tokens + d.completion_tokens);
  const grandTotal = totals.reduce((a, b) => a + b, 0);
  const maxTotal = Math.max(...totals, 1) * 1.1;
  const n = totals.length;
  const xAt = (i) => (n > 1 ? X0 + (i / (n - 1)) * (X1 - X0) : (X0 + X1) / 2);
  const yAt = (v) => Y_BOTTOM - (v / maxTotal) * (Y_BOTTOM - Y_TOP);
  const points = totals.map((v, i) => ({ x: xAt(i), y: yAt(v) }));
  const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const area = points.length ? `${line} L${points[n - 1].x.toFixed(1)},${Y_BOTTOM} L${points[0].x.toFixed(1)},${Y_BOTTOM} Z` : "";
  const step = n > 1 ? (X1 - X0) / (n - 1) : X1 - X0;
  const hovered = hoverIdx != null ? tokensByDay[hoverIdx] : null;

  return (
    <div className="rag-insight-panel">
      <div className="rag-insight-panel__title" style={{ justifyContent: "space-between" }}>
        <span>Token usage</span>
        <span className="rag-mono rag-dim">{grandTotal.toLocaleString()} total</span>
      </div>
      <div className="rag-chart-hover-wrap">
        <svg viewBox="0 0 780 202" preserveAspectRatio="none" className="rag-gate-trend__svg">
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.28" />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.02" />
            </linearGradient>
          </defs>
          <line x1={X0} y1={Y_BOTTOM} x2={X1} y2={Y_BOTTOM} className="rag-chart__axis" />
          <line x1={X0} y1="96" x2={X1} y2="96" className="rag-chart__axis" strokeDasharray="3 3" />
          <line x1={X0} y1={Y_TOP} x2={X1} y2={Y_TOP} className="rag-chart__axis" />
          <text x={X0 - 8} y={Y_BOTTOM + 3} textAnchor="end" className="rag-chart__value">0</text>
          <text x={X0 - 8} y={Y_TOP + 3} textAnchor="end" className="rag-chart__value">{Math.round(maxTotal).toLocaleString()}</text>
          <path d={area} fill={`url(#${gradientId})`} />
          <path d={line} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <text x={X0} y="198" className="rag-chart__value">{rangeStartLabel}</text>
          <text x={X1} y="198" textAnchor="end" className="rag-chart__value">{rangeEndLabel}</text>
          {hoverIdx != null && (
            <>
              <line x1={xAt(hoverIdx)} y1={Y_TOP} x2={xAt(hoverIdx)} y2={Y_BOTTOM} className="rag-chart__guide" />
              <circle cx={xAt(hoverIdx)} cy={yAt(totals[hoverIdx])} r="4" fill="var(--accent)" stroke="var(--panel)" strokeWidth="1.5" />
            </>
          )}
          {tokensByDay.map((d, i) => (
            <rect
              key={d.date}
              x={Math.max(X0, xAt(i) - step / 2)} y={Y_TOP}
              width={Math.min(X1, xAt(i) + step / 2) - Math.max(X0, xAt(i) - step / 2)} height={Y_BOTTOM - Y_TOP}
              fill="transparent"
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx((h) => (h === i ? null : h))}
            />
          ))}
        </svg>
        {hovered && (
          <ChartTooltip leftPct={(xAt(hoverIdx) / 780) * 100} topPct={4}>
            <div className="rag-chart-tooltip__title">{fmtDateShort(hovered.date)}</div>
            <div className="rag-chart-tooltip__row">Prompt · {hovered.prompt_tokens.toLocaleString()}</div>
            <div className="rag-chart-tooltip__row">Completion · {hovered.completion_tokens.toLocaleString()}</div>
            <div className="rag-chart-tooltip__row" style={{ fontWeight: 700 }}>
              Total · {(hovered.prompt_tokens + hovered.completion_tokens).toLocaleString()}
            </div>
          </ChartTooltip>
        )}
      </div>
    </div>
  );
}

function DateRangePicker({ range, minDate, maxDate, dayCount, onChange }) {
  if (!range) return null;
  return (
    <div className="rag-date-range">
      <span className="rag-date-range__label">Date range</span>
      <input
        type="date" className="rag-date-range__input" value={range.start}
        min={minDate} max={range.end} onChange={(e) => onChange(e.target.value, range.end)}
      />
      <span className="rag-dim">to</span>
      <input
        type="date" className="rag-date-range__input" value={range.end}
        min={range.start} max={maxDate} onChange={(e) => onChange(range.start, e.target.value)}
      />
      <span className="rag-hint" style={{ margin: 0 }}>{dayCount} day{dayCount === 1 ? "" : "s"} selected</span>
    </div>
  );
}

const PAGE_SIZE = 10;

// Shared by ChatsTable/ChunkExplorer — `page` is clamped against the current item count
// rather than reset via an effect, so a filter/sort change that shrinks the list below
// the current page just falls back to the last valid page instead of showing blank.
function usePage(itemCount) {
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(itemCount / PAGE_SIZE));
  const clamped = Math.min(page, totalPages);
  return [clamped, setPage, totalPages];
}

function Pagination({ page, setPage, totalPages, totalItems }) {
  if (totalItems === 0) return null;
  const start = (page - 1) * PAGE_SIZE + 1;
  const end = Math.min(page * PAGE_SIZE, totalItems);
  return (
    <div className="rag-pagination">
      <span className="rag-dim">{start}–{end} of {totalItems}</span>
      <div className="rag-pagination__buttons">
        <button className="rag-btn--ghost rag-pagination__btn" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          ‹ Prev
        </button>
        <span className="rag-dim">Page {page} of {totalPages}</span>
        <button className="rag-btn--ghost rag-pagination__btn" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
          Next ›
        </button>
      </div>
    </div>
  );
}

function ChatsTable({ chats, onSelect }) {
  const [page, setPage, totalPages] = usePage(chats.length);
  const pageChats = chats.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div className="rag-insight-table">
      <div className="rag-insight-table__title">Chats</div>
      <div className="rag-insight-table__head" style={{ gridTemplateColumns: "1.7fr 60px 90px 90px 90px 110px 20px" }}>
        <span>Chat</span><span>Msgs</span><span>Active</span><span>Cache hit</span><span>Avg latency</span><span>Tokens (in/out)</span><span />
      </div>
      {chats.length === 0 && <p className="rag-hint" style={{ padding: 16 }}>No chats in this space yet.</p>}
      {pageChats.map((c) => (
        <div
          key={c.id}
          className="rag-insight-table__row"
          style={{ gridTemplateColumns: "1.7fr 60px 90px 90px 90px 110px 20px" }}
          onClick={() => onSelect(c.id)}
        >
          <span className="rag-insight-table__strong">{c.title}</span>
          <span className="rag-mono">{c.message_count}</span>
          <span className="rag-dim">{new Date(c.last_active).toLocaleString()}</span>
          <span className="rag-mono">{fmtPct(c.cache_hit_rate)}</span>
          <span className="rag-mono">{fmtMs(c.avg_latency_ms)}</span>
          <span className="rag-mono">{c.tokens.prompt_tokens.toLocaleString()} / {c.tokens.completion_tokens.toLocaleString()}</span>
          <span className="rag-dim">›</span>
        </div>
      ))}
      <Pagination page={page} setPage={setPage} totalPages={totalPages} totalItems={chats.length} />
    </div>
  );
}

function ChunkExplorer({ chunkData }) {
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState({ key: "candidate_count", dir: "desc" });

  const filtered = chunkData
    ? chunkData.chunks.filter(
        (c) => !filter.trim() || c.file_path.toLowerCase().includes(filter.trim().toLowerCase())
      )
    : [];
  filtered.sort((a, b) => (sort.dir === "desc" ? b[sort.key] - a[sort.key] : a[sort.key] - b[sort.key]));
  const [page, setPage, totalPages] = usePage(filtered.length);
  const pageChunks = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  if (!chunkData) return null;

  function toggleSort(key) {
    setSort((s) => ({ key, dir: s.key === key && s.dir === "desc" ? "asc" : "desc" }));
  }
  const arrow = (key) => (sort.key === key ? (sort.dir === "desc" ? " ↓" : " ↑") : "");

  return (
    <div className="rag-insight-table">
      <div className="rag-insight-table__title" style={{ display: "flex", justifyContent: "space-between", gap: 14 }}>
        <span>Chunk Explorer <span className="rag-dim">· {filtered.length} of {chunkData.chunk_count} chunks</span></span>
        <input
          className="rag-input"
          style={{ width: 230 }}
          placeholder="Filter by source or location…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      <div className="rag-insight-table__head" style={{ gridTemplateColumns: "1.7fr 70px 90px 80px 90px" }}>
        <span>Chunk</span>
        <span onClick={() => toggleSort("candidate_count")} style={{ cursor: "pointer" }}>Retr.{arrow("candidate_count")}</span>
        <span>Rerank</span>
        <span onClick={() => toggleSort("used_count")} style={{ cursor: "pointer" }}>Used{arrow("used_count")}</span>
        <span>Avg score</span>
      </div>
      <div>
        {filtered.length === 0 && <p className="rag-hint" style={{ padding: 16 }}>No chunks match this filter.</p>}
        {pageChunks.map((c) => (
          <div key={c.chunk_id} className="rag-insight-table__row" style={{ gridTemplateColumns: "1.7fr 70px 90px 80px 90px", cursor: "default" }}>
            <span className="rag-mono rag-insight-table__truncate">{c.file_path}</span>
            <span className="rag-mono">{c.candidate_count}</span>
            <span className="rag-mono rag-dim">{c.reranked_count}</span>
            <span className="rag-mono">{c.used_count}</span>
            <span className="rag-mono rag-dim">{c.avg_rerank_score != null ? c.avg_rerank_score.toFixed(2) : "—"}</span>
          </div>
        ))}
      </div>
      <Pagination page={page} setPage={setPage} totalPages={totalPages} totalItems={filtered.length} />
    </div>
  );
}

// File-level rollup of the chunk-level stats chunkInsights returns — the mockup's
// "Most retrieved files" panel groups by file, not by chunk.
function TopFilesPanel({ chunkData }) {
  if (!chunkData || chunkData.chunks.length === 0) return null;

  const byFile = new Map();
  for (const c of chunkData.chunks) {
    const entry = byFile.get(c.file_path) || { path: c.file_path, candidates: 0, used: 0, scoreSum: 0, scoreCount: 0 };
    entry.candidates += c.candidate_count;
    entry.used += c.used_count;
    if (c.avg_rerank_score != null) {
      entry.scoreSum += c.avg_rerank_score;
      entry.scoreCount += 1;
    }
    byFile.set(c.file_path, entry);
  }
  const files = [...byFile.values()]
    .map((f) => ({ ...f, score: f.scoreCount ? f.scoreSum / f.scoreCount : null }))
    .sort((a, b) => b.candidates - a.candidates)
    .slice(0, 8);
  const maxCandidates = Math.max(...files.map((f) => f.candidates), 1);

  return (
    <div className="rag-insight-panel" style={{ marginBottom: 16 }}>
      <div className="rag-insight-panel__title">Most retrieved files</div>
      <p className="rag-hint" style={{ margin: "-8px 0 12px" }}>
        Grey shows chunks retrieved as candidates, green shows chunks that made the final prompt.
      </p>
      <div className="rag-topfiles">
        {files.map((f) => (
          <div className="rag-topfiles__row" key={f.path}>
            <span className="rag-mono rag-topfiles__path">{f.path}</span>
            <span className="rag-topfiles__track">
              <span className="rag-topfiles__fill rag-topfiles__fill--candidates" style={{ width: `${(f.candidates / maxCandidates) * 100}%` }} />
              <span className="rag-topfiles__fill rag-topfiles__fill--used" style={{ width: `${(f.used / maxCandidates) * 100}%` }} />
            </span>
            <span className="rag-mono rag-topfiles__count">{f.used}/{f.candidates}</span>
            <span className="rag-mono rag-topfiles__score">{f.score != null ? f.score.toFixed(2) : "—"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function SpaceInsightsView({ ctl }) {
  const { spaceData, chunkData, error, openChat, range, setDateRange } = ctl;
  if (error) return <p className="rag-error" style={{ margin: 24 }}>{error}</p>;
  if (!spaceData) return <p className="rag-hint" style={{ margin: 24 }}>Loading…</p>;

  const rangeStartLabel = fmtDateShort(spaceData.range_start);
  const rangeEndLabel = fmtDateShort(spaceData.range_end);
  const faithfulVals = spaceData.faithfulness_by_day.map((d) => d.faithful_rate).filter((v) => v != null);
  const faithfulAvg = avg(faithfulVals);

  return (
    <div className="rag-insights">
      <p className="rag-hint" style={{ margin: "0 0 16px" }}>This space's retrieval pipeline, aggregated.</p>
      <DateRangePicker
        range={range} minDate={spaceData.range_min_date} maxDate={spaceData.range_max_date}
        dayCount={spaceData.range_day_count} onChange={setDateRange}
      />
      <div className="rag-insight-cards">
        <KpiCard
          label="Cache hit rate" value={fmtPct(spaceData.cache_hit_rate)} color="var(--accent2)"
          series={spaceData.cache_hit_by_day.map((d) => d.hit_rate)}
        />
        <KpiCard
          label="Avg total latency" value={fmtMs(spaceData.avg_latency_ms)} color="var(--accent)"
          series={spaceData.latency_by_day.map((d) => d.avg_ms)}
        />
        <KpiCard
          label="Tokens / day" value={Math.round(avg(spaceData.tokens_by_day.map((d) => d.prompt_tokens + d.completion_tokens))).toLocaleString()}
          color="var(--doc)" series={spaceData.tokens_by_day.map((d) => d.prompt_tokens + d.completion_tokens)}
        />
        <KpiCard
          label="Decomposition rate" value={fmtPct(spaceData.decomposition_rate)} color="var(--accent)"
          series={spaceData.decomposition_by_day.map((d) => d.rate)}
        />
      </div>
      <div className="rag-insight-row">
        <GateOutcomeTrend
          gateOutcomes={spaceData.gate_outcomes} gateOutcomesByDay={spaceData.gate_outcomes_by_day}
          rangeDayCount={spaceData.range_day_count} rangeStartLabel={rangeStartLabel} rangeEndLabel={rangeEndLabel}
        />
        <StageLatencyPanel stageLatency={spaceData.stage_latency} />
      </div>
      <div className="rag-insight-row rag-insight-row--token">
        <TokenUsageTrend
          tokensByDay={spaceData.tokens_by_day} rangeDayCount={spaceData.range_day_count}
          rangeStartLabel={rangeStartLabel} rangeEndLabel={rangeEndLabel}
        />
        <GaugeCard
          label="Cache hit rate" valuePct={spaceData.cache_hit_rate} color="var(--accent2)" softColor="var(--accent2-soft)"
          byDay={spaceData.cache_hit_by_day} field="hit_rate" note={`avg over ${spaceData.range_day_count} days`}
          noDataLabel="No questions this day"
        />
        <GaugeCard
          label="Faithfulness rate" valuePct={faithfulAvg} color="var(--doc)" softColor="var(--doc-soft)"
          byDay={spaceData.faithfulness_by_day} field="faithful_rate" note="fully-supported claims"
          noDataLabel="No answers with claims to check"
        />
      </div>
      <ChatsTable chats={spaceData.chats} onSelect={openChat} />
      <TopFilesPanel chunkData={chunkData} />
      <ChunkExplorer chunkData={chunkData} />
    </div>
  );
}

export function ChatInsightsView({ ctl }) {
  const { chatData, error, openQuestion, loadingTraceId } = ctl;
  const [filter, setFilter] = useState("");
  const [outcomeFilter, setOutcomeFilter] = useState("all");

  if (error) return <p className="rag-error" style={{ margin: 24 }}>{error}</p>;
  if (!chatData) return <p className="rag-hint" style={{ margin: 24 }}>Loading…</p>;

  const filtered = chatData.questions.filter((q) => {
    if (filter.trim() && !q.question.toLowerCase().includes(filter.trim().toLowerCase())) return false;
    if (outcomeFilter === "all") return true;
    // gate_outcome is preserved correctly through a cache hit (the backend copies the
    // ORIGINAL run's real outcome, sufficiency included) — no special-casing needed.
    return q.gate_outcome === outcomeFilter;
  });

  return (
    <div className="rag-insights">
      <div className="rag-insight-cards" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
        <StatCard label="Questions" value={chatData.question_count} />
        <StatCard label="Cache hit rate" value={fmtPct(chatData.cache_hit_rate)} />
        <StatCard label="Avg latency" value={fmtMs(chatData.avg_latency_ms)} />
      </div>
      <div style={{ display: "flex", gap: 8, margin: "16px 0 14px", flexWrap: "wrap" }}>
        <input
          className="rag-input" style={{ width: 240 }} placeholder="Filter questions…"
          value={filter} onChange={(e) => setFilter(e.target.value)}
        />
        {["all", "answered", "partial", "no-match", "wide-fallback"].map((k) => (
          <button
            key={k}
            className={cls("rag-filter-btn", outcomeFilter === k && "rag-filter-btn--active")}
            onClick={() => setOutcomeFilter(k)}
          >
            {k === "all" ? "All" : GATE_LABEL[k]}
          </button>
        ))}
      </div>
      <div className="rag-insight-table">
        <div className="rag-insight-table__head" style={{ gridTemplateColumns: "70px minmax(180px,2fr) 92px 56px 55px 60px 100px 90px 14px" }}>
          <span>Time</span><span>Question</span><span>Gate</span><span>Cache</span><span>Chunks</span><span>Latency</span><span>Tokens (in/out)</span><span>Model</span><span />
        </div>
        {filtered.length === 0 && <p className="rag-hint" style={{ padding: 16 }}>No questions match this filter.</p>}
        {filtered.map((q) => (
          <div
            key={q.message_id}
            className="rag-insight-table__row"
            style={{ gridTemplateColumns: "70px minmax(180px,2fr) 92px 56px 55px 60px 100px 90px 14px" }}
            onClick={() => openQuestion(q)}
          >
            <span className="rag-dim">{new Date(q.asked_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
            <span className="rag-insight-table__truncate">{loadingTraceId === q.message_id ? "Loading…" : q.question}</span>
            <span className={cls("rag-tag", `rag-tag--${GATE_TONE[q.gate_outcome]}`)}>
              {GATE_LABEL[q.gate_outcome]}
            </span>
            <span className={q.cache_hit ? "rag-dim" : "rag-dim"} style={{ fontWeight: 600, color: q.cache_hit ? "var(--accent2-ink)" : undefined }}>
              {q.cache_hit ? "Hit" : "Miss"}
            </span>
            <span className="rag-mono">{q.cache_hit ? "—" : q.chunk_count}</span>
            <span className="rag-mono">{fmtMs(q.latency_ms)}</span>
            <span className="rag-mono">{q.tokens.prompt_tokens.toLocaleString()} / {q.tokens.completion_tokens.toLocaleString()}</span>
            <span className="rag-mono rag-insight-table__model">{q.model}</span>
            <span className="rag-dim">›</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function InsightsMain({ spaceId, spaceName, ctl }) {
  const { chatId, chatData, trace, setTrace, backToSpace } = ctl;
  const [subTab, setSubTab] = useState("overview"); // "overview" | "connectors" — space-level only

  return (
    <div className="rag-insights-shell">
      <div className="rag-insights-breadcrumb">
        <button className={cls("rag-breadcrumb__link", !chatId && "rag-breadcrumb__link--current")} onClick={backToSpace}>
          {spaceName} Insights
        </button>
        {chatId && (
          <>
            <span className="rag-breadcrumb__sep">›</span>
            <span className="rag-breadcrumb__current">{chatData?.title || "…"}</span>
          </>
        )}
        {!chatId && (
          <div className="rag-tabs" style={{ marginLeft: "auto" }}>
            <button className={cls("rag-tabs__btn", subTab === "overview" && "rag-tabs__btn--active")} onClick={() => setSubTab("overview")}>
              Overview
            </button>
            <button className={cls("rag-tabs__btn", subTab === "connectors" && "rag-tabs__btn--active")} onClick={() => setSubTab("connectors")}>
              Connectors
            </button>
          </div>
        )}
      </div>
      <div className="rag-insights-scroll">
        {chatId ? (
          <ChatInsightsView ctl={ctl} />
        ) : subTab === "connectors" ? (
          <ConnectorsPanel spaceId={spaceId} />
        ) : (
          <SpaceInsightsView ctl={ctl} />
        )}
      </div>
      {trace && (
        <PipelineOverlay mode="query" data={trace} title={trace.question} onClose={() => setTrace(null)} />
      )}
    </div>
  );
}
