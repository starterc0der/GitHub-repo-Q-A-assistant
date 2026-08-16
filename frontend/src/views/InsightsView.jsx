import { useEffect, useState } from "react";
import { chatInsights, chunkInsights, messageTrace, messageVectors, spaceInsights } from "../api.js";
import { cls } from "../components/RagAtoms.jsx";
import { PipelineOverlay } from "../components/PipelineOverlay.jsx";

// "partial": a decomposed question where some (not all) sub-questions found no
// evidence — a real answer was generated, but not a full success, so it gets its own
// color rather than being folded into "Answered". See Pipeline._retrieve.
const GATE_LABEL = { answered: "Answered", "no-match": "No match", "wide-fallback": "Wide fallback", partial: "Partial" };
const GATE_TONE = { answered: "accent2", "no-match": "warn", "wide-fallback": "accent", partial: "doc" };
const STAGE_LABEL = {
  cache: "Cache check", decompose: "Decomposition", embed: "Embed", route: "Route",
  hybrid: "Hybrid search", rerank: "Rerank", gate: "Gate", retry: "Retry",
  compress: "Compression", generate: "Generation",
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

// 3 fixed outcomes, one line each, real daily share over the last 14 days from actual
// message timestamps — a day with zero questions leaves a gap in that line rather than
// drawing a fake 0%, so a mostly-empty history (a brand-new space) reads as "no data
// yet", not as "everything got no-matched."
function pointsFor(byDay, pctField, x0, x1, y0, y1) {
  const n = byDay.length;
  return byDay
    .map((d, i) => {
      if (d[pctField] == null) return null;
      return {
        x: x0 + (i / (n - 1)) * (x1 - x0),
        y: y1 - d[pctField] * (y1 - y0),
      };
    })
    .filter(Boolean);
}

function GateOutcomeTrend({ gateOutcomes, gateOutcomesByDay, rangeDayCount, rangeStartLabel, rangeEndLabel }) {
  const total = Object.values(gateOutcomes).reduce((a, b) => a + b, 0);
  const X0 = 34, X1 = 290, Y_TOP = 10, Y_BOTTOM = 100;

  return (
    <div className="rag-insight-panel">
      <div className="rag-insight-panel__title">Gate outcomes ({rangeDayCount}d)</div>
      <div className="rag-gate-trend__legend">
        {GATE_KEYS.map((k) => (
          <span key={k} className="rag-gate-bar__legend-item">
            <span className={cls("rag-gate-bar__dot", `rag-gate-bar__dot--${GATE_TONE[k]}`)} />
            {GATE_LABEL[k]}
            <span className="rag-mono rag-dim">{gateOutcomes[k]} · {fmtPct(total ? gateOutcomes[k] / total : 0)}</span>
          </span>
        ))}
      </div>
      {gateOutcomesByDay && (
        <svg viewBox="0 0 300 122" className="rag-gate-trend__svg">
          <line x1={X0} y1={Y_TOP} x2={X1} y2={Y_TOP} className="rag-chart__axis" />
          <line x1={X0} y1="55" x2={X1} y2="55" className="rag-chart__axis" />
          <line x1={X0} y1={Y_BOTTOM} x2={X1} y2={Y_BOTTOM} className="rag-chart__axis" />
          <line x1={X0} y1={Y_TOP} x2={X0} y2={Y_BOTTOM} className="rag-chart__axis" />
          <text x={X0 - 5} y={Y_TOP + 3} textAnchor="end" className="rag-chart__value">100%</text>
          <text x={X0 - 5} y="58" textAnchor="end" className="rag-chart__value">50%</text>
          <text x={X0 - 5} y={Y_BOTTOM + 3} textAnchor="end" className="rag-chart__value">0%</text>
          {GATE_KEYS.map((k) => {
            const points = pointsFor(gateOutcomesByDay, GATE_PCT_FIELD[k], X0, X1, Y_TOP, Y_BOTTOM);
            const color = GATE_COLOR_VAR[GATE_TONE[k]];
            return (
              <g key={k}>
                {/* A polyline needs 2+ points to draw anything — a marker per point
                    keeps a single real day of data (or any isolated day next to a gap)
                    visible instead of silently vanishing. */}
                <polyline
                  points={points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}
                  fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                />
                {points.map((p, i) => (
                  <circle key={i} cx={p.x} cy={p.y} r="4" fill={color} stroke="var(--panel)" strokeWidth="1.5" />
                ))}
              </g>
            );
          })}
          <text x={X0} y="116" className="rag-chart__value">{rangeStartLabel}</text>
          <text x={X1} y="116" textAnchor="end" className="rag-chart__value">{rangeEndLabel}</text>
        </svg>
      )}
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
  const X0 = 34, X1 = 290, Y_TOP = 10, Y_BOTTOM = 100;
  const totals = tokensByDay.map((d) => d.prompt_tokens + d.completion_tokens);
  const grandTotal = totals.reduce((a, b) => a + b, 0);
  const maxTotal = Math.max(...totals, 1) * 1.1;
  const n = totals.length;
  const points = totals.map((v, i) => ({
    x: n > 1 ? X0 + (i / (n - 1)) * (X1 - X0) : (X0 + X1) / 2,
    y: Y_BOTTOM - (v / maxTotal) * (Y_BOTTOM - Y_TOP),
  }));

  return (
    <div className="rag-insight-panel">
      <div className="rag-insight-panel__title">Token usage trend ({rangeDayCount}d)</div>
      <p className="rag-hint" style={{ margin: "-8px 0 10px" }}>{grandTotal.toLocaleString()} tokens over {rangeDayCount}d</p>
      <svg viewBox="0 0 300 122" className="rag-gate-trend__svg">
        <line x1={X0} y1={Y_TOP} x2={X1} y2={Y_TOP} className="rag-chart__axis" />
        <line x1={X0} y1="55" x2={X1} y2="55" className="rag-chart__axis" />
        <line x1={X0} y1={Y_BOTTOM} x2={X1} y2={Y_BOTTOM} className="rag-chart__axis" />
        <line x1={X0} y1={Y_TOP} x2={X0} y2={Y_BOTTOM} className="rag-chart__axis" />
        <text x={X0 - 5} y={Y_TOP + 3} textAnchor="end" className="rag-chart__value">{Math.round(maxTotal).toLocaleString()}</text>
        <text x={X0 - 5} y={Y_BOTTOM + 3} textAnchor="end" className="rag-chart__value">0</text>
        <polyline
          points={points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}
          fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        />
        <text x={X0} y="116" className="rag-chart__value">{rangeStartLabel}</text>
        <text x={X1} y="116" textAnchor="end" className="rag-chart__value">{rangeEndLabel}</text>
      </svg>
    </div>
  );
}

// Bar height is a real 0-100% share (not normalized to the visible range's own max) so a
// low-hit-rate stretch reads as genuinely low, not stretched to look full — same honesty
// convention as GateOutcomeTrend's fixed 0-100% axis. A day with no questions gets a
// faint placeholder bar, distinguishable from a real 0% day.
function CacheHitHistogram({ cacheHitByDay, rangeDayCount, rangeStartLabel, rangeEndLabel }) {
  return (
    <div className="rag-insight-panel" style={{ display: "flex", flexDirection: "column" }}>
      <div className="rag-insight-panel__title">Cache hit rate by day ({rangeDayCount}d)</div>
      <div className="rag-cache-hist">
        {cacheHitByDay.map((d) => {
          const hasData = d.hit_rate != null;
          const label = hasData
            ? `${fmtDateShort(d.date)}: ${fmtPct(d.hit_rate)} cache hit`
            : `${fmtDateShort(d.date)}: no questions`;
          return (
            <div className="rag-cache-hist__col" key={d.date} title={label}>
              <div
                className={cls("rag-cache-hist__bar", !hasData && "rag-cache-hist__bar--empty")}
                style={{ height: `${Math.round((d.hit_rate || 0) * 100)}%` }}
              />
            </div>
          );
        })}
      </div>
      <div className="rag-cache-hist__axis">
        <span>{rangeStartLabel}</span><span>{rangeEndLabel}</span>
      </div>
    </div>
  );
}

// Same layout/CSS as CacheHitHistogram, same "no data" vs "checked, some unsupported"
// distinction — a day with no answers that had claims to check (all meta/no-match/wide-
// fallback) gets the faint placeholder bar, not a misleading 0%.
function FaithfulnessHistogram({ faithfulnessByDay, rangeDayCount, rangeStartLabel, rangeEndLabel }) {
  return (
    <div className="rag-insight-panel" style={{ display: "flex", flexDirection: "column" }}>
      <div className="rag-insight-panel__title">Faithfulness rate by day ({rangeDayCount}d)</div>
      <div className="rag-cache-hist">
        {faithfulnessByDay.map((d) => {
          const hasData = d.faithful_rate != null;
          const label = hasData
            ? `${fmtDateShort(d.date)}: ${fmtPct(d.faithful_rate)} fully faithful (${Math.round(d.faithful_rate * d.total)} of ${d.total})`
            : `${fmtDateShort(d.date)}: no answers with claims to check`;
          return (
            <div className="rag-cache-hist__col" key={d.date} title={label}>
              <div
                className={cls("rag-cache-hist__bar", !hasData && "rag-cache-hist__bar--empty")}
                style={{ height: `${Math.round((d.faithful_rate || 0) * 100)}%` }}
              />
            </div>
          );
        })}
      </div>
      <div className="rag-cache-hist__axis">
        <span>{rangeStartLabel}</span><span>{rangeEndLabel}</span>
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

function ChatsTable({ chats, onSelect }) {
  return (
    <div className="rag-insight-table">
      <div className="rag-insight-table__title">
        Chats <span className="rag-dim">· ranked by activity</span>
      </div>
      <div className="rag-insight-table__head" style={{ gridTemplateColumns: "1.7fr 60px 90px 90px 90px 110px 20px" }}>
        <span>Chat</span><span>Msgs</span><span>Active</span><span>Cache hit</span><span>Avg latency</span><span>Tokens (in/out)</span><span />
      </div>
      {chats.length === 0 && <p className="rag-hint" style={{ padding: 16 }}>No chats in this space yet.</p>}
      {chats.map((c) => (
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
    </div>
  );
}

function ChunkExplorer({ chunkData }) {
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState({ key: "candidate_count", dir: "desc" });

  if (!chunkData) return null;
  const filtered = chunkData.chunks.filter(
    (c) => !filter.trim() || c.file_path.toLowerCase().includes(filter.trim().toLowerCase())
  );
  filtered.sort((a, b) => (sort.dir === "desc" ? b[sort.key] - a[sort.key] : a[sort.key] - b[sort.key]));

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
      <div style={{ maxHeight: 380, overflowY: "auto" }}>
        {filtered.length === 0 && <p className="rag-hint" style={{ padding: 16 }}>No chunks match this filter.</p>}
        {filtered.map((c) => (
          <div key={c.chunk_id} className="rag-insight-table__row" style={{ gridTemplateColumns: "1.7fr 70px 90px 80px 90px", cursor: "default" }}>
            <span className="rag-mono rag-insight-table__truncate">{c.file_path}</span>
            <span className="rag-mono">{c.candidate_count}</span>
            <span className="rag-mono rag-dim">{c.reranked_count}</span>
            <span className="rag-mono">{c.used_count}</span>
            <span className="rag-mono rag-dim">{c.avg_rerank_score != null ? c.avg_rerank_score.toFixed(2) : "—"}</span>
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

  return (
    <div className="rag-insights">
      <p className="rag-hint" style={{ margin: "0 0 16px" }}>This space's retrieval pipeline, aggregated.</p>
      <DateRangePicker
        range={range} minDate={spaceData.range_min_date} maxDate={spaceData.range_max_date}
        dayCount={spaceData.range_day_count} onChange={setDateRange}
      />
      <div className="rag-insight-cards">
        <StatCard label="Cache hit rate" value={fmtPct(spaceData.cache_hit_rate)} />
        <StatCard label="Decomposition rate" value={fmtPct(spaceData.decomposition_rate)} />
        <StatCard label="Avg total latency" value={fmtMs(spaceData.avg_latency_ms)} />
        <StatCard
          label="Tokens (prompt / completion)"
          value={`${spaceData.tokens.prompt_tokens.toLocaleString()} / ${spaceData.tokens.completion_tokens.toLocaleString()}`}
        />
      </div>
      <div className="rag-insight-row">
        <GateOutcomeTrend
          gateOutcomes={spaceData.gate_outcomes} gateOutcomesByDay={spaceData.gate_outcomes_by_day}
          rangeDayCount={spaceData.range_day_count} rangeStartLabel={rangeStartLabel} rangeEndLabel={rangeEndLabel}
        />
        <StageLatencyPanel stageLatency={spaceData.stage_latency} />
      </div>
      <div className="rag-insight-row">
        <TokenUsageTrend
          tokensByDay={spaceData.tokens_by_day} rangeDayCount={spaceData.range_day_count}
          rangeStartLabel={rangeStartLabel} rangeEndLabel={rangeEndLabel}
        />
        <CacheHitHistogram
          cacheHitByDay={spaceData.cache_hit_by_day} rangeDayCount={spaceData.range_day_count}
          rangeStartLabel={rangeStartLabel} rangeEndLabel={rangeEndLabel}
        />
      </div>
      <div className="rag-insight-row">
        <FaithfulnessHistogram
          faithfulnessByDay={spaceData.faithfulness_by_day} rangeDayCount={spaceData.range_day_count}
          rangeStartLabel={rangeStartLabel} rangeEndLabel={rangeEndLabel}
        />
      </div>
      <ChatsTable chats={spaceData.chats} onSelect={openChat} />
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

export function InsightsMain({ spaceName, ctl }) {
  const { chatId, chatData, trace, setTrace, backToSpace } = ctl;
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
      </div>
      <div className="rag-insights-scroll">
        {chatId ? <ChatInsightsView ctl={ctl} /> : <SpaceInsightsView ctl={ctl} />}
      </div>
      {trace && (
        <PipelineOverlay mode="query" data={trace} title={trace.question} onClose={() => setTrace(null)} />
      )}
    </div>
  );
}
