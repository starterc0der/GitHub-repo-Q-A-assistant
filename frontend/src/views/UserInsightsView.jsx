import { useEffect, useState } from "react";
import { userInsights } from "../api.js";
import { AVATAR_BG, AVATAR_INK, RagTag } from "../components/RagAtoms.jsx";
import { GaugeCard, KpiCard, TokenUsageTrend, fmtDateShort, fmtPct } from "./InsightsView.jsx";

const TURNS_PAGE_SIZE = 8;
const RANGE_OPTIONS = [7, 14, 30];

function avg(vals) {
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

const GATE_KEYS = ["answered", "partial", "wide-fallback", "no-match"];
const GATE_LABEL = { answered: "Answered", partial: "Partial", "wide-fallback": "Wide fallback", "no-match": "No match" };
const GATE_COLOR = { answered: "var(--accent2)", partial: "var(--doc)", "wide-fallback": "var(--accent)", "no-match": "var(--warn)" };
const GATE_PCT_FIELD = { answered: "answered_pct", partial: "partial_pct", "wide-fallback": "wide_fallback_pct", "no-match": "no_match_pct" };

// Discrete per-day bars, not an interpolated line/band — a continuous band would draw a
// straight edge between two real days even when the days between them have zero
// questions, visually implying a gradual change that never happened. A day with no
// questions is a genuine gap: it renders as empty space, not a dip to 0% or a stretch of
// its neighbor's value across the blank stretch.
function QuestionOutcomesChart({ gateOutcomesByDay }) {
  const X0 = 30, X1 = 290, Y_TOP = 10, Y_BOTTOM = 130;
  const total = gateOutcomesByDay.length;
  const slotW = total > 0 ? (X1 - X0) / total : 0;
  const barW = Math.max(2, slotW * 0.62);
  const toY = (f) => Y_BOTTOM - f * (Y_BOTTOM - Y_TOP);

  const bars = gateOutcomesByDay
    .map((d, idx) => {
      if (!d.total) return null;
      const c1 = d.answered_pct || 0;
      const c2 = c1 + (d.partial_pct || 0);
      const c3 = c2 + (d.wide_fallback_pct || 0);
      const c4 = c3 + (d.no_match_pct || 0);
      const x = X0 + slotW * (idx + 0.5) - barW / 2;
      const segments = [
        { key: "answered", top: 0, bot: c1 },
        { key: "partial", top: c1, bot: c2 },
        { key: "wide-fallback", top: c2, bot: c3 },
        { key: "no-match", top: c3, bot: c4 },
      ].filter((s) => s.bot > s.top);
      return { date: d.date, x, segments };
    })
    .filter(Boolean);
  const n = bars.length;

  const totals = { answered: 0, partial: 0, "wide-fallback": 0, "no-match": 0 };
  let dayCount = 0;
  for (const d of gateOutcomesByDay) {
    if (!d.total) continue;
    dayCount += 1;
    for (const k of GATE_KEYS) totals[k] += d[GATE_PCT_FIELD[k]] || 0;
  }

  return (
    <div className="rag-insight-panel">
      <div className="rag-insight-panel__title">Question outcomes</div>
      <div className="rag-outcome-legend">
        {GATE_KEYS.map((k) => (
          <span key={k} className="rag-outcome-legend__item">
            <span className="rag-dot" style={{ background: GATE_COLOR[k] }} />
            {GATE_LABEL[k]} <span className="rag-mono rag-dim">{fmtPct(dayCount ? totals[k] / dayCount : 0)}</span>
          </span>
        ))}
      </div>
      <div className="rag-outcome-chart">
        <svg viewBox="0 0 300 150" preserveAspectRatio="none" className="rag-outcome-chart__svg">
          <line x1={X0} y1={Y_BOTTOM} x2={X1} y2={Y_BOTTOM} className="rag-chart__axis" />
          <line x1={X0} y1="70" x2={X1} y2="70" className="rag-chart__axis" strokeDasharray="3 3" />
          <line x1={X0} y1={Y_TOP} x2={X1} y2={Y_TOP} className="rag-chart__axis" />
          {bars.map((bar) => (
            <g key={bar.date}>
              {bar.segments.map((s) => (
                <rect
                  key={s.key} x={bar.x.toFixed(1)} y={toY(s.bot).toFixed(1)} width={barW.toFixed(1)}
                  height={(toY(s.top) - toY(s.bot)).toFixed(1)} fill={GATE_COLOR[s.key]} opacity="0.88"
                />
              ))}
            </g>
          ))}
        </svg>
        {/* Plain HTML, not an SVG <text> node — text inside a preserveAspectRatio="none"
            viewBox stretches non-uniformly with the panel's real aspect ratio, distorting
            the glyphs; only the geometric bands above are meant to stretch that way. */}
        {n === 0 && <div className="rag-outcome-chart__empty rag-chart__value">Not enough data yet</div>}
      </div>
    </div>
  );
}

function MostActiveSpaces({ bySpace }) {
  const max = Math.max(...bySpace.map((s) => s.question_count), 1);
  return (
    <div className="rag-insight-panel">
      <div className="rag-insight-panel__title">Most active spaces</div>
      {bySpace.length === 0 ? (
        <p className="rag-hint">No activity yet.</p>
      ) : (
        <div className="rag-active-spaces">
          {bySpace.map((s) => (
            <div key={s.space_id} className="rag-active-spaces__row">
              <div className="rag-active-spaces__bar-wrap">
                <span className="rag-active-spaces__name">{s.space_name}</span>
                <div className="rag-active-spaces__track">
                  <div className="rag-active-spaces__fill" style={{ width: `${Math.round((s.question_count / max) * 100)}%` }} />
                </div>
              </div>
              <span className="rag-mono rag-active-spaces__count">{s.question_count}q</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function UserInsightsView({ userId, onBack }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [rangeDays, setRangeDays] = useState(14);
  const [turnsPage, setTurnsPage] = useState(0);

  function load(days, offset) {
    setError(null);
    const end = new Date();
    const start = new Date(end.getTime() - (days - 1) * 86400000);
    const range = { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
    userInsights(userId, range, { turns_offset: offset * TURNS_PAGE_SIZE, turns_limit: TURNS_PAGE_SIZE })
      .then(setData)
      .catch((err) => setError(err.message));
  }

  useEffect(() => {
    load(rangeDays, 0);
    setTurnsPage(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  function setRange(days) {
    setRangeDays(days);
    setTurnsPage(0);
    load(days, 0);
  }

  function goToPage(p) {
    setTurnsPage(p);
    load(rangeDays, p);
  }

  if (error) return <p className="rag-error" style={{ margin: 24 }}>{error}</p>;
  if (!data) return <p className="rag-hint" style={{ margin: 24 }}>Loading…</p>;

  const { user } = data;
  const tokenTotals = data.tokens_by_day.map((d) => d.prompt_tokens + d.completion_tokens);
  const questionTotals = data.questions_by_day.map((d) => d.total);
  const avgTokenSeries = data.tokens_by_day.map((d, i) => {
    const q = data.questions_by_day[i].total;
    return q ? Math.round((d.prompt_tokens + d.completion_tokens) / q) : 0;
  });
  const cacheHitSeries = data.cache_hit_by_day.map((d) => d.hit_rate);
  const faithfulSeries = data.faithfulness_by_day.map((d) => d.faithful_rate).filter((v) => v != null);
  const faithfulAvg = avg(faithfulSeries);

  const rangeStartLabel = fmtDateShort(data.range_start);
  const rangeEndLabel = fmtDateShort(data.range_end);
  const totalTurnPages = Math.max(1, Math.ceil(data.turns_total / TURNS_PAGE_SIZE));

  return (
    <div className="rag-user-shell">
      <aside className="rag-user-sidebar">
        <div className="rag-user-sidebar__section">
          <button className="rag-space-sidebar__back" onClick={onBack}>
            <span className="rag-space-sidebar__back-icon">←</span>Users
          </button>
          <div className="rag-user-sidebar__identity">
            <span
              className="rag-avatar rag-user-sidebar__avatar"
              style={{ background: AVATAR_BG.accent, color: AVATAR_INK.accent }}
            >
              {user.name.trim().charAt(0).toUpperCase() || "?"}
            </span>
            <div>
              <div className="rag-user-sidebar__name">{user.name}</div>
              <div className="rag-user-sidebar__email">{user.email}</div>
            </div>
            <RagTag tone={user.role === "admin" ? "accent" : "neutral"}>{user.role}</RagTag>
            <div className="rag-user-sidebar__joined">Member since {new Date(user.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}</div>
          </div>
        </div>

        <div className="rag-user-sidebar__section">
          <div className="rag-spaces__nav-label">Snapshot</div>
          <div className="rag-user-sidebar__row">
            <span>Total tokens</span>
            <span className="rag-mono">{data.total_tokens.toLocaleString()}</span>
          </div>
          <div className="rag-user-sidebar__row">
            <span>Questions asked</span>
            <span className="rag-mono">{data.question_count.toLocaleString()}</span>
          </div>
          <div className="rag-user-sidebar__row">
            <span>Avg tokens / q</span>
            <span className="rag-mono">{Math.round(data.avg_tokens_per_question).toLocaleString()}</span>
          </div>
          <div className="rag-user-sidebar__row">
            <span>Cache hit rate</span>
            <span className="rag-mono">{fmtPct(data.cache_hit_rate)}</span>
          </div>
        </div>

        <div className="rag-user-sidebar__section" style={{ borderBottom: "none" }}>
          <div className="rag-spaces__nav-label">Spaces access</div>
          <div className="rag-user-sidebar__chips">
            {data.assigned_spaces.length === 0
              ? <span className="rag-user-sidebar__joined">No spaces assigned yet</span>
              : data.assigned_spaces.map((s) => <span key={s.space_id} className="rag-user-sidebar__chip">{s.space_name}</span>)}
          </div>
        </div>
      </aside>

      <div className="rag-user-main">
        <div className="rag-user-main__container">
          <div className="rag-user-main__header">
            <div>
              <h1 className="rag-user-main__title">Activity overview</h1>
              <p className="rag-hint" style={{ margin: "4px 0 0" }}>{rangeStartLabel} – {rangeEndLabel} · {rangeDays} days</p>
            </div>
            <div className="rag-range-toggle">
              {RANGE_OPTIONS.map((d) => (
                <button
                  key={d}
                  className={`rag-range-toggle__btn${rangeDays === d ? " rag-range-toggle__btn--active" : ""}`}
                  onClick={() => setRange(d)}
                >
                  {d}d
                </button>
              ))}
            </div>
          </div>

          <div className="rag-insight-cards">
            <KpiCard label="Total tokens" value={data.total_tokens.toLocaleString()} series={tokenTotals} color="var(--doc)" />
            <KpiCard label="Questions asked" value={String(data.question_count)} series={questionTotals} color="var(--accent)" />
            <KpiCard
              label="Avg tokens / question"
              value={Math.round(data.avg_tokens_per_question).toLocaleString()}
              series={avgTokenSeries} color="var(--accent2)"
            />
            <KpiCard label="Cache hit rate" value={fmtPct(data.cache_hit_rate)} series={cacheHitSeries} color="var(--accent2)" />
          </div>

          <div className="rag-insight-row rag-user-insight-row--token">
            <TokenUsageTrend
              tokensByDay={data.tokens_by_day} rangeDayCount={data.tokens_by_day.length}
              rangeStartLabel={rangeStartLabel} rangeEndLabel={rangeEndLabel}
            />
            <QuestionOutcomesChart gateOutcomesByDay={data.gate_outcomes_by_day} />
          </div>

          <div className="rag-user-gauge-row">
            <GaugeCard
              label="Cache hit rate" valuePct={data.cache_hit_rate} color="var(--accent2)" softColor="var(--accent2-soft)"
              byDay={data.cache_hit_by_day} field="hit_rate" note={`avg over ${rangeDays} days`}
              noDataLabel="No questions this day"
            />
            <GaugeCard
              label="Faithfulness rate" valuePct={faithfulAvg} color="var(--doc)" softColor="var(--doc-soft)"
              byDay={data.faithfulness_by_day} field="faithful_rate" note="fully-supported claims"
              noDataLabel="No answers with claims to check"
            />
            <MostActiveSpaces bySpace={data.by_space} />
          </div>

          <div className="rag-insight-table">
            <div className="rag-insight-table__title">Recent questions</div>
            <div className="rag-insight-table__head" style={{ gridTemplateColumns: "1.7fr 1fr 100px 110px 70px" }}>
              <span>Question</span><span>Space</span><span>When</span><span>Tokens (in/out)</span><span>Cache</span>
            </div>
            {data.turns.length === 0 && <p className="rag-hint" style={{ padding: 16 }}>No questions in this range.</p>}
            {data.turns.map((t) => (
              <div
                key={`${t.chat_id}-${t.created_at}`} className="rag-insight-table__row"
                style={{ gridTemplateColumns: "1.7fr 1fr 100px 110px 70px", cursor: "default" }}
              >
                <span className="rag-insight-table__truncate">{t.question}</span>
                <span className="rag-dim">{t.space_name}</span>
                <span className="rag-dim">{new Date(t.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}, {new Date(t.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                <span className="rag-mono">{t.prompt_tokens.toLocaleString()} / {t.completion_tokens.toLocaleString()}</span>
                <span className="rag-dim">{t.cache_hit ? "hit" : "—"}</span>
              </div>
            ))}
            {data.turns_total > 0 && (
              <div className="rag-pagination">
                <span className="rag-dim">
                  {turnsPage * TURNS_PAGE_SIZE + 1}–{Math.min((turnsPage + 1) * TURNS_PAGE_SIZE, data.turns_total)} of {data.turns_total}
                </span>
                <div className="rag-pagination__buttons">
                  <button
                    className="rag-btn--ghost rag-pagination__btn" disabled={turnsPage <= 0}
                    onClick={() => goToPage(turnsPage - 1)}
                  >
                    ‹ Prev
                  </button>
                  <span className="rag-dim">Page {turnsPage + 1} of {totalTurnPages}</span>
                  <button
                    className="rag-btn--ghost rag-pagination__btn" disabled={turnsPage + 1 >= totalTurnPages}
                    onClick={() => goToPage(turnsPage + 1)}
                  >
                    Next ›
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
