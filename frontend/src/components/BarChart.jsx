// Hand-rolled bar SVG — same "no charting library" approach as RagVectorSpace. Renders a
// {title, categories, series: [{name, values}]} spec from the answer model's ```chart
// block (see src/generate/answer.py's ChartParser).
//
// Two layouts: vertical grouped bars for a few short categories (e.g. comparing two
// products across a handful of specs), horizontal rows past MANY_THRESHOLD categories —
// long/many labels never fit a fixed-width x-axis without colliding, but a label to the
// left of its own row always reads regardless of how many rows there are.

// Same expanded 6-color palette as TrendChart.jsx (see its comment) — a comparison
// chart routinely has 5+ series, and 3 cycling colors made distinct series
// indistinguishable.
const COLORS = [
  "var(--accent)", "var(--accent2)", "var(--warn)", "var(--doc)",
  "oklch(62% 0.16 300)", "oklch(55% 0.13 125)",
];
const WIDTH = 480;
const MANY_THRESHOLD = 8;

function truncate(text, max) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

// Raw DB floats routinely carry 10+ decimal digits (e.g. 2.7447193877551) — rendered
// verbatim, a label that long overlaps its neighbors regardless of bar spacing. Round
// for display only; the underlying data/tooltip-free chart never needed the precision.
function fmtValue(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return value;
  const rounded = Math.round(value * 100) / 100;
  return rounded.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function VerticalBars({ categories, series, maxValue }) {
  const HEIGHT = 200;
  const PAD = { top: 22, right: 12, bottom: 34, left: 12 };
  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const groupW = plotW / categories.length;
  const barW = Math.min(24, (groupW * 0.7) / series.length);
  const groupPad = (groupW - barW * series.length) / 2;

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="rag-chart__svg">
      <line
        x1={PAD.left} y1={PAD.top + plotH} x2={WIDTH - PAD.right} y2={PAD.top + plotH}
        className="rag-chart__axis"
      />
      {categories.map((cat, ci) => {
        const groupX = PAD.left + ci * groupW + groupPad;
        return (
          <g key={cat}>
            {series.map((s, si) => {
              const value = s.values[ci];
              const barH = (value / maxValue) * plotH;
              const x = groupX + si * barW;
              const y = PAD.top + plotH - barH;
              return (
                <g key={s.name}>
                  <rect x={x + 1} y={y} width={barW - 2} height={barH} fill={COLORS[si % COLORS.length]} rx="2" />
                  <text x={x + barW / 2} y={y - 4} textAnchor="middle" className="rag-chart__value">
                    {fmtValue(value)}
                  </text>
                </g>
              );
            })}
            <text
              x={groupX + (barW * series.length) / 2}
              y={HEIGHT - PAD.bottom + 16}
              textAnchor="middle"
              className="rag-chart__cat"
            >
              {truncate(cat, 12)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function HorizontalBars({ categories, series, maxValue }) {
  const LABEL_W = 140;
  const VALUE_GUTTER = 44;
  const PAD_TOP = 6;
  const BAND_GAP = 6;
  // Fixed at a 22px band regardless of series count, this squeezed N stacked bars
  // (and their value labels) into a space that only comfortably fits ~2 — a 9px label
  // next to a 5px-tall bar collides with its neighbors above/below well before N gets
  // large. Each series now gets a fixed, label-sized row instead of a shrinking slice
  // of a fixed band; the chart grows taller (scrolls via .rag-chart__scroll) rather
  // than the content getting correspondingly less readable.
  const SERIES_H = 13;
  const BAND_H = series.length * SERIES_H + BAND_GAP;
  const barsAreaW = WIDTH - LABEL_W - VALUE_GUTTER;
  const height = PAD_TOP + categories.length * BAND_H + 4;

  return (
    <svg viewBox={`0 0 ${WIDTH} ${height}`} className="rag-chart__svg">
      {categories.map((cat, ci) => {
        const bandY = PAD_TOP + ci * BAND_H;
        return (
          <g key={cat}>
            <text x={LABEL_W - 8} y={bandY + BAND_H / 2 + 3} textAnchor="end" className="rag-chart__cat">
              {truncate(cat, 20)}
            </text>
            {series.map((s, si) => {
              const value = s.values[ci];
              const barW = Math.max(1, (value / maxValue) * barsAreaW);
              const y = bandY + si * SERIES_H;
              return (
                <g key={s.name}>
                  <rect x={LABEL_W} y={y} width={barW} height={SERIES_H - 2} fill={COLORS[si % COLORS.length]} rx="2" />
                  <text x={LABEL_W + barW + 4} y={y + SERIES_H / 2 + 2} className="rag-chart__value">
                    {fmtValue(value)}
                  </text>
                </g>
              );
            })}
            {series.length > 1 && ci < categories.length - 1 && (
              <line
                x1={0} y1={bandY + BAND_H - 1} x2={WIDTH} y2={bandY + BAND_H - 1}
                className="rag-chart__axis" opacity="0.5"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}

export function BarChart({ chart }) {
  const { title, categories, series } = chart;
  const maxValue = Math.max(1, ...series.flatMap((s) => s.values));
  // Many categories never fit a fixed-width x-axis; many series per group never fit a
  // bar narrow enough to hold a value label without it overlapping its neighbors —
  // both cases route to the same horizontal layout, where each bar's label sits
  // outside it instead of squeezed on top.
  const horizontal = categories.length > MANY_THRESHOLD || series.length > 2;

  return (
    <div className="rag-chart">
      {title && <div className="rag-chart__title">{title}</div>}
      {horizontal ? (
        <div className="rag-chart__scroll">
          <HorizontalBars categories={categories} series={series} maxValue={maxValue} />
        </div>
      ) : (
        <VerticalBars categories={categories} series={series} maxValue={maxValue} />
      )}
      {/* A single series needs no legend box — the title already names what's plotted. */}
      {series.length > 1 && (
        <div className="rag-chart__legend">
          {series.map((s, i) => (
            <span key={s.name} className="rag-chart__legend-item">
              <span className="rag-chart__swatch" style={{ background: COLORS[i % COLORS.length] }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
