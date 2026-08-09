// Hand-rolled bar SVG — same "no charting library" approach as RagVectorSpace. Renders a
// {title, categories, series: [{name, values}]} spec from the answer model's ```chart
// block (see src/generate/answer.py's ChartParser).
//
// Two layouts: vertical grouped bars for a few short categories (e.g. comparing two
// products across a handful of specs), horizontal rows past MANY_THRESHOLD categories —
// long/many labels never fit a fixed-width x-axis without colliding, but a label to the
// left of its own row always reads regardless of how many rows there are.

const COLORS = ["var(--accent)", "var(--accent2)", "var(--warn)"];
const WIDTH = 480;
const MANY_THRESHOLD = 8;

function truncate(text, max) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
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
                    {value}
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
  const BAND_H = 22;
  const barsAreaW = WIDTH - LABEL_W - VALUE_GUTTER;
  const seriesH = Math.max(5, Math.floor((BAND_H - 4) / series.length));
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
              const y = bandY + si * (seriesH + 1);
              return (
                <g key={s.name}>
                  <rect x={LABEL_W} y={y} width={barW} height={seriesH} fill={COLORS[si % COLORS.length]} rx="2" />
                  <text x={LABEL_W + barW + 4} y={y + seriesH / 2 + 3} className="rag-chart__value">
                    {value}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

export function BarChart({ chart }) {
  const { title, categories, series } = chart;
  const maxValue = Math.max(1, ...series.flatMap((s) => s.values));
  const horizontal = categories.length > MANY_THRESHOLD;

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
