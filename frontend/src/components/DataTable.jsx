// Renders a {columns, rows} spec from the answer model's ```table block (see
// src/connectors/live_data.py's TableParser) — the live-data answer path's default
// format. Same "no library, plain elements" approach as BarChart.

export function DataTable({ table }) {
  const { columns, rows } = table;
  return (
    <div className="rag-data-table__scroll">
      <table className="rag-data-table">
        <thead>
          <tr>
            {columns.map((c, i) => (
              <th key={i}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => (
                <td key={ci}>{cell === "" ? "—" : String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
