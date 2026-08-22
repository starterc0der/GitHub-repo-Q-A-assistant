import { Fragment } from "react";
import { BarChart } from "../components/BarChart.jsx";
import { TrendChart } from "../components/TrendChart.jsx";
import { DataTable } from "../components/DataTable.jsx";
import { RagChunkCard, RagEmbedding, RagScorePill, RagSection, RagStageStep, RagTag } from "../components/RagAtoms.jsx";
import {
  RagVectorSpace,
  pointsForCompress,
  pointsForFileSpace,
  pointsForHybrid,
  pointsForRerank,
  pointsForRoute,
} from "../components/RagVectorSpace.jsx";

export const RETRIEVAL_STAGES = [
  { id: "q-embed", title: "Embed question" },
  { id: "route", title: "Route to files" },
  { id: "hybrid", title: "Hybrid search" },
  { id: "rerank", title: "Cross-rerank" },
  { id: "compress", title: "Compress" },
  { id: "prompt", title: "Final prompt" },
  { id: "answer", title: "LLM answer" },
];

// A live-data or report question (see Pipeline._try_live_data_answer /
// _try_report_answer) skips rerank/compress entirely and fetches from Redis/Postgres
// instead — its own stage in the rail, shown only when one of those paths actually ran,
// right before "Final prompt" since the fetched data is what the prompt is built from.
export function retrievalStagesFor(data) {
  if (!data.live_data_tool && !data.report_tool) return RETRIEVAL_STAGES;
  const stages = [...RETRIEVAL_STAGES];
  stages.splice(5, 0, { id: "tool", title: "Tool called" });
  return stages;
}

function splitSuffix(data) {
  return data.sub_questions?.length ? ` · from ${data.sub_questions.length} sub-qs` : "";
}

// stage id -> the timings/tokens key(s) it corresponds to, and whether that key is ever
// a real LLM call (vs. always-local computation) — see STAGE_IS_API on the backend
// (src/api/insights_routes.py) for the matching source of truth.
const STAGE_TIMING_KEY = {
  "q-embed": "embed", route: "route", hybrid: "hybrid", rerank: "rerank",
  compress: "compress", answer: "generate",
};
const TIMING_IS_API = { embed: false, route: false, hybrid: false, rerank: false, compress: true, generate: true };

function fmtMs(ms) {
  if (ms == null) return null;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function timingSuffix(data, id) {
  const key = STAGE_TIMING_KEY[id];
  const ms = key && data.timings?.[key];
  if (ms == null) return "";
  return ` · ${fmtMs(ms)}${TIMING_IS_API[key] ? "" : " (local)"}`;
}

function retrievalStageMeta(data, id) {
  switch (id) {
    case "q-embed":
      return `${data.query_embedding.dim}d vector${splitSuffix(data)}${timingSuffix(data, id)}`;
    case "route":
      return `${data.routed_files.length} files${splitSuffix(data)}${timingSuffix(data, id)}`;
    case "hybrid":
      return `${data.candidates.length} candidates${splitSuffix(data)}${timingSuffix(data, id)}`;
    case "rerank":
      return data.reranked.length === 0 && data.candidates.length > 0
        ? `0 of ${data.candidates.length} — no answer found${timingSuffix(data, id)}`
        : `${data.reranked.length} reranked${splitSuffix(data)}${timingSuffix(data, id)}`;
    case "compress":
      return data.final_chunks.length === 0
        ? "nothing to compress"
        : `${data.final_chunks.length} final chunks${timingSuffix(data, id)}`;
    case "tool": {
      if (data.live_data_tool) {
        const t = data.live_data_tool;
        const places = t.matched_places.length;
        const keys = t.redis_keys.length;
        return `${places} place${places === 1 ? "" : "s"} · ${keys} key${keys === 1 ? "" : "s"} fetched`;
      }
      if (data.report_tool) {
        const t = data.report_tool;
        const places = t.matched_places.length;
        return `${places} place${places === 1 ? "" : "s"} · ${t.metric}/${t.granularity} · ${t.start_date}${t.start_date === t.end_date ? "" : ` to ${t.end_date}`}`;
      }
      return null;
    }
    case "prompt":
      return `${data.final_prompt.length} chars`;
    case "answer": {
      if (!data.answer) return null;
      const unsupported = (data.answer.citations || []).filter((c) => c.chunk_ids.length === 0).length;
      const flag = unsupported > 0 ? ` · ${unsupported} unsupported claim${unsupported > 1 ? "s" : ""}` : "";
      return (data.answer.error ? "failed" : "answered") + flag + timingSuffix(data, id);
    }
    default:
      return null;
  }
}

// An empty stage is ambiguous — it reads as "the pipeline broke", not "the filter did its
// job". Every stage that can legitimately end up with nothing says so explicitly instead.
function RagEmptyStage({ title, children }) {
  return (
    <div className="rag-empty">
      <span className="rag-empty__title">{title}</span>
      <div className="rag-empty__body">{children}</div>
    </div>
  );
}

function AnswerText({ text }) {
  return <p className="rag-answer__text">{text}</p>;
}

export function RetrievalStageList({ data, statuses, current, onSelect }) {
  const stages = retrievalStagesFor(data);
  return (
    <div className="rag-steps">
      {stages.map((s, i) => (
        <RagStageStep
          key={s.id}
          num={i + 1}
          title={s.title}
          meta={retrievalStageMeta(data, s.id)}
          status={statuses[s.id]}
          current={current === s.id}
          onClick={() => onSelect(s.id)}
          isLast={i === stages.length - 1}
        />
      ))}
    </div>
  );
}

export function RetrievalSections({ data, visible }) {
  // Maps each sub-question to a short "Q1"/"Q2" (or "Hop 1"/"Hop 2") label — only
  // non-empty when the question was actually decomposed, so route/hybrid/rerank rows
  // can show which part produced them without repeating the full text on every row.
  const subQIndex = new Map((data.sub_questions || []).map((q, i) => [q, i + 1]));
  const isSequential = data.decompose_mode === "sequential";
  const hopLabel = (n) => (isSequential ? `Hop ${n}` : `Q${n}`);
  const chunkById = new Map((data.final_chunks || []).map((f) => [f.chunk.id, f.chunk]));
  // Citations on a live-data answer point at a synthetic chunk (see
  // Pipeline._try_live_data_answer) wrapping the tool's fetched-readings context —
  // it never went through retrieval, so it's not in final_chunks above. Without this,
  // a fully-verified live-data claim rendered with no source tag at all, indistinguishable
  // from an unsupported one, which read as "no tool was called" even though it was.
  if (data.live_data_tool) {
    const id = `live-data:${data.space_id}`;
    chunkById.set(id, {
      id, space_id: data.space_id, source_id: "", file_path: "Live data tool (Redis)",
      language: "text", symbol_name: null, start_line: 1,
      end_line: data.live_data_tool.context.split("\n").length,
      code: data.live_data_tool.context, context_header: "",
    });
  }

  return (
    <Fragment>
      {visible["q-embed"] && (
        <RagSection
          id="q-embed"
          num="1"
          title="Embed question"
          description="The question is embedded with the same model used for chunks and file summaries."
          plain="Turn the question into that same kind of number-list, so it can be compared to every file and chunk on equal footing."
        >
          <div className="rag-callout">
            <span className="rag-callout__label">question</span>
            <p className="rag-mono">&ldquo;{data.question}&rdquo;</p>
            {(data.is_broad || data.wants_chart) && (
              <div style={{ marginTop: 6, display: "flex", gap: 6 }}>
                {data.is_broad && (
                  <RagTag tone="accent" title="Classified upfront as needing the wide-source path — hybrid search and rerank were skipped entirely.">
                    broad — skipped rerank
                  </RagTag>
                )}
                {data.wants_chart && (
                  <RagTag tone="accent" title="Classified upfront as asking for a comparison/graph — the answer prompt got an explicit chart hint.">
                    wants chart
                  </RagTag>
                )}
              </div>
            )}
          </div>
          {subQIndex.size > 0 && (
            <div className="rag-callout">
              <span className="rag-callout__label">
                {isSequential
                  ? `split into ${subQIndex.size} chained hops — each hop's query is built from the previous hop's own retrieval result before it runs`
                  : `split into ${subQIndex.size} independent sub-questions — each retrieved in parallel, merged before compression`}
              </span>
              <ul className="rag-ranked">
                {data.sub_questions.map((sq, i) => {
                  const noEvidence = (data.insufficient_sub_questions || []).includes(sq);
                  const retriedAs = (data.retried_sub_questions || {})[sq];
                  const recovered = retriedAs && !noEvidence;
                  return (
                    <li key={i} style={{ flexDirection: "column", alignItems: "stretch" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span className="rag-ranked__rank">
                          {isSequential ? `Hop ${i + 1}` : `Q${i + 1}`}
                        </span>
                        <span className="rag-mono rag-ranked__path">{sq}</span>
                        {noEvidence && (
                          <RagTag tone="warn">
                            {retriedAs ? "no evidence found (after retry)" : "no evidence found"}
                          </RagTag>
                        )}
                        {recovered && <RagTag tone="accent2">recovered via retry</RagTag>}
                      </div>
                      {retriedAs && (
                        <p className="rag-hint" style={{ margin: "4px 0 0 30px" }}>
                          Retried as: &ldquo;{retriedAs}&rdquo;
                        </p>
                      )}
                    </li>
                  );
                })}
              </ul>
              {data.sufficiency === "partial" && (
                <p className="rag-hint" style={{ margin: "10px 0 0" }}>
                  A real answer was still generated from what WAS found — the model was told
                  explicitly which part had no evidence, so it can state that gap directly
                  instead of guessing.
                </p>
              )}
            </div>
          )}
          <RagEmbedding embedding={data.query_embedding} label="query vector" />
          <RagVectorSpace
            {...pointsForFileSpace(data)}
            legend={[{ tone: "neutral", label: "file summary" }, { tone: "accent", label: "query" }]}
            caption="The question lands in the same space as every file summary in the repo."
          />
        </RagSection>
      )}

      {visible.route && (
        <RagSection
          id="route"
          num="2"
          title="Route to files"
          description="Doc-index search over file summaries narrows the whole repo down to a shortlist before any chunk search runs."
          plain="Figure out which files are even worth opening, before reading every single piece of code in the repo."
        >
          <RagVectorSpace
            {...pointsForRoute(data)}
            legend={[{ tone: "accent2", label: "routed file" }, { tone: "dim", label: "not routed" }]}
            caption="The closest file summaries are shortlisted — chunk search will only run inside them."
          />
          <ul className="rag-ranked">
            {data.routed_files.map((f, i) => (
              <li key={f.file_path}>
                <span className="rag-ranked__rank">{i + 1}</span>
                <span className="rag-mono rag-ranked__path">{f.file_path}</span>
                {subQIndex.has(f.source_question) && (
                  <RagTag tone="dim" title={f.source_question}>
                    {hopLabel(subQIndex.get(f.source_question))}
                  </RagTag>
                )}
                <RagScorePill label="score" value={f.score} tone="accent2" />
              </li>
            ))}
          </ul>
        </RagSection>
      )}

      {visible.hybrid && (
        <RagSection
          id="hybrid"
          num="3"
          title="Hybrid search candidates"
          description="Dense cosine + BM25, fused 4:1, scoped to the routed files above."
          plain="Find the closest-meaning pieces (via the number-lists) and also the pieces sharing exact keywords — then blend both rankings together."
        >
          <RagVectorSpace
            {...pointsForHybrid(data)}
            legend={[{ tone: "accent", label: "candidate chunk" }, { tone: "dim", label: "other chunk" }]}
            caption="Line opacity tracks the fused dense+BM25 score — not just raw distance."
          />
          {data.candidates.map((c) => (
            <RagChunkCard
              key={c.chunk.id}
              chunk={c.chunk}
              scores={[
                ["dense", c.dense_score],
                ["bm25", c.bm25_score],
                ["fused", c.fused_score, "accent"],
              ]}
              badge={
                subQIndex.has(c.source_question) && (
                  <RagTag tone="dim" title={c.source_question}>
                    {hopLabel(subQIndex.get(c.source_question))}
                  </RagTag>
                )
              }
            />
          ))}
        </RagSection>
      )}

      {visible.rerank && (
        <RagSection
          id="rerank"
          num="4"
          title="Cross-encoder reranked"
          description="Joint (question, chunk) attention re-scores and re-orders the candidates above."
          plain="Take a second, closer look at the top matches — reading the question and each piece together, side by side, instead of comparing number-lists from a distance."
        >
          <RagVectorSpace
            {...pointsForRerank(data)}
            legend={[{ tone: "accent2", label: "reranked · size = score" }, { tone: "dim", label: "not selected" }]}
            caption="Cross-encoder reordering doesn't always match raw distance — joint attention can promote a chunk the embedding space placed further away."
          />
          {data.reranked.length === 0 ? (
            <RagEmptyStage title={`No answer found among ${data.candidates.length} candidates`}>
              <p>
                Even the best-scoring candidate fell below <strong>{data.rerank_min_top_score}</strong>,
                so the whole list was rejected — nothing in this repo answers the question.
              </p>
              <p>
                This is the only stage that can say "nothing here". The fused score above is
                <em> relative</em> — min-max normalized within the candidate pool, so the best
                candidate always scores near the top even when the whole pool is irrelevant.
                The cross-encoder scores each pair on its own merits, so its number is absolute:
                a 0.0 means genuinely unrelated, not merely "worst of these".
              </p>
              <p>
                Only the <em>top</em> score is checked, not each chunk's. A chunk that supports
                an answer without containing it scores low but is still worth sending — so once
                the best chunk clears the gate, the whole top-k is kept.
              </p>
            </RagEmptyStage>
          ) : (
            data.reranked.map((r) => (
              <RagChunkCard
                key={r.chunk.id}
                chunk={r.chunk}
                scores={[["rerank", r.rerank_score, "accent"]]}
                badge={
                  <>
                    {subQIndex.has(r.source_question) && (
                      <RagTag tone="dim" title={r.source_question}>
                        {hopLabel(subQIndex.get(r.source_question))}
                      </RagTag>
                    )}
                    {r.diversity_pick && (
                      <RagTag
                        tone="accent2"
                        title="MMR picked this chunk over a higher-scoring alternative because it covers different ground — not just relevance order."
                      >
                        picked for diversity
                      </RagTag>
                    )}
                  </>
                }
              />
            ))
          )}
        </RagSection>
      )}

      {visible.compress && (
        <RagSection
          id="compress"
          num="5"
          title="Compress"
          description="The LLM picks question-relevant line numbers per chunk; everything outside that window is dropped before the prompt is built."
          plain="Trim each match down to just the lines that actually answer the question — no need to read a whole function to use three lines of it."
        >
          <RagVectorSpace
            {...pointsForCompress(data)}
            legend={[
              { tone: "accent2", label: "kept" },
              { tone: "warn", label: "dropped" },
              { tone: "dim", label: "not selected" },
            ]}
            caption="Only the kept chunks below continue on to the final prompt."
          />
          {data.final_chunks.length === 0 && (
            <RagEmptyStage title="Nothing reached compression">
              <p>
                Reranking kept no chunks, so there was nothing to trim. Compression only ever
                narrows what it is given — it never adds chunks back.
              </p>
            </RagEmptyStage>
          )}
          {data.final_chunks.length > 0 && (() => {
            const totalOriginal = data.final_chunks.reduce((sum, f) => sum + (f.original_tokens || 0), 0);
            const totalCompressed = data.final_chunks.reduce((sum, f) => sum + (f.compressed_tokens || 0), 0);
            if (!totalOriginal) return null;
            const pct = Math.round((1 - totalCompressed / totalOriginal) * 100);
            return (
              <p className="rag-hint">
                {totalOriginal.toLocaleString()} → {totalCompressed.toLocaleString()} tokens across{" "}
                {data.final_chunks.length} chunk{data.final_chunks.length === 1 ? "" : "s"}
                {pct > 0 ? ` · ${pct}% reduction` : ""}
              </p>
            );
          })()}
          {data.final_chunks.map((f, i) => (
            <RagChunkCard
              key={f.chunk.id + i}
              chunk={f.chunk}
              badge={
                f.dropped ? (
                  <RagTag tone="warn">dropped</RagTag>
                ) : (
                  <RagTag tone="accent2">
                    {f.original_line_count} → {f.compressed_line_count} lines
                    {f.original_tokens > 0 && ` · ${f.original_tokens} → ${f.compressed_tokens} tokens`}
                  </RagTag>
                )
              }
            />
          ))}
        </RagSection>
      )}

      {visible.tool && data.live_data_tool && (
        <RagSection
          id="tool"
          num="5b"
          title="Tool called"
          description="Rerank/compress were skipped — this question resolved to a place/device match, and its live reading was fetched from Redis instead."
          plain="Look up which place(s) the question is about, then pull that place's current sensor reading straight from Redis — no chunk ranking needed once the place is known."
        >
          <div className="rag-callout">
            <span className="rag-callout__label">matched place(s)</span>
            <p className="rag-mono">{data.live_data_tool.matched_places.join(", ")}</p>
          </div>
          <p className="rag-hint">Redis keys fetched, and exactly what came back:</p>
          <ul className="rag-ranked">
            {data.live_data_tool.redis_keys.map((key) => (
              <li key={key} style={{ flexDirection: "column", alignItems: "stretch" }}>
                <span className="rag-mono rag-ranked__path">{key}</span>
                <pre className="rag-prompt" style={{ marginTop: 6 }}>
                  {JSON.stringify(data.live_data_tool.readings[key], null, 2) ?? "null (key not found)"}
                </pre>
              </li>
            ))}
          </ul>
          <p className="rag-hint">Narrated context built from the readings above, sent to the LLM:</p>
          <div className="rag-prompt">
            <pre>{data.live_data_tool.context}</pre>
          </div>
        </RagSection>
      )}

      {visible.tool && data.report_tool && (
        <RagSection
          id="tool"
          num="5b"
          title="Tool called"
          description="Rerank/compress were skipped — this question resolved to a place match and a time window, and its historical reading was fetched from Postgres (watco_stream_db) instead."
          plain="Look up which place(s) the question is about and which time window it covers, then pull that window's report data straight from the database — no chunk ranking needed once both are known."
        >
          <div className="rag-callout">
            <span className="rag-callout__label">matched place(s)</span>
            <p className="rag-mono">{data.report_tool.matched_places.join(", ")}</p>
          </div>
          <div className="rag-callout" style={{ marginTop: 10 }}>
            <span className="rag-callout__label">resolved window</span>
            <p className="rag-mono">
              {data.report_tool.metric} · {data.report_tool.granularity} ·{" "}
              {data.report_tool.start_date}
              {data.report_tool.start_date === data.report_tool.end_date ? "" : ` to ${data.report_tool.end_date}`}
            </p>
          </div>
          <p className="rag-hint">Narrated context built from the fetched data, sent to the LLM:</p>
          <div className="rag-prompt">
            <pre>{data.report_tool.context}</pre>
          </div>
        </RagSection>
      )}

      {visible.prompt && (
        <RagSection
          id="prompt"
          num="6"
          title="Final prompt"
          description="Exactly what gets sent to the answer model — system prompt, conversation history, and user prompt, verbatim."
          plain="Assemble everything above into the exact instructions — system prompt, prior chat turns, trimmed matches, and your question — that get handed to the LLM to write the final answer. This is the real payload, shown character for character."
        >
          {/* pre, not p: the system prompt is a numbered rule list and its line breaks
              are meaningful — collapsing them makes it unreadable. */}
          <p className="rag-hint">System prompt, verbatim:</p>
          <div className="rag-prompt">
            <pre>{data.system_prompt}</pre>
          </div>
          {data.history.length > 0 && (
            <>
              <p className="rag-hint">
                Conversation history sent as real chat turns (not folded into the prompt text below):
              </p>
              <div className="rag-meta-trace">
                {data.history.map((m, i) => (
                  <div key={i} className={`rag-meta-trace__turn rag-meta-trace__turn--${m.role}`}>
                    <span className="rag-meta-trace__role">{m.role}</span>
                    <span className="rag-meta-trace__text">{m.content}</span>
                  </div>
                ))}
              </div>
            </>
          )}
          <div className="rag-stat-row">
            <div className="rag-stat rag-stat--accent">
              <span className="rag-stat__n">
                {data.live_data_tool
                  ? data.live_data_tool.redis_keys.length
                  : data.report_tool
                    ? data.report_tool.matched_places.length
                    : data.final_chunks.length}
              </span>
              <span className="rag-stat__l">
                {data.live_data_tool ? "live keys in prompt" : data.report_tool ? "places in prompt" : "chunks in prompt"}
              </span>
            </div>
            <div className="rag-stat">
              <span className="rag-stat__n">{data.final_prompt.length}</span>
              <span className="rag-stat__l">characters</span>
            </div>
          </div>
          {data.final_chunks.length === 0 && !data.live_data_tool && !data.report_tool && (
            <RagEmptyStage title="No prompt was sent">
              <p>
                With no chunks to ground an answer, the pipeline short-circuits before the
                generation call — see <code>Pipeline.NO_MATCH</code>. The empty prompt below is
                what <em>would</em> have been built; it was never sent to the model.
              </p>
            </RagEmptyStage>
          )}
          <p className="rag-hint">Full user prompt, verbatim:</p>
          <div className="rag-prompt">
            <pre>{data.final_prompt}</pre>
          </div>
        </RagSection>
      )}

      {visible.answer && data.answer && (
        <RagSection
          id="answer"
          num="7"
          title="LLM answer"
          description="The prompt above, sent to the answer model, produces this answer."
          plain="The AI reads the trimmed code and writes the answer in plain prose."
        >
          {data.answer.error ? (
            <p className="rag-error">{data.answer.error}</p>
          ) : (
            <div className="rag-answer">
              <span className="rag-callout__label">answer</span>
              {data.answer.text && <AnswerText text={data.answer.text} />}
              {data.answer.table && <DataTable table={data.answer.table} />}
              {data.answer.chart && (data.answer.chart.kind === "trend" ? <TrendChart chart={data.answer.chart} /> : <BarChart chart={data.answer.chart} />)}
              {!data.answer.text && !data.answer.table && !data.answer.chart && (
                <p className="rag-hint">(empty — nothing else to show)</p>
              )}
              <span className="rag-answer__model">{data.answer.model}</span>
            </div>
          )}
          {data.answer.citations?.length > 0 && (
            <div className="rag-callout" style={{ marginTop: 12 }}>
              <span className="rag-callout__label">
                claim-level citations & verification — computed after generation, not
                self-reported by the model; never shown in chat itself
              </span>
              <p className="rag-hint" style={{ margin: "4px 0 8px" }}>
                {data.answer.citations.filter((c) => c.chunk_ids.length > 0).length} of{" "}
                {data.answer.citations.length} claims verified against a source chunk
                {data.answer.citations.some((c) => c.chunk_ids.length === 0) &&
                  " — unsupported claims flagged below"}
              </p>
              {data.answer.citations.map((c, i) => {
                const evidenceChunk = c.evidence_chunk_id && chunkById.get(c.evidence_chunk_id);
                return (
                  <div className="rag-citation" key={i}>
                    <div className="rag-citation__head">
                      <span className="rag-mono">{c.claim}</span>
                      {c.chunk_ids.length > 0 ? (
                        c.chunk_ids.map((id) => {
                          const chunk = chunkById.get(id);
                          return (
                            chunk && (
                              <RagTag key={id} tone={id === c.evidence_chunk_id ? "accent2" : "dim"}>
                                {chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}
                              </RagTag>
                            )
                          );
                        })
                      ) : (
                        <RagTag tone="warn">unsupported</RagTag>
                      )}
                    </div>
                    {evidenceChunk ? (
                      <RagChunkCard chunk={evidenceChunk} highlightLine={c.evidence_line} />
                    ) : (
                      c.evidence && (
                        <p className="rag-hint" style={{ margin: "4px 0 0" }}>
                          Evidence: &ldquo;{c.evidence}&rdquo;
                        </p>
                      )
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </RagSection>
      )}
    </Fragment>
  );
}
