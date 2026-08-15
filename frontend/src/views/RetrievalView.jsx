import { Fragment } from "react";
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
    case "prompt":
      return `${data.final_prompt.length} chars`;
    case "answer":
      if (!data.answer) return null;
      return (data.answer.error ? "failed" : "answered") + timingSuffix(data, id);
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
  return (
    <div className="rag-steps">
      {RETRIEVAL_STAGES.map((s, i) => (
        <RagStageStep
          key={s.id}
          num={i + 1}
          title={s.title}
          meta={retrievalStageMeta(data, s.id)}
          status={statuses[s.id]}
          current={current === s.id}
          onClick={() => onSelect(s.id)}
          isLast={i === RETRIEVAL_STAGES.length - 1}
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
                    <li key={i}>
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
                      {retriedAs && (
                        <p className="rag-hint" style={{ margin: "4px 0 0" }}>
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
                  subQIndex.has(r.source_question) && (
                    <RagTag tone="dim" title={r.source_question}>
                      {hopLabel(subQIndex.get(r.source_question))}
                    </RagTag>
                  )
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
                  </RagTag>
                )
              }
            />
          ))}
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
              <span className="rag-stat__n">{data.final_chunks.length}</span>
              <span className="rag-stat__l">chunks in prompt</span>
            </div>
            <div className="rag-stat">
              <span className="rag-stat__n">{data.final_prompt.length}</span>
              <span className="rag-stat__l">characters</span>
            </div>
          </div>
          {data.final_chunks.length === 0 && (
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
              <AnswerText text={data.answer.text} />
              <span className="rag-answer__model">{data.answer.model}</span>
            </div>
          )}
        </RagSection>
      )}
    </Fragment>
  );
}
