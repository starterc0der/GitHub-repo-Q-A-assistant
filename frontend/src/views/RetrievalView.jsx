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
];

function retrievalStageMeta(data, id) {
  switch (id) {
    case "q-embed":
      return `${data.query_embedding.dim}d vector`;
    case "route":
      return `${data.routed_files.length} files`;
    case "hybrid":
      return `${data.candidates.length} candidates`;
    case "rerank":
      return `${data.reranked.length} reranked`;
    case "compress":
      return `${data.final_chunks.length} final chunks`;
    case "prompt":
      return `${data.final_prompt.length} chars`;
    default:
      return null;
  }
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
          {data.reranked.map((r) => (
            <RagChunkCard key={r.chunk.id} chunk={r.chunk} scores={[["rerank", r.rerank_score, "accent"]]} />
          ))}
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
          description="Nothing gets sent to an LLM here — this is exactly the system prompt and user prompt that would be, verbatim."
          plain="Assemble everything above into the exact instructions — system prompt, trimmed matches, and your question — that would be handed to an LLM to write the final answer. Shown here as-is; no LLM is actually called."
        >
          <div className="rag-callout">
            <span className="rag-callout__label">system prompt</span>
            <p>{data.system_prompt}</p>
          </div>
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
          <p className="rag-hint">Full user prompt, verbatim:</p>
          <div className="rag-prompt">
            <pre>{data.final_prompt}</pre>
          </div>
        </RagSection>
      )}
    </Fragment>
  );
}
