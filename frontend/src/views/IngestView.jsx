import { Fragment, useState } from "react";
import { RagChunkCard, RagEmbedding, RagSection, RagStageStep, RagTag } from "../components/RagAtoms.jsx";
import { RagVectorSpace, pointsForIngestChunks } from "../components/RagVectorSpace.jsx";

const REPO_STAGES = [
  { id: "clone", title: "Clone repo" },
  { id: "walk", title: "Walk files" },
  { id: "chunk", title: "AST-chunk" },
  { id: "contextualize", title: "Summarize + contextualize" },
  { id: "embed", title: "Embed + upsert" },
];

// PDF/docx/pasted-text sources have no clone/walk step and no LLM summary or context
// header (see Contextualizer's prose fast-path) — same tail, shorter head.
const PROSE_STAGES = [
  { id: "chunk", title: "Chunk" },
  { id: "contextualize", title: "Template summaries" },
  { id: "embed", title: "Embed + upsert" },
];

export function ingestStagesFor(kind) {
  return kind === "repo" ? REPO_STAGES : PROSE_STAGES;
}

const MAX_FILES_SHOWN = 8;

// Big repos render hundreds of files; showing them all at once is a slow, unscrollable
// wall. Default to a slice, but let the whole set be expanded — the summary and context
// header for *every* file are in the payload, so nothing here is a fetch.
function useFileSlice(files) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? files : files.slice(0, MAX_FILES_SHOWN);
  const hidden = files.length - shown.length;
  const toggle =
    files.length <= MAX_FILES_SHOWN ? null : (
      <button className="rag-more-btn" onClick={() => setExpanded(!expanded)}>
        {expanded ? `Show first ${MAX_FILES_SHOWN} files only` : `Show all ${files.length} files`}
      </button>
    );
  return { shown, hidden, toggle };
}

function totalChunks(data) {
  return data.files.reduce((n, f) => n + f.chunks.length, 0);
}

function ingestStageMeta(data, id) {
  switch (id) {
    case "clone":
      return data.name;
    case "walk":
      return `${data.walk.kept} files kept`;
    case "chunk":
      return `${totalChunks(data)} chunks`;
    case "contextualize":
      return `${data.files.length} summaries`;
    case "embed":
      return `${totalChunks(data) + data.files.length} vectors`;
    default:
      return null;
  }
}

export function IngestStageList({ data, statuses, current, onSelect }) {
  const stages = ingestStagesFor(data.kind);
  return (
    <div className="rag-steps">
      {stages.map((s, i) => (
        <RagStageStep
          key={s.id}
          num={i + 1}
          title={s.title}
          meta={ingestStageMeta(data, s.id)}
          status={statuses[s.id]}
          current={current === s.id}
          onClick={() => onSelect(s.id)}
          isLast={i === stages.length - 1}
        />
      ))}
    </div>
  );
}

export function IngestSections({ data, visible }) {
  const { shown, hidden: hiddenCount, toggle } = useFileSlice(data.files);
  const stageNum = (id) => ingestStagesFor(data.kind).findIndex((s) => s.id === id) + 1;

  return (
    <Fragment>
      {visible.clone && data.clone && (
        <RagSection
          id="clone"
          num={stageNum("clone")}
          title="Clone repo"
          description="git clone --depth 1 — shallow, no history."
          plain="Like copying a folder of code onto your computer so you can look through it — without downloading its entire history."
        >
          <div className="rag-kv">
            <div><span className="rag-kv__k">source</span><span className="rag-kv__v rag-mono">{data.uri}</span></div>
            <div><span className="rag-kv__k">commit</span><span className="rag-kv__v rag-mono">{data.clone.commit}</span></div>
            <div><span className="rag-kv__k">local path</span><span className="rag-kv__v rag-mono">{data.clone.local_path}</span></div>
          </div>
        </RagSection>
      )}

      {visible.walk && data.walk && (
        <RagSection
          id="walk"
          num={stageNum("walk")}
          title="Walk files"
          description="Extension + size filtered; skips vendor dirs, lockfiles, binaries."
          plain="Skip the junk — build folders, dependency folders, huge or generated files — and keep only the real, hand-written source code."
        >
          <div className="rag-stat-row">
            <div className="rag-stat"><span className="rag-stat__n">{data.walk.total_scanned}</span><span className="rag-stat__l">scanned</span></div>
            <div className="rag-stat rag-stat--accent"><span className="rag-stat__n">{data.walk.kept}</span><span className="rag-stat__l">kept</span></div>
            <div className="rag-stat"><span className="rag-stat__n">{data.walk.skipped.length}</span><span className="rag-stat__l">skipped</span></div>
          </div>
          {data.walk.skipped.length > 0 && (
            <>
              <p className="rag-hint">Examples of skipped paths:</p>
              <ul className="rag-plainlist">
                {data.walk.skipped.slice(0, 8).map((s) => (
                  <li key={s.path}>
                    <span className="rag-mono rag-dim">{s.path}</span>
                    <RagTag tone="neutral">{s.reason}</RagTag>
                  </li>
                ))}
              </ul>
              {data.walk.skipped.length > 8 && (
                <p className="rag-hint">+ {data.walk.skipped.length - 8} more skipped files.</p>
              )}
            </>
          )}
        </RagSection>
      )}

      {visible.chunk && (
        <RagSection
          id="chunk"
          num={stageNum("chunk")}
          title={data.kind === "repo" ? "AST-chunk" : "Chunk"}
          description={
            data.kind === "repo"
              ? "tree-sitter splits on function/class boundaries — a chunk is always a whole definition."
              : "Split on line boundaries with overlap — no function/class structure to split on in prose."
          }
          plain="Cut each source into pieces small enough to search and cite individually."
        >
          {shown.map((file) => (
            <div className="rag-file" key={file.file_path}>
              <div className="rag-file__head">
                <span className="rag-file__path">{file.file_path}</span>
                <RagTag tone="neutral">{file.language}</RagTag>
                <RagTag tone="accent">{file.chunks.length} chunks</RagTag>
              </div>
              {file.chunks.map((c) => (
                <RagChunkCard key={c.chunk.id} chunk={c.chunk} mode="full" />
              ))}
            </div>
          ))}
          {hiddenCount > 0 && <p className="rag-hint">+ {hiddenCount} more files chunked the same way.</p>}
          {toggle}
        </RagSection>
      )}

      {visible.contextualize && (
        <RagSection
          id="contextualize"
          num={stageNum("contextualize")}
          title={data.kind === "repo" ? "Summarize + contextualize" : "Template summaries"}
          description={
            data.kind === "repo"
              ? "An LLM writes a file summary (feeds routing) and a one-sentence context header per chunk (feeds embedding)."
              : "Prose is already self-describing, so both the routing summary and the chunk header are templated — no LLM call."
          }
          plain="A short note above each piece explaining what it is and where it's from — so a tiny, out-of-context snippet still makes sense on its own."
        >
          {data.summary && (
            <div className="rag-callout">
              <span className="rag-callout__label">summary</span>
              <p>{data.summary}</p>
            </div>
          )}
          {shown.map((file) => (
            <div className="rag-file" key={file.file_path}>
              <div className="rag-file__head">
                <span className="rag-file__path">{file.file_path}</span>
                <RagTag tone="neutral">{file.symbols.length} symbols</RagTag>
              </div>
              <p className="rag-file__summary">{file.summary}</p>
              {file.chunks.map((c) => (
                <RagChunkCard key={c.chunk.id} chunk={c.chunk} mode="context" />
              ))}
            </div>
          ))}
          {hiddenCount > 0 && (
            <p className="rag-hint">+ {hiddenCount} more files summarized the same way.</p>
          )}
          {toggle}
        </RagSection>
      )}

      {visible.embed && (
        <RagSection
          id="embed"
          num={stageNum("embed")}
          title="Embed + upsert"
          description="Summaries embed into file_summaries (routing); context-headered chunks embed into chunks (retrieval)."
          plain="Turn every note and code piece into a list of numbers (a 'vector') that captures its meaning — similar code ends up with similar numbers, so 'closeness' becomes measurable."
        >
          <RagVectorSpace
            {...pointsForIngestChunks(data)}
            legend={[{ tone: "neutral", label: "chunk embedding · hover for file" }]}
            caption="Every chunk from this ingest, projected to 3D via PCA — drag to rotate. Chunks from the same file tend to land close together; the third axis separates points that would overlap in a flat plot. This is the space hybrid search and reranking will search over."
          />
          {shown.map((file) => (
            <div className="rag-file" key={file.file_path}>
              <div className="rag-file__head">
                <span className="rag-file__path">{file.file_path}</span>
                <RagTag tone="accent2">→ file_summaries</RagTag>
              </div>
              <RagEmbedding embedding={file.summary_embedding} label="summary vector" />
              <div className="rag-chunk-rows">
                {file.chunks.map((c) => (
                  <RagChunkCard key={c.chunk.id} chunk={c.chunk} mode="embedOnly" embedding={c.embedding} />
                ))}
              </div>
            </div>
          ))}
          <p className="rag-hint">
            → chunks collection, {totalChunks(data)} vectors upserted{hiddenCount > 0 ? ` (showing first ${MAX_FILES_SHOWN} files)` : ""}.
          </p>
          {toggle}
        </RagSection>
      )}
    </Fragment>
  );
}
