from __future__ import annotations

from pathlib import Path

from tree_sitter_language_pack import get_parser

from src.index.schema import CodeChunk
from src.ingest.chunker import RecursiveChunker, Tokenizer

# Top-level node types treated as one chunk each. Anything else at top level
# (imports, module docstrings, package decls, stray statements) is grouped
# into "filler" chunks covering the surrounding lines.
DEFINITION_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "rust": {"function_item", "struct_item", "enum_item", "impl_item", "trait_item"},
}

# Nodes that wrap a real definition (decorators, export statements) — unwrap
# to the field holding the actual definition to classify/name it correctly.
WRAPPER_FIELD_BY_TYPE = {
    "decorated_definition": "definition",
    "export_statement": "declaration",
}


class ASTChunker:
    """Splits source files on function/class boundaries via tree-sitter.

    Falls back to RecursiveChunker for unsupported languages or parse failures —
    one exotic/malformed file must never abort ingestion of the rest of the repo.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        fallback: RecursiveChunker,
        max_chars: int = 2000,
        overlap_chars: int = 200,
    ):
        self.tokenizer = tokenizer
        self.fallback = fallback
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_file(
        self, path: Path, root: Path, space_id: str, source_id: str, language: str
    ) -> list[CodeChunk]:
        definitions = DEFINITION_TYPES.get(language)
        if definitions is None:
            return self.fallback.chunk_file(path, root, space_id, source_id)

        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = get_parser(language).parse(text.encode("utf-8"))
        except Exception:
            return self.fallback.chunk_file(path, root, space_id, source_id)

        file_path = str(path.relative_to(root))
        lines = text.splitlines()
        chunks: list[CodeChunk] = []
        filler_start: int | None = None  # 0-indexed row where a filler run began

        def flush_filler(end_row: int) -> None:
            nonlocal filler_start
            if filler_start is None:
                return
            code = "\n".join(lines[filler_start:end_row])
            if code.strip():
                chunks.append(
                    self._make_chunk(
                        file_path, space_id, source_id, language, None,
                        filler_start + 1, end_row, code,
                    )
                )
            filler_start = None

        for node in tree.root_node.children:
            symbol_name, is_definition = self._inspect(node, definitions)
            if not is_definition:
                if filler_start is None:
                    filler_start = node.start_point[0]
                continue
            flush_filler(node.start_point[0])
            start_line, end_line = node.start_point[0] + 1, node.end_point[0] + 1
            node_text = node.text.decode("utf-8", errors="replace")
            chunks.extend(
                self._split_if_oversized(
                    file_path, space_id, source_id, language, symbol_name,
                    start_line, end_line, node_text,
                )
            )
        flush_filler(len(lines))

        return chunks if chunks else self.fallback.chunk_file(path, root, space_id, source_id)

    def _inspect(self, node, definitions: set[str]) -> tuple[str | None, bool]:
        target = node
        field_name = WRAPPER_FIELD_BY_TYPE.get(node.type)
        if field_name:
            inner = node.child_by_field_name(field_name)
            if inner is not None:
                target = inner
        if target.type not in definitions:
            return None, False
        name_node = target.child_by_field_name("name")
        symbol_name = name_node.text.decode("utf-8", errors="replace") if name_node else None
        return symbol_name, True

    def _split_if_oversized(
        self,
        file_path: str,
        space_id: str,
        source_id: str,
        language: str,
        symbol_name: str | None,
        start_line: int,
        end_line: int,
        node_text: str,
    ) -> list[CodeChunk]:
        if len(node_text) <= self.max_chars:
            return [
                self._make_chunk(
                    file_path, space_id, source_id, language, symbol_name,
                    start_line, end_line, node_text,
                )
            ]
        return [
            self._make_chunk(
                file_path,
                space_id,
                source_id,
                language,
                symbol_name,
                start_line + rel_start - 1,
                start_line + rel_end - 1,
                block,
            )
            for block, rel_start, rel_end in self.tokenizer.split_with_lines(
                node_text, self.max_chars, self.overlap_chars
            )
        ]

    def _make_chunk(
        self,
        file_path: str,
        space_id: str,
        source_id: str,
        language: str,
        symbol_name: str | None,
        start_line: int,
        end_line: int,
        code: str,
    ) -> CodeChunk:
        return CodeChunk(
            id=f"{file_path}::{start_line}-{end_line}",
            space_id=space_id,
            source_id=source_id,
            file_path=file_path,
            language=language,
            symbol_name=symbol_name,
            start_line=start_line,
            end_line=end_line,
            code=code,
        )
