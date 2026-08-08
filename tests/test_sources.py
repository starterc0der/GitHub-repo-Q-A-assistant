from __future__ import annotations

from src.ingest.sources import load_docx, load_pdf, load_text


def test_load_text_wraps_pasted_string_as_one_logical_file() -> None:
    files = load_text("hello world", "Pasted note")
    assert len(files) == 1
    assert files[0].name == "Pasted note"
    assert files[0].text == "hello world"


def test_load_text_skips_blank_paste() -> None:
    assert load_text("   ", "Pasted note") == []


def test_load_docx_reads_paragraphs_as_one_logical_file(tmp_path) -> None:
    from docx import Document

    doc = Document()
    doc.add_paragraph("First paragraph.")
    doc.add_paragraph("Second paragraph.")
    path = tmp_path / "notes.docx"
    doc.save(path)

    files = load_docx(path, "notes.docx")

    assert len(files) == 1
    assert files[0].name == "notes.docx"
    assert "First paragraph." in files[0].text
    assert "Second paragraph." in files[0].text


def test_load_pdf_skips_pages_with_no_extractable_text(tmp_path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as f:
        writer.write(f)

    assert load_pdf(path, "blank.pdf") == []
