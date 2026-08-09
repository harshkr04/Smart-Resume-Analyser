"""Tests for src/parser.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import parse_resume, parse_pdf, ResumeDocument


class TestParseResume:
    """Test the resume parser."""

    def test_parse_resume_returns_document(self):
        """parse_resume should return a ResumeDocument."""
        # Test with the included sample PDF
        pdf_path = Path(__file__).resolve().parent.parent / "info resume.pdf"
        if pdf_path.is_file():
            doc = parse_resume(str(pdf_path))
            assert isinstance(doc, ResumeDocument)
            assert doc.file_type == "pdf"
            assert doc.parsing_method == "pdfplumber"
            assert not doc.is_empty
            assert "John" in doc.raw_text or "Doe" in doc.raw_text

    def test_parse_unknown_format(self):
        """parse_resume should handle unknown formats gracefully."""
        doc = parse_resume("nonexistent_file.xyz")
        assert doc.file_type == "unknown"
        assert doc.is_empty
        assert len(doc.warnings) > 0

    def test_resume_document_properties(self):
        """ResumeDocument properties should work correctly."""
        doc = ResumeDocument(
            raw_text="A short text",
            file_type="pdf",
            parsing_method="test",
        )
        assert doc.is_empty  # < 50 chars
        assert doc.char_count == len("A short text")

        doc2 = ResumeDocument(
            raw_text="A" * 100,
            file_type="pdf",
            parsing_method="test",
        )
        assert not doc2.is_empty
