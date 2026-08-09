"""
PDF Report Generator
====================

Generates a downloadable PDF analysis report using fpdf2.
Includes: match score, ATS score, skill gap, recommendations, suggestions.
Handles Unicode characters safely to prevent FPDFUnicodeEncodingException.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from datetime import datetime
from typing import Optional

from fpdf import FPDF

from src.scorer import MatchResult
from src.recommender import JobRecommendation
from src.llm_suggestions import SuggestionResult

logger = logging.getLogger(__name__)


# ── Unicode Safety / Text Sanitization ─────────────────────────────────────

def sanitize_pdf_text(text: str) -> str:
    """
    Sanitize text to ensure it can be rendered safely in FPDF without throwing
    FPDFUnicodeEncodingException. Replaces unicode dashes, quotes, bullets, etc.
    with standard latin-1/ASCII equivalents.
    """
    if not text:
        return ""

    # Map common non-latin-1 unicode characters to standard ASCII/latin-1 equivalents
    replacements = {
        "—": "-",
        "–": "-",
        "‒": "-",
        "―": "-",
        "•": "*",
        "●": "*",
        "◦": "*",
        "▪": "*",
        "→": "->",
        "►": "->",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "™": "(TM)",
        "®": "(R)",
        "©": "(C)",
        "\u00a0": " ",  # non-breaking space
    }

    for char, repl in replacements.items():
        text = text.replace(char, repl)

    # Encode to latin-1 with replacement for any remaining unrepresentable chars
    cleaned = text.encode("latin-1", errors="replace").decode("latin-1")
    return cleaned


# ── Color Helpers ──────────────────────────────────────────────────────────

def _score_color(score: float) -> tuple[int, int, int]:
    """Return RGB color based on score (0–100)."""
    if score >= 75:
        return (34, 139, 34)   # Green
    elif score >= 50:
        return (218, 165, 32)  # Goldenrod
    elif score >= 25:
        return (255, 140, 0)   # Orange
    else:
        return (220, 20, 60)   # Crimson


# ── PDF Builder ────────────────────────────────────────────────────────────

class ReportPDF(FPDF):
    """Custom PDF with header/footer styling and sanitized text rendering."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, sanitize_pdf_text("Smart Resume Analyser - Report"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(127, 140, 141)
        self.cell(0, 5, sanitize_pdf_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        # Separator line
        self.set_draw_color(189, 195, 199)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(127, 140, 141)
        self.cell(0, 10, sanitize_pdf_text(f"Page {self.page_no()}/{{nb}}"), align="C")

    def section_title(self, title: str):
        self.ln(5)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(44, 62, 80)
        self.cell(0, 8, sanitize_pdf_text(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(52, 152, 219)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def score_badge(self, label: str, score: float):
        r, g, b = _score_color(score)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(44, 62, 80)
        self.cell(60, 7, sanitize_pdf_text(f"{label}:"), new_x="RIGHT")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(r, g, b)
        self.cell(30, 7, sanitize_pdf_text(f"{score:.1f}/100"), new_x="LMARGIN", new_y="NEXT")

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(52, 73, 94)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5, sanitize_pdf_text(text), new_x="LMARGIN", new_y="NEXT")

    def bullet_item(self, text: str, indent: int = 10):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(52, 73, 94)
        self.set_x(self.l_margin + indent)
        self.cell(5, 5, "*", new_x="RIGHT")
        self.multi_cell(0, 5, sanitize_pdf_text(text), new_x="LMARGIN", new_y="NEXT")


# ── Report Generation ─────────────────────────────────────────────────────

def generate_report(
    match_result: MatchResult,
    recommendations: Optional[list[JobRecommendation]] = None,
    suggestions: Optional[SuggestionResult] = None,
    candidate_name: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF analysis report.

    Parameters
    ----------
    match_result : MatchResult
        Scoring results from scorer.py.
    recommendations : list[JobRecommendation], optional
        Job recommendations from recommender.py.
    suggestions : SuggestionResult, optional
        LLM suggestions from llm_suggestions.py.
    candidate_name : str, optional
        Candidate name for personalization.

    Returns
    -------
    bytes
        PDF file content as bytes.
    """
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Candidate Info ──
    if candidate_name:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 8, sanitize_pdf_text(f"Candidate: {candidate_name}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ── Resume Info Summary ──
    if match_result.resume_info:
        info = match_result.resume_info
        pdf.section_title("Extracted Candidate Overview")
        if info.contact.email or info.contact.phone:
            contact_str = " | ".join(filter(None, [info.contact.email, info.contact.phone]))
            pdf.body_text(f"Contact: {contact_str}")
        if info.experience_years:
            pdf.body_text(f"Estimated Experience: ~{info.experience_years:.0f} years")
        if info.education:
            pdf.body_text(f"Highest Education: {', '.join(info.education[:3])}")
        if info.job_titles:
            pdf.body_text(f"Detected Roles: {', '.join(info.job_titles[:5])}")

    # ── Score Summary ──
    pdf.section_title("Score Summary")
    if match_result.has_jd:
        pdf.score_badge("Overall Match Score", match_result.overall_score)
        pdf.score_badge("Semantic Similarity", match_result.semantic_similarity)
        pdf.score_badge("Skill Overlap", match_result.skill_overlap_pct)
        pdf.score_badge("Experience Match", match_result.experience_match)
        pdf.score_badge("Education Match", match_result.education_match)
    else:
        pdf.body_text("Analysis Mode: Resume-Only (No Job Description provided)")
        pdf.score_badge("ATS Resume Quality Score", match_result.overall_score)

    pdf.score_badge("ATS Formatting Score", match_result.ats_score.total_score)

    # ── ATS Details ──
    pdf.section_title("ATS Formatting Analysis")
    pdf.score_badge("Section Coverage", match_result.ats_score.section_score)
    pdf.score_badge("Resume Length", match_result.ats_score.length_score)
    pdf.score_badge("Bullet Points", match_result.ats_score.bullet_score)
    pdf.score_badge("Keyword / Quality Density", match_result.ats_score.keyword_density_score)
    pdf.score_badge("Contact Info Completeness", match_result.ats_score.contact_score)
    pdf.score_badge("Formatting Cleanliness", match_result.ats_score.formatting_score)

    if match_result.ats_score.feedback:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 6, sanitize_pdf_text("ATS Feedback & Optimization Tips:"), new_x="LMARGIN", new_y="NEXT")
        for fb in match_result.ats_score.feedback:
            pdf.bullet_item(fb)

    # ── Skill Analysis ──
    pdf.section_title("Skill Analysis")

    gap = match_result.gap_analysis
    if match_result.has_jd:
        pdf.score_badge("Skill Overlap Percentage", gap.skill_overlap_pct)
        pdf.ln(2)

        if gap.matched_skills:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(34, 139, 34)
            pdf.cell(0, 6, sanitize_pdf_text(f"Matched Skills ({len(gap.matched_skills)}):"), new_x="LMARGIN", new_y="NEXT")
            pdf.body_text(", ".join(gap.matched_skills))

        if gap.missing_skills:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(220, 20, 60)
            pdf.cell(0, 6, sanitize_pdf_text(f"Missing Skills from JD ({len(gap.missing_skills)}):"), new_x="LMARGIN", new_y="NEXT")
            pdf.body_text(", ".join(gap.missing_skills))
        else:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(34, 139, 34)
            pdf.cell(0, 6, sanitize_pdf_text("No critical skills missing for target JD!"), new_x="LMARGIN", new_y="NEXT")

        if gap.missing_keywords:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(255, 140, 0)
            pdf.cell(0, 6, sanitize_pdf_text(f"Missing High-Frequency Keywords ({len(gap.missing_keywords)}):"), new_x="LMARGIN", new_y="NEXT")
            pdf.body_text(", ".join(gap.missing_keywords))
    else:
        pdf.body_text("Extracted Skills from Resume:")
        if gap.extra_skills:
            pdf.body_text(", ".join(gap.extra_skills))
        else:
            pdf.body_text("No explicit technical skills extracted.")

    # ── Job Recommendations ──
    if recommendations:
        pdf.section_title("Job Recommendations")
        for i, rec in enumerate(recommendations[:8], 1):
            r, g, b = _score_color(rec.similarity_score)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(r, g, b)
            pdf.cell(10, 6, sanitize_pdf_text(f"{i}."), new_x="RIGHT")
            pdf.set_text_color(44, 62, 80)
            pdf.cell(80, 6, sanitize_pdf_text(rec.role), new_x="RIGHT")
            pdf.set_text_color(r, g, b)
            pdf.cell(30, 6, sanitize_pdf_text(f"{rec.similarity_score:.1f}%"), new_x="LMARGIN", new_y="NEXT")

    # ── LLM Suggestions ──
    if suggestions and suggestions.llm_available and suggestions.suggestions:
        pdf.section_title("AI-Powered Improvement Suggestions")
        if suggestions.summary:
            pdf.body_text(suggestions.summary)
            pdf.ln(3)

        for i, s in enumerate(suggestions.suggestions, 1):
            severity_colors = {
                "high": (220, 20, 60),
                "medium": (255, 140, 0),
                "low": (52, 152, 219),
            }
            r, g, b = severity_colors.get(s.severity, (52, 73, 94))

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(r, g, b)
            pdf.cell(0, 5, sanitize_pdf_text(f"{i}. [{s.category}] {s.severity.upper()}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(52, 73, 94)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, sanitize_pdf_text(f"   {s.suggestion}"), new_x="LMARGIN", new_y="NEXT")
            if s.explanation:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(127, 140, 141)
                pdf.multi_cell(0, 4, sanitize_pdf_text(f"   Why: {s.explanation}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    # Output as bytes
    return bytes(pdf.output())
