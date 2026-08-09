"""Tests for src/scorer.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scorer import (
    compute_ats_score,
    compute_gap_analysis,
    score_resume,
    _score_sections,
    _score_length,
    _score_bullets,
)


class TestATSScoring:
    """Test ATS scoring components."""

    def test_section_scoring_all_present(self):
        sections = ["contact", "summary", "skills", "experience", "education"]
        score, missing, feedback = _score_sections(sections)
        assert score >= 80
        assert len(missing) == 0

    def test_section_scoring_none_present(self):
        score, missing, feedback = _score_sections([])
        assert score < 30
        assert len(missing) > 0

    def test_length_scoring_ideal(self):
        text = "x" * 2000  # Within ideal range
        score, feedback = _score_length(text)
        assert score == 100.0

    def test_length_scoring_too_short(self):
        text = "short"
        score, feedback = _score_length(text)
        assert score < 50

    def test_bullet_scoring(self):
        text = """
Summary
I am a software engineer.

Experience
- Built scalable APIs
- Designed microservices
- Led team of 5 engineers

Skills
Python, Java, SQL
        """
        score, feedback = _score_bullets(text)
        assert score > 0  # Has some bullets

    def test_dynamic_ats_score_changes_with_resume(self):
        resume_good = "John Doe\njohn@example.com\n123-456-7890\n\nSummary\nExperienced developer.\n\nExperience\n- Developed scalable APIs and microservices for 5 years.\n- Managed team of 4 engineers.\n\nEducation\nBachelor of Science in Computer Science.\n\nSkills\nPython, Java, SQL, AWS, Docker"
        resume_poor = "bad text"

        score_good = compute_ats_score(resume_good, "", ["contact", "summary", "experience", "education", "skills"])
        score_poor = compute_ats_score(resume_poor, "", [])

        assert score_good.total_score > score_poor.total_score
        assert score_good.contact_score == 100.0
        assert score_poor.contact_score == 0.0


class TestGapAnalysis:
    """Test gap analysis."""

    def test_gap_analysis_basic(self):
        resume_skills = ["Python", "Java", "SQL"]
        jd_skills = ["Python", "JavaScript", "React", "SQL"]
        result = compute_gap_analysis(resume_skills, jd_skills, "python java sql", "python javascript react sql")
        matched_lower = [s.lower() for s in result.matched_skills]
        assert "python" in matched_lower
        assert "sql" in matched_lower

    def test_gap_analysis_full_overlap(self):
        skills = ["Python", "Java"]
        result = compute_gap_analysis(skills, skills, "python java", "python java")
        assert result.skill_overlap_pct == 100.0
        assert len(result.missing_skills) == 0

    def test_gap_analysis_no_jd(self):
        result = compute_gap_analysis(["Python"], [], "python", "")
        assert len(result.missing_skills) == 0
        assert len(result.matched_skills) == 0
