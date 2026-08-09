"""Tests for src/matcher.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.matcher import normalize_skill, compute_skill_overlap


class TestSkillNormalization:
    """Test skill normalization and alias resolution."""

    def test_alias_normalization(self):
        assert normalize_skill("ML") == "machine learning"
        assert normalize_skill("Machine Learning") == "machine learning"
        assert normalize_skill("AI") == "artificial intelligence"
        assert normalize_skill("JS") == "javascript"
        assert normalize_skill("Python") == "python"

    def test_overlap_with_aliases(self):
        resume_skills = ["ML", "Python", "JS"]
        jd_skills = ["Machine Learning", "JavaScript", "React"]

        overlap = compute_skill_overlap(resume_skills, jd_skills)

        # ML <-> Machine Learning and JS <-> JavaScript should match
        assert len(overlap["matched"]) == 2
        assert "React" in overlap["missing"]
