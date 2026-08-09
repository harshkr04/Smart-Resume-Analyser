"""Tests for src/extractor.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extractor import (
    extract_email,
    extract_phone,
    extract_contact,
    detect_sections,
    estimate_experience_years,
    extract_job_titles,
)


class TestContactExtraction:
    """Test contact info extraction."""

    def test_extract_email(self):
        assert extract_email("Contact me at john@example.com for info") == "john@example.com"
        assert extract_email("No email here") is None

    def test_extract_phone(self):
        assert extract_phone("Call me at (123) 456-7890") is not None
        assert extract_phone("No phone here") is None

    def test_extract_email_complex(self):
        assert extract_email("jane.doe+tag@company.co.uk is my email") == "jane.doe+tag@company.co.uk"


class TestSectionDetection:
    """Test resume section detection."""

    def test_detect_common_sections(self):
        text = """
        Education
        BS Computer Science

        Experience
        Software Engineer at ABC Corp

        Skills
        Python, Java, SQL
        """
        sections = detect_sections(text)
        assert "education" in sections
        assert "experience" in sections
        assert "skills" in sections

    def test_detect_no_sections(self):
        sections = detect_sections("Just some random text without headers")
        # Should find very few or no sections
        assert isinstance(sections, list)


class TestExperienceEstimation:
    """Test experience years estimation."""

    def test_explicit_years(self):
        text = "I have 5 years of experience in software development"
        years = estimate_experience_years(text)
        assert years == 5.0

    def test_date_range(self):
        text = "Software Engineer | 2018 - 2023"
        years = estimate_experience_years(text)
        assert years is not None
        assert years >= 4  # 2023-2018 = 5

    def test_no_experience(self):
        text = "Fresh graduate looking for opportunities"
        years = estimate_experience_years(text)
        assert years is None


class TestJobTitleExtraction:
    """Test job title extraction."""

    def test_extract_common_titles(self):
        text = "I worked as a Software Engineer and later as a Senior Developer"
        titles = extract_job_titles(text)
        assert len(titles) > 0

    def test_extract_manager_titles(self):
        text = "Project Manager at XYZ Corp, previously Product Director"
        titles = extract_job_titles(text)
        assert any("Manager" in t or "Director" in t for t in titles)
