"""
Information Extractor
=====================

Extracts structured information from resume text:
- Contact info (name, email, phone)
- Skills (via spaCy NER + ESCO-based skillNer + supplementary taxonomy)
- Education (degrees, institutions)
- Experience (years, job titles)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import spacy
from spacy.matcher import PhraseMatcher

from src.config import SKILLS_TAXONOMY_PATH, SPACY_MODEL_NAME

logger = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class ContactInfo:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

@dataclass
class ExtractedInfo:
    contact: ContactInfo = field(default_factory=ContactInfo)
    skills: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    experience_years: Optional[float] = None
    job_titles: list[str] = field(default_factory=list)
    sections_found: list[str] = field(default_factory=list)


# ── spaCy Model Loading ───────────────────────────────────────────────────

_nlp_cache: Optional[spacy.language.Language] = None


def get_nlp():
    """Load spaCy model (cached singleton)."""
    global _nlp_cache
    if _nlp_cache is None:
        _nlp_cache = spacy.load(SPACY_MODEL_NAME)
    return _nlp_cache


# ── Contact Extraction (regex-based) ──────────────────────────────────────

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

_PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"        # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"       # optional area code
    r"\d{3,4}[\s\-.]?\d{3,4}"          # main number
)


def extract_email(text: str) -> Optional[str]:
    """Extract the first email address from text."""
    match = _EMAIL_PATTERN.search(text)
    return match.group() if match else None


def extract_phone(text: str) -> Optional[str]:
    """Extract the first phone number from text."""
    match = _PHONE_PATTERN.search(text)
    if match:
        phone = match.group().strip()
        # Filter out numbers that are too short (likely dates or IDs)
        digits = re.sub(r"\D", "", phone)
        if len(digits) >= 7:
            return phone
    return None


def extract_name(text: str, nlp=None) -> Optional[str]:
    """
    Extract candidate full name using header line heuristics + spaCy NER.
    Prevents truncating first/middle names.
    """
    if not text or not text.strip():
        return None

    skip_keywords = {
        "resume", "curriculum", "vitae", "cv", "profile", "summary",
        "contact", "email", "phone", "address", "mobile", "linkedin",
        "github", "portfolio", "page", "developer", "engineer", "manager",
        "experience", "education", "skills", "projects", "languages"
    }

    # Strategy 1: Check top 5 non-empty lines for a standalone 2-4 word full name
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines[:5]:
        if "@" in line or "http" in line or "www." in line or ":" in line or "|" in line:
            continue
        if any(char.isdigit() for char in line):
            continue

        words = line.split()
        if 2 <= len(words) <= 4:
            line_lower_words = [w.lower() for w in words]
            if any(w in skip_keywords for w in line_lower_words):
                continue

            if all(re.match(r"^[A-Za-z'\.-]+$", w) for w in words):
                return " ".join(w.capitalize() for w in words)

    # Strategy 2: spaCy PERSON entity fallback
    if nlp is None:
        nlp = get_nlp()

    doc = nlp(text[:1000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            words = name.split()
            if 2 <= len(words) <= 4 and not any(c.isdigit() for c in name):
                if not any(w.lower() in skip_keywords for w in words):
                    return " ".join(w.capitalize() for w in words)

    return None


def extract_contact(text: str, nlp=None) -> ContactInfo:
    """Extract all contact information."""
    return ContactInfo(
        name=extract_name(text, nlp),
        email=extract_email(text),
        phone=extract_phone(text),
    )


# ── Section Detection ─────────────────────────────────────────────────────

_SECTION_PATTERNS = {
    "contact": r"(?i)\b(?:contact\s*(?:info(?:rmation)?)?|personal\s*(?:info(?:rmation)?|details?))\b",
    "summary": r"(?i)\b(?:summary|profile|about\s*me|professional\s*summary)\b",
    "objective": r"(?i)\b(?:objective|career\s*objective)\b",
    "skills": r"(?i)\b(?:skills|technical\s*skills|core\s*competencies|competencies|expertise)\b",
    "experience": r"(?i)\b(?:experience|work\s*(?:experience|history)|employment|professional\s*experience)\b",
    "education": r"(?i)\b(?:education|academic|qualifications?|certifications?)\b",
    "projects": r"(?i)\b(?:projects?|portfolio)\b",
    "certifications": r"(?i)\b(?:certifications?|licenses?|credentials?)\b",
    "languages": r"(?i)\b(?:languages?)\b",
}


def detect_sections(text: str) -> list[str]:
    """Detect which standard resume sections are present."""
    found = []
    for section_name, pattern in _SECTION_PATTERNS.items():
        if re.search(pattern, text):
            found.append(section_name)
    return found


# ── Skills Extraction ─────────────────────────────────────────────────────

def _load_taxonomy_skills() -> list[dict]:
    """Load supplementary skills from the JSON taxonomy file."""
    if not SKILLS_TAXONOMY_PATH.is_file():
        logger.warning("Skills taxonomy not found at %s", SKILLS_TAXONOMY_PATH)
        return []

    with open(SKILLS_TAXONOMY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("skills", [])


def _build_phrase_matcher(nlp, skills_data: list[dict]) -> tuple[PhraseMatcher, dict]:
    """
    Build a spaCy PhraseMatcher from the taxonomy.
    Returns the matcher and a mapping from lowercase term → canonical name.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    canonical_map: dict[str, str] = {}

    patterns = []
    for entry in skills_data:
        canonical = entry["name"]
        all_forms = [canonical] + entry.get("aliases", [])
        for form in all_forms:
            form_lower = form.lower()
            if form_lower not in canonical_map:
                canonical_map[form_lower] = canonical
                patterns.append(nlp.make_doc(form))

    if patterns:
        matcher.add("SKILLS", patterns)

    return matcher, canonical_map


def extract_skills(text: str, nlp=None) -> list[str]:
    """
    Extract skills from resume text using multiple strategies:
    1. spaCy PhraseMatcher with ESCO-sourced taxonomy
    2. Deduplication to canonical skill names
    """
    if nlp is None:
        nlp = get_nlp()

    skills_data = _load_taxonomy_skills()
    found_skills: set[str] = set()

    # Strategy 1: PhraseMatcher with taxonomy
    if skills_data:
        matcher, canonical_map = _build_phrase_matcher(nlp, skills_data)
        doc = nlp(text)
        matches = matcher(doc)
        for match_id, start, end in matches:
            span_text = doc[start:end].text.lower()
            canonical = canonical_map.get(span_text, doc[start:end].text)
            found_skills.add(canonical)

    return sorted(found_skills)


# ── Education Extraction ──────────────────────────────────────────────────

_DEGREE_PATTERNS = [
    r"(?i)\b(?:Ph\.?D|Doctorate|Doctor of)\b",
    r"(?i)\b(?:M\.?S\.?|M\.?Sc\.?|Master(?:'s)?(?:\s+of\s+\w+)?|M\.?A\.?|MBA|M\.?Tech\.?|M\.?E\.?)\b",
    r"(?i)\b(?:B\.?S\.?|B\.?Sc\.?|Bachelor(?:'s)?(?:\s+of\s+\w+)?|B\.?A\.?|B\.?Tech\.?|B\.?E\.?|BCA|BBA)\b",
    r"(?i)\b(?:Associate(?:'s)?(?:\s+(?:of|in)\s+\w+)?)\b",
    r"(?i)\b(?:Diploma|Certificate|Certification)\b",
]

_FIELD_KEYWORDS = [
    "Computer Science", "Information Technology", "Software Engineering",
    "Electrical Engineering", "Mechanical Engineering", "Civil Engineering",
    "Business Administration", "Finance", "Accounting", "Marketing",
    "Data Science", "Artificial Intelligence", "Machine Learning",
    "Mathematics", "Physics", "Chemistry", "Biology", "Economics",
    "Psychology", "Communications", "English", "History",
    "Graphic Design", "Web Development", "Cybersecurity",
    "Nursing", "Medicine", "Pharmacy", "Law",
]


def extract_education(text: str, nlp=None) -> list[str]:
    """Extract education qualifications (degrees + fields of study)."""
    if nlp is None:
        nlp = get_nlp()

    education: list[str] = []

    # Extract degree mentions
    for pattern in _DEGREE_PATTERNS:
        for match in re.finditer(pattern, text):
            degree = match.group().strip()
            if degree not in education:
                education.append(degree)

    # Extract field of study mentions
    text_lower = text.lower()
    for field_name in _FIELD_KEYWORDS:
        if field_name.lower() in text_lower:
            if field_name not in education:
                education.append(field_name)

    # Extract university names via spaCy ORG entities
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org = ent.text.strip()
            org_lower = org.lower()
            if any(
                kw in org_lower
                for kw in ["university", "college", "institute", "school", "academy"]
            ):
                if org not in education:
                    education.append(org)

    return education


# ── Experience Extraction ─────────────────────────────────────────────────

_DATE_RANGE_PATTERN = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*)"
    r"(\d{4})"
    r"\s*(?:[-–—to]+)\s*"
    r"(?:"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*)?(\d{4})"
    r"|(?:Present|Current|Now|Ongoing)"
    r")",
    re.IGNORECASE,
)

_YEAR_RANGE_PATTERN = re.compile(
    r"\b(\d{4})\s*[-–—]\s*(?:(\d{4})|(?:Present|Current|Now))\b",
    re.IGNORECASE,
)

_EXPERIENCE_YEARS_PATTERN = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp\.?)?",
    re.IGNORECASE,
)

_JOB_TITLE_PATTERNS = [
    r"(?i)\b(?:Senior|Junior|Lead|Principal|Staff|Chief)?\s*(?:Software|Web|Full[\s-]?Stack|Frontend|Backend|Mobile|DevOps|Cloud|Data|ML|AI|QA|Test)?\s*(?:Engineer|Developer|Architect|Designer|Analyst|Scientist|Manager|Director|Consultant|Specialist|Administrator|Coordinator)\b",
    r"(?i)\b(?:Project|Product|Program|Engineering|Technical|Operations|Marketing|Sales|HR|Finance)\s+(?:Manager|Director|Lead|Head|VP|Officer)\b",
    r"(?i)\b(?:CTO|CEO|CFO|COO|CIO|VP|SVP)\b",
]


def _get_experience_section_text(text: str) -> Optional[str]:
    """Extract text strictly within work experience / employment sections."""
    lines = text.split("\n")
    exp_heading_pat = re.compile(
        r"(?i)^\s*(?:work\s+experience|professional\s+experience|employment\s+history|work\s+history|experience|employment)\s*[:\-]?\s*$"
    )
    other_heading_pat = re.compile(
        r"(?i)^\s*(?:education|academic|projects?|skills|certifications?|languages?|summary|objective|awards?|publications?)\s*[:\-]?\s*$"
    )

    in_exp = False
    exp_lines = []

    for line in lines:
        stripped = line.strip()
        if exp_heading_pat.match(stripped):
            in_exp = True
            continue
        elif in_exp and other_heading_pat.match(stripped):
            break
        elif in_exp:
            exp_lines.append(line)

    if exp_lines:
        return "\n".join(exp_lines)
    return None


def estimate_experience_years(text: str) -> Optional[float]:
    """Estimate total years of experience strictly from work experience section or explicit mentions."""
    # Check if an experience section exists
    exp_section = _get_experience_section_text(text)

    # Strategy 1: Look for explicit "X years of experience" mentions (only in experience section if found, else full text)
    search_target = exp_section if exp_section else text
    explicit = _EXPERIENCE_YEARS_PATTERN.findall(search_target)
    if explicit:
        return max(float(y) for y in explicit)

    # If no experience section was found, do NOT parse dates from education/projects
    if not exp_section:
        return 0.0

    # Strategy 2: Sum up date ranges strictly inside the experience section
    total_years = 0.0
    import datetime
    current_year = datetime.datetime.now().year

    for pattern in [_DATE_RANGE_PATTERN, _YEAR_RANGE_PATTERN]:
        for match in pattern.finditer(exp_section):
            start_year = int(match.group(1))
            end_str = match.group(2)
            end_year = int(end_str) if end_str else current_year

            if 1970 <= start_year <= current_year and start_year <= end_year <= current_year + 1:
                duration = end_year - start_year
                total_years += max(duration, 0)

    return total_years if total_years > 0 else 0.0


def extract_job_titles(text: str) -> list[str]:
    """Extract likely job titles from resume text."""
    titles: list[str] = []
    for pattern in _JOB_TITLE_PATTERNS:
        for match in re.finditer(pattern, text):
            title = match.group().strip()
            if title and title not in titles:
                titles.append(title)
    return titles


# ── Main Extraction Pipeline ──────────────────────────────────────────────

def extract_all(text: str) -> ExtractedInfo:
    """
    Run the full extraction pipeline on resume text.

    Parameters
    ----------
    text : str
        Raw resume text (from parser.py).

    Returns
    -------
    ExtractedInfo
        Structured extraction results.
    """
    nlp = get_nlp()

    return ExtractedInfo(
        contact=extract_contact(text, nlp),
        skills=extract_skills(text, nlp),
        education=extract_education(text, nlp),
        experience_years=estimate_experience_years(text),
        job_titles=extract_job_titles(text),
        sections_found=detect_sections(text),
    )
