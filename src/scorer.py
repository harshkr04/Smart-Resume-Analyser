"""
ATS Scorer
==========

Computes:
1. Overall Match Score (weighted composite)
2. ATS Formatting Score (structural quality checks)
3. Gap Analysis (missing skills, keywords)

Handles both JD-present and JD-absent modes. When no JD is provided,
ATS scoring uses resume-only heuristics instead of keyword comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.config import (
    ATS_EXPECTED_SECTIONS,
    RESUME_IDEAL_MAX,
    RESUME_IDEAL_MIN,
    RESUME_MAX_LENGTH,
    RESUME_MIN_LENGTH,
    SCORE_WEIGHTS,
)
from src.extractor import ExtractedInfo, extract_all
from src.matcher import compute_similarity, compute_skill_overlap


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class GapAnalysis:
    """Skills and keywords the resume is missing for a given JD."""
    missing_skills: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    skill_overlap_pct: float = 0.0


@dataclass
class ATSFormatScore:
    """ATS formatting / structural quality score."""
    total_score: float = 0.0  # 0–100
    section_score: float = 0.0
    length_score: float = 0.0
    bullet_score: float = 0.0
    keyword_density_score: float = 0.0
    contact_score: float = 0.0
    formatting_score: float = 0.0
    missing_sections: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)


@dataclass
class MatchResult:
    """Complete scoring result for a resume-JD pair."""
    overall_score: float = 0.0  # 0–100, weighted composite
    semantic_similarity: float = 0.0
    skill_overlap_pct: float = 0.0
    experience_match: float = 0.0
    education_match: float = 0.0
    ats_score: ATSFormatScore = field(default_factory=ATSFormatScore)
    gap_analysis: GapAnalysis = field(default_factory=GapAnalysis)
    resume_info: Optional[ExtractedInfo] = None
    jd_info: Optional[ExtractedInfo] = None
    has_jd: bool = False  # Whether a Job Description was provided


# ── ATS Formatting Score ──────────────────────────────────────────────────

def _score_sections(sections_found: list[str]) -> tuple[float, list[str], list[str]]:
    """Score based on presence of expected resume sections."""
    expected = set(ATS_EXPECTED_SECTIONS)
    found = set(sections_found)

    # "summary" and "objective" are interchangeable
    if "summary" in found or "objective" in found:
        found.update(["summary", "objective"])

    matched = expected & found
    missing = expected - found

    score = (len(matched) / max(len(expected), 1)) * 100

    feedback = []
    if missing:
        feedback.append(
            f"Missing resume sections: {', '.join(sorted(missing))}. "
            "Adding these improves ATS parsing."
        )

    return score, sorted(missing), feedback


def _score_length(text: str) -> tuple[float, list[str]]:
    """Score based on resume length (chars)."""
    length = len(text)
    feedback = []

    if length < RESUME_MIN_LENGTH:
        feedback.append(
            f"Resume is very short ({length} chars). "
            "ATS systems may flag this. Aim for 400-4000 characters."
        )
        return 20.0, feedback
    elif length < RESUME_IDEAL_MIN:
        feedback.append(
            f"Resume is somewhat short ({length} chars). Consider adding more detail."
        )
        return 60.0, feedback
    elif length <= RESUME_IDEAL_MAX:
        return 100.0, feedback
    elif length <= RESUME_MAX_LENGTH:
        feedback.append(
            f"Resume is slightly long ({length} chars). Consider trimming to ~4000 chars."
        )
        return 80.0, feedback
    else:
        feedback.append(
            f"Resume is very long ({length} chars). "
            "ATS systems may truncate. Aim for 1-2 pages."
        )
        return 40.0, feedback


def _score_bullets(text: str) -> tuple[float, list[str]]:
    """Score based on use of bullet points in experience descriptions."""
    # Count lines that start with bullet-like characters
    bullet_patterns = [
        r"^\s*[-\u2022\u25cf\u25e6\u25aa\u2192\u25ba]",     # common bullet chars
        r"^\s*\d+[.)]\s",       # numbered lists
        r"^\s*[a-z][.)]\s",     # lettered lists
    ]

    total_lines = len([l for l in text.split("\n") if l.strip()])
    bullet_lines = 0
    for line in text.split("\n"):
        for pat in bullet_patterns:
            if re.match(pat, line):
                bullet_lines += 1
                break

    if total_lines == 0:
        return 0.0, ["No content detected."]

    ratio = bullet_lines / total_lines
    feedback = []

    if ratio < 0.1:
        feedback.append(
            "Very few bullet points detected. Use bullets for experience "
            "descriptions to improve readability and ATS parsing."
        )
        return 30.0, feedback
    elif ratio < 0.25:
        feedback.append("Consider using more bullet points for your experience entries.")
        return 60.0, feedback
    elif ratio > 0.8:
        feedback.append(
            "Resume is almost entirely bullet points. "
            "Add some prose sections (summary, objective) for balance."
        )
        return 70.0, feedback
    else:
        return 100.0, feedback


def _score_keyword_density(resume_text: str, jd_text: str) -> tuple[float, list[str]]:
    """
    Score based on keyword overlap between resume and JD.
    When no JD is provided, scores resume's own keyword richness.
    """
    # Extract significant words (>3 chars, not stopwords)
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "have", "been", "from", "they", "will", "with", "this",
        "that", "each", "make", "like", "been", "into", "some",
        "them", "than", "its", "over", "such", "also", "more",
        "other", "their", "which", "about", "would", "these",
        "should", "through", "experience", "work", "working",
        "ability", "strong", "including", "using", "years",
        "required", "preferred", "responsibilities", "qualifications",
    }

    def extract_keywords(text: str) -> set[str]:
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        return {w for w in words if w not in stopwords}

    if not jd_text.strip():
        # ── No JD: score resume's own keyword quality ──
        resume_keywords = extract_keywords(resume_text)
        total_words = len(re.findall(r"\b\w+\b", resume_text))

        if total_words == 0:
            return 0.0, ["Resume appears empty."]

        # Action verbs that ATS systems look for
        action_verbs = {
            "developed", "designed", "implemented", "managed", "created",
            "built", "improved", "optimized", "analyzed", "reduced",
            "increased", "established", "delivered", "achieved", "automated",
            "coordinated", "executed", "generated", "launched", "maintained",
            "resolved", "spearheaded", "streamlined", "supervised", "trained",
            "collaborated", "contributed", "engineered", "integrated", "migrated",
            "deployed", "architected", "mentored", "facilitated", "evaluated",
        }

        # Quantifiable achievements (numbers, percentages)
        quant_matches = re.findall(
            r"\b\d+[%+]?\b|\b\d+(?:k|m|x)\b",
            resume_text.lower(),
        )

        action_count = len(action_verbs & resume_keywords)
        quant_count = min(len(quant_matches), 15)

        # Score: action verbs (50%) + quantification (30%) + keyword diversity (20%)
        action_score = min(action_count / 8, 1.0) * 100  # 8+ action verbs = 100%
        quant_score = min(quant_count / 5, 1.0) * 100    # 5+ quantified items = 100%
        diversity_score = min(len(resume_keywords) / 40, 1.0) * 100  # 40+ unique kw = 100%

        combined = action_score * 0.50 + quant_score * 0.30 + diversity_score * 0.20

        feedback = []
        if action_count < 3:
            feedback.append(
                f"Only {action_count} action verbs detected. Use verbs like "
                "'developed', 'implemented', 'managed' to strengthen descriptions."
            )
        if quant_count < 2:
            feedback.append(
                "Few quantifiable achievements found. Add numbers/percentages "
                "to demonstrate impact (e.g., 'Increased sales by 20%')."
            )

        return round(combined, 1), feedback

    # ── JD present: compare keywords ──
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)

    if not jd_keywords:
        return 50.0, []

    overlap = jd_keywords & resume_keywords
    density = len(overlap) / len(jd_keywords) * 100

    feedback = []
    if density < 30:
        feedback.append(
            f"Only {density:.0f}% of JD keywords found in resume. "
            "Consider incorporating more relevant terms."
        )

    return min(density, 100.0), feedback


def _score_contact_info(resume_info: ExtractedInfo) -> tuple[float, list[str]]:
    """Score based on presence of contact information."""
    score = 0.0
    feedback = []
    checks = 0
    total = 3  # name, email, phone

    if resume_info.contact.name:
        checks += 1
    else:
        feedback.append("No name detected. Ensure your full name is at the top of your resume.")

    if resume_info.contact.email:
        checks += 1
    else:
        feedback.append("No email address detected. Include a professional email.")

    if resume_info.contact.phone:
        checks += 1
    else:
        feedback.append("No phone number detected. Include a contact number.")

    score = (checks / total) * 100
    return round(score, 1), feedback


def _score_formatting(text: str) -> tuple[float, list[str]]:
    """Score based on formatting quality and ATS-friendliness."""
    score = 100.0
    feedback = []

    # Check for suspicious/unusual characters that break ATS parsers
    unusual_chars = re.findall(r'[\u2018\u2019\u201c\u201d\u2014\u2013\u00ab\u00bb\u2026]', text)
    if len(unusual_chars) > 5:
        score -= 15
        feedback.append(
            f"Found {len(unusual_chars)} special Unicode characters (smart quotes, em-dashes). "
            "Some ATS systems may not parse these correctly. Use standard characters."
        )

    # Check for excessive special characters (tables/columns indicator)
    pipe_count = text.count("|")
    if pipe_count > 10:
        score -= 10
        feedback.append(
            "Detected many pipe characters (|), possibly from tables. "
            "ATS systems may not parse tables correctly."
        )

    # Check for excessive capitalization
    all_caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    if len(all_caps_words) > 10:
        score -= 10
        feedback.append(
            "Excessive all-caps text detected. Use title case for section headings."
        )

    # Check for very long paragraphs (no line breaks)
    paragraphs = text.split("\n\n")
    long_paras = [p for p in paragraphs if len(p) > 500]
    if len(long_paras) > 2:
        score -= 10
        feedback.append(
            "Some paragraphs are very long. Break them into bullet points "
            "for better readability and ATS parsing."
        )

    # Check for consistent line spacing
    empty_line_groups = re.findall(r"\n{4,}", text)
    if len(empty_line_groups) > 2:
        score -= 5
        feedback.append(
            "Excessive blank lines detected. Use consistent spacing."
        )

    return max(score, 0.0), feedback


def compute_ats_score(
    resume_text: str,
    jd_text: str,
    sections_found: list[str],
    resume_info: Optional[ExtractedInfo] = None,
) -> ATSFormatScore:
    """
    Compute the ATS formatting/structural quality score.
    Uses different weight distributions depending on whether a JD is provided.
    """
    section_score, missing, section_feedback = _score_sections(sections_found)
    length_score, length_feedback = _score_length(resume_text)
    bullet_score, bullet_feedback = _score_bullets(resume_text)
    kw_score, kw_feedback = _score_keyword_density(resume_text, jd_text)

    # Contact info + formatting scores
    contact_score, contact_feedback = (0.0, [])
    formatting_score, formatting_feedback = (0.0, [])

    if resume_info is not None:
        contact_score, contact_feedback = _score_contact_info(resume_info)
    formatting_score, formatting_feedback = _score_formatting(resume_text)

    has_jd = bool(jd_text.strip())

    if has_jd:
        # With JD: keyword density matters more
        total = (
            section_score * 0.20
            + length_score * 0.15
            + bullet_score * 0.15
            + kw_score * 0.25
            + contact_score * 0.10
            + formatting_score * 0.15
        )
    else:
        # Without JD: resume-only assessment, keyword density is action-verb based
        total = (
            section_score * 0.25
            + length_score * 0.15
            + bullet_score * 0.15
            + kw_score * 0.15
            + contact_score * 0.15
            + formatting_score * 0.15
        )

    all_feedback = (
        section_feedback + length_feedback + bullet_feedback
        + kw_feedback + contact_feedback + formatting_feedback
    )

    return ATSFormatScore(
        total_score=round(total, 1),
        section_score=round(section_score, 1),
        length_score=round(length_score, 1),
        bullet_score=round(bullet_score, 1),
        keyword_density_score=round(kw_score, 1),
        contact_score=round(contact_score, 1),
        formatting_score=round(formatting_score, 1),
        missing_sections=missing,
        feedback=all_feedback,
    )


# ── Experience Match ──────────────────────────────────────────────────────

def _extract_jd_experience_requirement(jd_text: str) -> Optional[float]:
    """Extract required years of experience from JD text."""
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp\.?)",
        r"(?:minimum|at\s+least|min\.?)\s*(\d+)\s*(?:years?|yrs?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, jd_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _score_experience_match(
    resume_years: Optional[float], jd_text: str
) -> float:
    """Score how well resume experience matches JD requirement."""
    if not jd_text.strip():
        # No JD: score based on whether experience is detectable
        if resume_years is not None and resume_years > 0:
            return 75.0  # Has detectable experience
        return 40.0  # No experience detected

    jd_years = _extract_jd_experience_requirement(jd_text)

    if jd_years is None or resume_years is None:
        return 50.0  # Neutral if we can't determine

    if resume_years >= jd_years:
        return 100.0
    elif resume_years >= jd_years * 0.7:
        return 75.0
    elif resume_years >= jd_years * 0.5:
        return 50.0
    else:
        return 25.0


# ── Education Match ───────────────────────────────────────────────────────

_DEGREE_LEVELS = {
    "phd": 5, "doctorate": 5, "doctor": 5,
    "master": 4, "mba": 4, "m.s.": 4, "m.sc.": 4, "m.tech": 4, "m.a.": 4, "m.e.": 4,
    "bachelor": 3, "b.s.": 3, "b.sc.": 3, "b.tech": 3, "b.a.": 3, "b.e.": 3, "bca": 3, "bba": 3,
    "associate": 2, "diploma": 1, "certificate": 1, "certification": 1,
}


def _get_highest_degree(education: list[str]) -> int:
    """Return the highest degree level found (0-5)."""
    max_level = 0
    for item in education:
        item_lower = item.lower()
        for keyword, level in _DEGREE_LEVELS.items():
            if keyword in item_lower:
                max_level = max(max_level, level)
    return max_level


def _score_education_match(
    resume_education: list[str], jd_text: str
) -> float:
    """Score how well resume education matches JD requirements."""
    if not jd_text.strip():
        # No JD: score based on whether education is detectable
        level = _get_highest_degree(resume_education)
        if level >= 3:
            return 80.0  # Bachelor's or higher
        elif level >= 1:
            return 55.0  # Some education
        return 35.0  # None detected

    jd_education = []
    jd_lower = jd_text.lower()
    for keyword, level in _DEGREE_LEVELS.items():
        if keyword in jd_lower:
            jd_education.append(level)

    if not jd_education:
        return 50.0  # Neutral if JD doesn't specify education

    required_level = max(jd_education)
    actual_level = _get_highest_degree(resume_education)

    if actual_level >= required_level:
        return 100.0
    elif actual_level >= required_level - 1:
        return 70.0
    else:
        return 30.0


# ── Gap Analysis ──────────────────────────────────────────────────────────

def compute_gap_analysis(
    resume_skills: list[str],
    jd_skills: list[str],
    resume_text: str,
    jd_text: str,
) -> GapAnalysis:
    """Compute a full gap analysis between resume and JD."""
    if not jd_text.strip():
        # No JD: no comparison possible, return resume skills as "extra"
        return GapAnalysis(
            missing_skills=[],
            missing_keywords=[],
            matched_skills=[],
            extra_skills=sorted({s.title() if s == s.lower() else s for s in resume_skills}),
            skill_overlap_pct=0.0,
        )

    overlap = compute_skill_overlap(resume_skills, jd_skills)

    # Find high-frequency JD keywords not in resume
    jd_words = re.findall(r"\b[a-zA-Z]{4,}\b", jd_text.lower())
    resume_lower = resume_text.lower()

    word_freq: dict[str, int] = {}
    for w in jd_words:
        word_freq[w] = word_freq.get(w, 0) + 1

    # Top keywords missing from resume
    missing_keywords = []
    for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        if freq >= 2 and word not in resume_lower and len(missing_keywords) < 15:
            missing_keywords.append(word)

    return GapAnalysis(
        missing_skills=overlap["missing"],
        missing_keywords=missing_keywords,
        matched_skills=overlap["matched"],
        extra_skills=overlap["extra"],
        skill_overlap_pct=overlap["overlap_pct"],
    )


# ── Main Scoring Pipeline ─────────────────────────────────────────────────

def score_resume(
    resume_text: str,
    jd_text: str,
    resume_info: Optional[ExtractedInfo] = None,
    jd_info: Optional[ExtractedInfo] = None,
) -> MatchResult:
    """
    Full scoring pipeline for a resume against a job description.

    Parameters
    ----------
    resume_text : str
        Raw resume text.
    jd_text : str
        Job description text (can be empty for resume-only mode).
    resume_info : ExtractedInfo, optional
        Pre-extracted resume info. If None, extracted internally.
    jd_info : ExtractedInfo, optional
        Pre-extracted JD info. If None, extracted internally.

    Returns
    -------
    MatchResult
        Complete scoring result.
    """
    has_jd = bool(jd_text.strip())

    # Extract info if not provided
    if resume_info is None:
        resume_info = extract_all(resume_text)
    if jd_info is None and has_jd:
        jd_info = extract_all(jd_text)
    elif jd_info is None:
        jd_info = ExtractedInfo()  # Empty placeholder

    # ATS formatting score (computed first, used in overall for no-JD mode)
    ats = compute_ats_score(resume_text, jd_text, resume_info.sections_found, resume_info)

    if has_jd:
        # ── Full comparison mode ──
        semantic_sim = compute_similarity(resume_text, jd_text)

        skill_overlap = compute_skill_overlap(resume_info.skills, jd_info.skills)
        skill_pct = skill_overlap["overlap_pct"]

        exp_score = _score_experience_match(resume_info.experience_years, jd_text)
        edu_score = _score_education_match(resume_info.education, jd_text)

        # Weighted overall score
        overall = (
            semantic_sim * SCORE_WEIGHTS["semantic_similarity"]
            + skill_pct * SCORE_WEIGHTS["skill_overlap"]
            + exp_score * SCORE_WEIGHTS["experience_match"]
            + edu_score * SCORE_WEIGHTS["education_match"]
        )
    else:
        # ── Resume-only mode ──
        semantic_sim = 0.0
        skill_pct = 0.0
        exp_score = _score_experience_match(resume_info.experience_years, "")
        edu_score = _score_education_match(resume_info.education, "")

        # Overall = ATS score (the only meaningful composite in resume-only mode)
        overall = ats.total_score

    # Gap analysis
    gap = compute_gap_analysis(
        resume_info.skills,
        jd_info.skills if jd_info else [],
        resume_text,
        jd_text,
    )

    return MatchResult(
        overall_score=round(overall, 1),
        semantic_similarity=round(semantic_sim, 1),
        skill_overlap_pct=round(skill_pct, 1),
        experience_match=round(exp_score, 1),
        education_match=round(edu_score, 1),
        ats_score=ats,
        gap_analysis=gap,
        resume_info=resume_info,
        jd_info=jd_info,
        has_jd=has_jd,
    )
