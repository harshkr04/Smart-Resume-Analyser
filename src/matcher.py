"""
Semantic Matcher
================

Resume-vs-job-description matching using sentence-transformer embeddings.
Uses all-MiniLM-L6-v2 for fast, high-quality 384-dim embeddings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import SENTENCE_MODEL_NAME, SKILLS_TAXONOMY_PATH

logger = logging.getLogger(__name__)

# ── Model Cache ────────────────────────────────────────────────────────────

_model_cache: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Load sentence-transformer model (cached singleton)."""
    global _model_cache
    if _model_cache is None:
        logger.info("Loading sentence-transformer model: %s", SENTENCE_MODEL_NAME)
        _model_cache = SentenceTransformer(SENTENCE_MODEL_NAME)
        logger.info("Model loaded successfully.")
    return _model_cache


# ── Skill Normalization ────────────────────────────────────────────────────

# Common abbreviation expansions (hardcoded for speed + reliability)
_COMMON_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "tf": "tensorflow",
    "k8s": "kubernetes",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "ci/cd": "continuous integration",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "sql": "sql",
    "nosql": "nosql",
    "oop": "object-oriented programming",
    "api": "api",
    "rest": "rest api",
    "restful": "rest api",
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node.js",
    "nodejs": "node.js",
    "vue.js": "vue",
    "vuejs": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "express.js": "express",
    "expressjs": "express",
    "next.js": "next.js",
    "nextjs": "next.js",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "sci-kit learn": "scikit-learn",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "c++": "c++",
    "c#": "c#",
    "c sharp": "c#",
    "dot net": ".net",
    "dotnet": ".net",
    ".net core": ".net",
    "asp.net": ".net",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "ms word": "word",
    "power bi": "power bi",
    "powerbi": "power bi",
    "tableau": "tableau",
    "sas": "sas",
    "r programming": "r",
    "r language": "r",
}

_normalization_map: Optional[dict[str, str]] = None


def _build_normalization_map() -> dict[str, str]:
    """Build normalization map from taxonomy + common aliases."""
    norm_map: dict[str, str] = dict(_COMMON_ALIASES)

    try:
        if SKILLS_TAXONOMY_PATH.is_file():
            with open(SKILLS_TAXONOMY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("skills", []):
                canonical = entry["name"].lower().strip()
                norm_map[canonical] = canonical
                for alias in entry.get("aliases", []):
                    alias_lower = alias.lower().strip()
                    if alias_lower not in norm_map:
                        norm_map[alias_lower] = canonical
    except Exception as e:
        logger.warning("Could not load skills taxonomy for normalization: %s", e)

    return norm_map


def _get_normalization_map() -> dict[str, str]:
    """Get or build the normalization map (cached singleton)."""
    global _normalization_map
    if _normalization_map is None:
        _normalization_map = _build_normalization_map()
    return _normalization_map


def normalize_skill(skill: str) -> str:
    """Normalize a skill name to its canonical lowercase form."""
    norm_map = _get_normalization_map()
    skill_lower = skill.lower().strip()
    return norm_map.get(skill_lower, skill_lower)


# ── Core Similarity ────────────────────────────────────────────────────────

def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using sentence embeddings.

    Parameters
    ----------
    text_a : str
        First text (e.g., resume).
    text_b : str
        Second text (e.g., job description).

    Returns
    -------
    float
        Similarity score in range [0, 100].
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    model = get_model()

    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    cosine_sim = float(np.dot(embeddings[0], embeddings[1]))

    # Clamp to [0, 1] and convert to percentage
    return round(max(0.0, min(1.0, cosine_sim)) * 100, 2)


def compute_embedding(text: str) -> np.ndarray:
    """
    Compute a single embedding vector for the given text.

    Parameters
    ----------
    text : str

    Returns
    -------
    np.ndarray
        Normalized 384-dim embedding vector.
    """
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


def compute_batch_embeddings(texts: list[str]) -> np.ndarray:
    """
    Compute embeddings for a batch of texts.

    Parameters
    ----------
    texts : list[str]

    Returns
    -------
    np.ndarray
        Matrix of shape (len(texts), 384).
    """
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


# ── Section-Level Similarity ──────────────────────────────────────────────

_SECTION_SPLIT_KEYWORDS = [
    "experience", "skills", "education", "projects",
    "summary", "objective", "certifications",
]


def _split_into_sections(text: str) -> dict[str, str]:
    """
    Heuristically split text into sections based on common resume headings.
    Returns a dict of section_name → section_text.
    """
    import re

    sections: dict[str, str] = {}
    pattern = r"(?i)^[\s]*(" + "|".join(_SECTION_SPLIT_KEYWORDS) + r")[\s]*[:\-]?\s*$"

    lines = text.split("\n")
    current_section = "header"
    current_lines: list[str] = []

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            # Save previous section
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = match.group(1).lower().strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def compute_section_similarities(
    resume_text: str, jd_text: str
) -> dict[str, float]:
    """
    Compute similarity between resume sections and the full job description.

    Returns a dict of section_name → similarity_score (0–100).
    """
    resume_sections = _split_into_sections(resume_text)

    if not resume_sections:
        return {"overall": compute_similarity(resume_text, jd_text)}

    results: dict[str, float] = {}
    for section_name, section_text in resume_sections.items():
        if section_text.strip():
            results[section_name] = compute_similarity(section_text, jd_text)

    return results


# ── Skill Overlap ──────────────────────────────────────────────────────────

def compute_skill_overlap(
    resume_skills: list[str], jd_skills: list[str]
) -> dict:
    """
    Compute skill overlap between resume and job description.
    Uses skill normalization to match aliases (e.g. ML → Machine Learning).

    Returns
    -------
    dict with keys:
        - matched: list of skills present in both (display names)
        - missing: list of skills in JD but not in resume
        - extra: list of skills in resume but not in JD
        - overlap_pct: percentage overlap (0–100)
    """
    # Normalize to canonical forms
    resume_norm = {}
    for s in resume_skills:
        canonical = normalize_skill(s)
        # Keep the best display name (prefer the original casing)
        if canonical not in resume_norm:
            resume_norm[canonical] = s

    jd_norm = {}
    for s in jd_skills:
        canonical = normalize_skill(s)
        if canonical not in jd_norm:
            jd_norm[canonical] = s

    resume_keys = set(resume_norm.keys())
    jd_keys = set(jd_norm.keys())

    matched_keys = resume_keys & jd_keys
    missing_keys = jd_keys - resume_keys
    extra_keys = resume_keys - jd_keys

    overlap_pct = (len(matched_keys) / max(len(jd_keys), 1)) * 100

    # Return display names (title-cased canonical names)
    def display(canonical: str, source_map: dict) -> str:
        original = source_map.get(canonical, canonical)
        return original.title() if original == original.lower() else original

    return {
        "matched": sorted(display(k, resume_norm) for k in matched_keys),
        "missing": sorted(display(k, jd_norm) for k in missing_keys),
        "extra": sorted(display(k, resume_norm) for k in extra_keys),
        "overlap_pct": round(overlap_pct, 1),
    }
