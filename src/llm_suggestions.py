"""
LLM-Powered Resume Suggestions
================================

Optional module — requires GROQ_API_KEY in .env.
Uses Groq API with llama-3.3-70b-versatile for fast, cost-efficient resume improvement suggestions.
Gracefully disabled when no API key is configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.config import GROQ_API_KEY

logger = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class Suggestion:
    """A single improvement suggestion."""
    category: str       # e.g., "Bullet Point", "Skills", "Formatting"
    severity: str       # "high", "medium", "low"
    original: str       # The problematic text (if applicable)
    suggestion: str     # The improvement suggestion
    explanation: str    # Why this matters


@dataclass
class SuggestionResult:
    """Collection of resume improvement suggestions."""
    suggestions: list[Suggestion] = field(default_factory=list)
    summary: str = ""
    llm_available: bool = False
    error: Optional[str] = None


# ── LLM Availability Check ────────────────────────────────────────────────

def is_llm_available() -> bool:
    """Check if the Groq API key is configured."""
    return bool(GROQ_API_KEY)


# ── Suggestion Generation ─────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert resume reviewer and career coach. 
Analyze the resume against the job description and provide specific, actionable improvement suggestions.

Focus on:
1. BULLET POINTS: Check for STAR method (Situation, Task, Action, Result), strong action verbs, quantifiable results
2. SKILLS GAP: Skills mentioned in the JD but missing from the resume
3. KEYWORD OPTIMIZATION: Important terms from the JD that should appear in the resume
4. FORMATTING: Section structure, readability, professional tone
5. IMPACT: Where the candidate can better quantify their achievements

Return your analysis as a JSON object with this structure:
{
    "summary": "1-2 sentence overall assessment",
    "suggestions": [
        {
            "category": "Bullet Point|Skills|Keywords|Formatting|Impact",
            "severity": "high|medium|low",
            "original": "the problematic text from the resume (or empty string)",
            "suggestion": "specific improvement suggestion",
            "explanation": "why this matters for this role"
        }
    ]
}

Limit to 8-10 most impactful suggestions. Be specific and actionable."""


def suggest_improvements(
    resume_text: str,
    jd_text: str,
    max_suggestions: int = 10,
) -> SuggestionResult:
    """
    Generate LLM-powered resume improvement suggestions via Groq API.

    Parameters
    ----------
    resume_text : str
        Raw resume text.
    jd_text : str
        Job description text.
    max_suggestions : int
        Maximum number of suggestions to return.

    Returns
    -------
    SuggestionResult
    """
    if not is_llm_available():
        return SuggestionResult(
            llm_available=False,
            summary="LLM suggestions are disabled. Set GROQ_API_KEY in .env to enable.",
        )

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        user_message = (
            f"## Resume:\n{resume_text[:4000]}\n\n"
            f"## Job Description:\n{jd_text[:2000]}\n\n"
            f"Provide up to {max_suggestions} improvement suggestions."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2000,
        )

        import json
        content = response.choices[0].message.content
        data = json.loads(content)

        suggestions = []
        for s in data.get("suggestions", [])[:max_suggestions]:
            suggestions.append(
                Suggestion(
                    category=s.get("category", "General"),
                    severity=s.get("severity", "medium"),
                    original=s.get("original", ""),
                    suggestion=s.get("suggestion", ""),
                    explanation=s.get("explanation", ""),
                )
            )

        return SuggestionResult(
            suggestions=suggestions,
            summary=data.get("summary", ""),
            llm_available=True,
        )

    except ImportError:
        return SuggestionResult(
            llm_available=False,
            error="groq package not installed. Run: pip install groq",
        )
    except Exception as exc:
        logger.error("Groq LLM suggestion failed: %s", exc)
        return SuggestionResult(
            llm_available=True,
            error=f"Groq API error: {str(exc)}",
        )
