"""
Job Recommender
===============

Embedding-based job recommendation using sentence-transformers.
Uses jobs_dataset_with_features.csv as a job database — NO classifier
is trained on it (avoids target leakage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from src.config import (
    JOBS_DATA_CSV,
    JOB_EMBEDDINGS_PATH,
    JOB_ROLES_PATH,
    SENTENCE_MODEL_NAME,
)
from src.matcher import compute_embedding, compute_batch_embeddings

logger = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class JobRecommendation:
    """A single job recommendation."""
    role: str
    similarity_score: float  # 0–100
    features_snippet: str    # Brief description excerpt


# ── Job Database ───────────────────────────────────────────────────────────

class JobDatabase:
    """
    Embedding-based job database for similarity search.

    On first call, loads the jobs CSV, deduplicates roles, embeds their
    features text, and caches the embeddings to disk. Subsequent calls
    load from cache.
    """

    def __init__(self):
        self.roles: Optional[list[str]] = None
        self.features: Optional[list[str]] = None
        self.embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def _load_from_cache(self) -> bool:
        """Try loading pre-computed embeddings from disk."""
        if JOB_EMBEDDINGS_PATH.is_file() and JOB_ROLES_PATH.is_file():
            try:
                self.embeddings = np.load(str(JOB_EMBEDDINGS_PATH))
                saved = joblib.load(str(JOB_ROLES_PATH))
                self.roles = saved["roles"]
                self.features = saved["features"]
                self._loaded = True
                logger.info(
                    "Loaded %d job embeddings from cache.", len(self.roles)
                )
                return True
            except Exception as exc:
                logger.warning("Cache load failed: %s", exc)
        return False

    def _build_from_csv(self) -> bool:
        """Build job embeddings from the CSV file."""
        # Check multiple locations for the CSV
        csv_path = None
        for candidate in [JOBS_DATA_CSV, JOBS_DATA_CSV.parent.parent / JOBS_DATA_CSV.name]:
            if candidate.is_file():
                csv_path = candidate
                break

        if csv_path is None:
            logger.warning(
                "Jobs dataset not found at %s. "
                "Job recommendations will be unavailable.",
                JOBS_DATA_CSV,
            )
            return False

        logger.info("Building job embeddings from %s (this may take a while)...", csv_path)

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Failed to read jobs CSV: %s", exc)
            return False

        # Identify columns
        role_col = None
        feat_col = None
        for c in ["Role", "role", "Job_Role", "Title"]:
            if c in df.columns:
                role_col = c
                break
        for c in ["Features", "features", "Description", "description"]:
            if c in df.columns:
                feat_col = c
                break

        if role_col is None or feat_col is None:
            logger.error(
                "Could not find role/features columns. Found: %s",
                list(df.columns),
            )
            return False

        # Deduplicate: take one representative per role
        # Group by role and concatenate features (take first 500 chars of a sample)
        role_features = (
            df.groupby(role_col)[feat_col]
            .first()
            .reset_index()
        )

        self.roles = role_features[role_col].tolist()
        self.features = role_features[feat_col].astype(str).tolist()

        # Compute embeddings
        logger.info("Embedding %d unique roles...", len(self.roles))
        self.embeddings = compute_batch_embeddings(self.features)

        # Cache to disk
        try:
            JOB_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(JOB_EMBEDDINGS_PATH), self.embeddings)
            joblib.dump(
                {"roles": self.roles, "features": self.features},
                str(JOB_ROLES_PATH),
            )
            logger.info("Job embeddings cached to %s", JOB_EMBEDDINGS_PATH)
        except Exception as exc:
            logger.warning("Could not cache embeddings: %s", exc)

        self._loaded = True
        return True

    def load(self) -> bool:
        """Load the job database (from cache or CSV)."""
        if self._loaded:
            return True
        if self._load_from_cache():
            return True
        return self._build_from_csv()

    def search(self, resume_text: str, top_n: int = 10) -> list[JobRecommendation]:
        """
        Find the top-N most similar jobs to the given resume text.

        Parameters
        ----------
        resume_text : str
            Resume text to match against jobs.
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        list[JobRecommendation]
        """
        if not self._loaded:
            if not self.load():
                return []

        if self.embeddings is None or self.roles is None:
            return []

        # Embed the resume
        resume_emb = compute_embedding(resume_text)

        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(self.embeddings, resume_emb)

        # Get top-N indices
        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = []
        for idx in top_indices:
            score = float(similarities[idx]) * 100
            snippet = self.features[idx][:200] + "..." if len(self.features[idx]) > 200 else self.features[idx]
            results.append(
                JobRecommendation(
                    role=self.roles[idx],
                    similarity_score=round(max(0, min(100, score)), 1),
                    features_snippet=snippet,
                )
            )

        return results


# ── Module-level singleton ─────────────────────────────────────────────────

_db: Optional[JobDatabase] = None


def get_job_database() -> JobDatabase:
    """Get or create the job database singleton."""
    global _db
    if _db is None:
        _db = JobDatabase()
    return _db


def recommend_jobs(resume_text: str, top_n: int = 10) -> list[JobRecommendation]:
    """
    Recommend jobs for a given resume.

    Parameters
    ----------
    resume_text : str
        Raw resume text.
    top_n : int
        Number of recommendations.

    Returns
    -------
    list[JobRecommendation]
        Ranked list of job recommendations with similarity scores.
    """
    db = get_job_database()
    return db.search(resume_text, top_n=top_n)
