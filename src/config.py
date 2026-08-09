"""
Centralized Configuration
=========================

All paths, model names, and environment variable loading for the project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────
load_dotenv()
# Fallback: also try .env.example if .env doesn't exist
_env_example = Path(__file__).resolve().parent.parent / ".env.example"
if _env_example.is_file():
    load_dotenv(_env_example)

# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
APP_DIR = PROJECT_ROOT / "app"
ASSETS_DIR = APP_DIR / "assets"

# ── Model Configuration ───────────────────────────────────────────────────
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
SPACY_MODEL_NAME = "en_core_web_sm"

# ── Data Files ─────────────────────────────────────────────────────────────
SKILLS_TAXONOMY_PATH = DATA_DIR / "skills_taxonomy.json"
RESUME_DATA_CSV = DATA_DIR / "clean_resume_data.csv"
JOBS_DATA_CSV = DATA_DIR / "jobs_dataset_with_features.csv"

# ── Saved Model Artifacts ─────────────────────────────────────────────────
CLASSIFIER_MODEL_PATH = MODELS_DIR / "resume_classifier.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
JOB_EMBEDDINGS_PATH = MODELS_DIR / "job_embeddings.npy"
JOB_ROLES_PATH = MODELS_DIR / "job_roles.pkl"

# ── API Keys (optional) ───────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    # Fallback: check if key stored in OPENAI_API_KEY is actually a Groq key (starts with gsk_)
    _opt_key = os.getenv("OPENAI_API_KEY", "").strip()
    if _opt_key.startswith("gsk_"):
        GROQ_API_KEY = _opt_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ── Scoring Weights ────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "semantic_similarity": 0.40,
    "skill_overlap": 0.30,
    "experience_match": 0.15,
    "education_match": 0.15,
}

# ── ATS Section Names (expected in a well-formatted resume) ────────────────
ATS_EXPECTED_SECTIONS = [
    "contact",
    "summary",
    "objective",
    "skills",
    "experience",
    "education",
]

# ── Resume Length Thresholds (chars) ───────────────────────────────────────
RESUME_MIN_LENGTH = 200
RESUME_MAX_LENGTH = 8000
RESUME_IDEAL_MIN = 400
RESUME_IDEAL_MAX = 4000
