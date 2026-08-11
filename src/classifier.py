"""
Resume Category Classifier
===========================

TF-IDF + RandomForest classifier trained on clean_resume_data.csv.
Used for predicting resume category (e.g., "ENGINEERING", "FINANCE").

Training and DistilBERT comparison live in scripts/train_classifier.py.
This module only handles inference with a pre-trained model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

from src.config import CLASSIFIER_MODEL_PATH, TFIDF_VECTORIZER_PATH

LABEL_ENCODER_PATH = CLASSIFIER_MODEL_PATH.parent / "label_encoder.pkl"

logger = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class CategoryPrediction:
    """A single category prediction with confidence."""
    category: str
    confidence: float  # 0–1

@dataclass
class ClassificationResult:
    """Classification result with top predictions."""
    top_prediction: CategoryPrediction
    all_predictions: list[CategoryPrediction]  # Sorted by confidence (desc)


# ── Classifier ─────────────────────────────────────────────────────────────

class ResumeClassifier:
    """
    Resume category classifier using pre-trained TF-IDF + RF model.
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self._loaded = False

    def load(self) -> bool:
        """Load pre-trained model and vectorizer from disk."""
        if self._loaded:
            return True

        if not CLASSIFIER_MODEL_PATH.is_file() or not TFIDF_VECTORIZER_PATH.is_file():
            logger.warning(
                "Classifier model not found at %s. "
                "Run scripts/train_classifier.py first.",
                CLASSIFIER_MODEL_PATH,
            )
            return False

        try:
            self.model = joblib.load(str(CLASSIFIER_MODEL_PATH))
            self.vectorizer = joblib.load(str(TFIDF_VECTORIZER_PATH))
            if LABEL_ENCODER_PATH.is_file():
                self.label_encoder = joblib.load(str(LABEL_ENCODER_PATH))
                logger.info("Label encoder loaded from %s", LABEL_ENCODER_PATH)
            self._loaded = True
            logger.info("Classifier loaded from %s", CLASSIFIER_MODEL_PATH)
            return True
        except Exception as exc:
            logger.error("Failed to load classifier: %s", exc)
            return False

    def predict(self, resume_text: str, top_n: int = 5) -> Optional[ClassificationResult]:
        """
        Predict the category of a resume.

        Parameters
        ----------
        resume_text : str
            Raw or cleaned resume text.
        top_n : int
            Number of top predictions to return.

        Returns
        -------
        ClassificationResult or None if model not loaded.
        """
        if not self._loaded:
            if not self.load():
                return None

        # Vectorize
        tfidf_vector = self.vectorizer.transform([resume_text])

        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(tfidf_vector)[0]
            classes = self.model.classes_

            # Sort by probability
            sorted_indices = np.argsort(probas)[::-1][:top_n]

            all_preds = [
                CategoryPrediction(
                    category=(
                        self.label_encoder.inverse_transform([classes[idx]])[0]
                        if self.label_encoder is not None
                        else str(classes[idx])
                    ),
                    confidence=round(float(probas[idx]), 4),
                )
                for idx in sorted_indices
            ]
        else:
            # Fallback: just predict
            pred = self.model.predict(tfidf_vector)[0]
            all_preds = [CategoryPrediction(category=str(pred), confidence=1.0)]

        return ClassificationResult(
            top_prediction=all_preds[0],
            all_predictions=all_preds,
        )


# ── Module-level singleton ─────────────────────────────────────────────────

_classifier: Optional[ResumeClassifier] = None


def get_classifier() -> ResumeClassifier:
    """Get or create the classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = ResumeClassifier()
    return _classifier


def classify_resume(resume_text: str, top_n: int = 5) -> Optional[ClassificationResult]:
    """Classify a resume into a job category."""
    clf = get_classifier()
    return clf.predict(resume_text, top_n=top_n)
