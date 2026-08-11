"""
Classifier Training Script
===========================

Trains and evaluates resume category classifiers on clean_resume_data.csv:
1. TF-IDF + RandomForest (baseline)
2. DistilBERT fine-tuned (optional, requires [training] extras)

Reports: stratified 5-fold CV, confusion matrix, precision/recall/F1 per class.
Saves the best model to models/ directory.

Usage:
    python scripts/train_classifier.py --data-dir data/ --output-dir models/

For DistilBERT comparison (requires torch + transformers):
    python scripts/train_classifier.py --data-dir data/ --output-dir models/ --include-bert
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder


# ── Text Cleaning ──────────────────────────────────────────────────────────

def clean_resume_text(text: str) -> str:
    """Clean resume text for classification."""
    text = re.sub(r"http\S+\s?", " ", text)
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"#\S+\s?", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── TF-IDF + RandomForest Training ────────────────────────────────────────

def train_tfidf_rf(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> dict:
    """Train and evaluate TF-IDF + RandomForest with stratified 5-fold CV."""
    print("\n" + "=" * 60)
    print("  Training: TF-IDF + RandomForest")
    print("=" * 60)

    start = time.time()

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(X)

    # Stratified 5-fold cross-validation
    print("\nRunning 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    y_pred = cross_val_predict(rf, X_tfidf, y, cv=skf, n_jobs=-1)

    elapsed = time.time() - start

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    class_names = label_encoder.classes_
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"\n  Per-class report:")
    print(classification_report(y, y_pred, target_names=class_names))

    # Train final model on full data
    print("Training final model on full dataset...")
    rf_final = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_final.fit(X_tfidf, y)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "resume_classifier.pkl"
    vectorizer_path = output_dir / "tfidf_vectorizer.pkl"

    joblib.dump(rf_final, str(model_path))
    joblib.dump(tfidf, str(vectorizer_path))

    model_size = model_path.stat().st_size / (1024 * 1024)
    vec_size = vectorizer_path.stat().st_size / (1024 * 1024)

    print(f"\n  Model saved: {model_path} ({model_size:.1f} MB)")
    print(f"  Vectorizer saved: {vectorizer_path} ({vec_size:.1f} MB)")

    return {
        "model": "TF-IDF + RandomForest",
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "training_time_s": elapsed,
        "model_size_mb": model_size + vec_size,
        "report": report,
    }


# ── DistilBERT Training (optional) ────────────────────────────────────────

def train_distilbert(
    texts: list[str],
    labels: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> dict:
    """Train and evaluate DistilBERT with stratified 5-fold CV."""
    try:
        import torch
        from transformers import (
            DistilBertTokenizer,
            DistilBertForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print("\n  ⚠️  torch/transformers not installed. Skipping DistilBERT.")
        print("  Install with: pip install torch transformers")
        return {"model": "DistilBERT", "error": "Dependencies not installed"}

    print("\n" + "=" * 60)
    print("  Training: DistilBERT (fine-tuned)")
    print("=" * 60)
    print("  ⚠️  This may take 10-30 minutes on CPU.")

    start = time.time()

    num_labels = len(label_encoder.classes_)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Truncate texts to 512 tokens max
    truncated_texts = [t[:2000] for t in texts]  # Rough char limit

    # Simple train/test split for speed (full CV would be very slow)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        truncated_texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Tokenize
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)

    class ResumeDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ResumeDataset(train_encodings, y_train.tolist())
    test_dataset = ResumeDataset(test_encodings, y_test.tolist())

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "distilbert_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    elapsed = time.time() - start

    # Evaluate
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Estimate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    return {
        "model": "DistilBERT (fine-tuned)",
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "training_time_s": elapsed,
        "model_size_mb": model_size,
        "report": report,
        "note": "80/20 split (not full 5-fold CV, for speed). Compare F1 weighted.",
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train resume category classifier.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--include-bert", action="store_true", help="Include DistilBERT comparison")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    csv_path = data_dir / "clean_resume_data.csv"
    if not csv_path.is_file():
        # Try project root
        csv_path = Path("clean_resume_data.csv")
    if not csv_path.is_file():
        print(f"❌ Cannot find clean_resume_data.csv in {data_dir} or project root.")
        print(f"   Download from: https://www.kaggle.com/datasets/noorsaeed/resume-datasets")
        sys.exit(1)

    print("=" * 60)
    print("  Smart Resume Analyser — Classifier Training")
    print("=" * 60)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n  Dataset: {csv_path}")
    print(f"  Shape: {df.shape}")

    # Find columns
    cat_col = None
    text_col = None
    for c in ["Category", "category", "label"]:
        if c in df.columns:
            cat_col = c
            break
    for c in ["Resume", "resume", "Resume_str", "text", "Clean_Resume", "Feature"]:
        if c in df.columns:
            text_col = c
            break

    if cat_col is None or text_col is None:
        print(f"  ❌ Could not find category ({cat_col}) or text ({text_col}) column.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"  Category column: {cat_col}")
    print(f"  Text column: {text_col}")
    print(f"  Classes: {df[cat_col].nunique()}")
    print(f"\n  Class distribution:\n{df[cat_col].value_counts().to_string()}\n")

    # Clean text
    df["cleaned_text"] = df[text_col].astype(str).apply(clean_resume_text)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[cat_col])
    X = df["cleaned_text"].values

    # Save label encoder
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, str(output_dir / "label_encoder.pkl"))

    # Train TF-IDF + RF
    results = []
    rf_result = train_tfidf_rf(X, y, le, output_dir)
    results.append(rf_result)

    # Optionally train DistilBERT
    if args.include_bert:
        bert_result = train_distilbert(X.tolist(), y, le, output_dir)
        results.append(bert_result)

    # Comparison table
    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    print(f"\n  {'Model':<30} {'Accuracy':>10} {'F1 (macro)':>12} {'F1 (wtd)':>10} {'Time (s)':>10} {'Size (MB)':>10}")
    print(f"  {'-'*82}")
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<30} {'SKIPPED':>10}")
        else:
            print(
                f"  {r['model']:<30} {r['accuracy']:>10.4f} {r['f1_macro']:>12.4f} "
                f"{r['f1_weighted']:>10.4f} {r['training_time_s']:>10.1f} {r['model_size_mb']:>10.1f}"
            )

    # Pick winner
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["f1_weighted"])
        print(f"\n  ✅ Best model: {best['model']} (F1 weighted: {best['f1_weighted']:.4f})")
        print(f"\n  📝 Tradeoffs:")
        if len(valid) > 1:
            rf_r = next(r for r in valid if "RandomForest" in r["model"])
            bert_r = next((r for r in valid if "DistilBERT" in r["model"]), None)
            if bert_r:
                print(f"     - TF-IDF+RF: {rf_r['model_size_mb']:.1f}MB, {rf_r['training_time_s']:.0f}s training")
                print(f"     - DistilBERT: {bert_r['model_size_mb']:.1f}MB, {bert_r['training_time_s']:.0f}s training")
                print(f"     - For deployment, TF-IDF+RF is recommended (smaller, faster inference)")
                print(f"     - DistilBERT may shine with more data (>10K samples)")

    print("\n  Done! Models saved to:", output_dir)


if __name__ == "__main__":
    main()
