"""
Data Acquisition & Validation Script
=====================================

Run BEFORE any training to:
1. Check that Kaggle CSVs exist in data/
2. Inspect clean_resume_data.csv: row count, class balance, min-sample warning
3. Inspect jobs_dataset_with_features.csv: detect target leakage
4. Print a validation report with recommendations

Usage:
    python scripts/validate_data.py --data-dir data/
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────

RESUME_CSV = "clean_resume_data.csv"
JOBS_CSV = "jobs_dataset_with_features.csv"

KAGGLE_URL = "https://www.kaggle.com/datasets/noorsaeed/resume-datasets"

MIN_SAMPLES_PER_CLASS = 30  # Below this, CV folds will be unreliable


# ── Helpers ────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_warning(msg: str) -> None:
    print(f"  ⚠️  WARNING: {msg}")


def print_ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def print_info(msg: str) -> None:
    print(f"  ℹ️  {msg}")


# ── Validation Functions ──────────────────────────────────────────────────

def check_files_exist(data_dir: Path) -> dict[str, bool]:
    """Check whether the required CSVs are present."""
    results = {}
    for fname in [RESUME_CSV, JOBS_CSV]:
        path = data_dir / fname
        exists = path.is_file()
        results[fname] = exists
        if exists:
            size_mb = path.stat().st_size / (1024 * 1024)
            print_ok(f"{fname} found ({size_mb:.1f} MB)")
        else:
            print_warning(f"{fname} NOT FOUND in {data_dir}")
    return results


def validate_resume_data(data_dir: Path) -> None:
    """Inspect clean_resume_data.csv for class balance and quality."""
    print_header("Validating: clean_resume_data.csv")

    path = data_dir / RESUME_CSV
    if not path.is_file():
        print_warning(f"File not found. Download from: {KAGGLE_URL}")
        return

    df = pd.read_csv(path)

    # Basic stats
    print_info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print_info(f"Columns: {list(df.columns)}")
    print_info(f"Null counts:\n{df.isnull().sum().to_string()}\n")

    # Identify the category/target column
    cat_col = None
    for candidate in ["Category", "category", "label", "Label"]:
        if candidate in df.columns:
            cat_col = candidate
            break

    if cat_col is None:
        print_warning("Could not find a category column. Check CSV structure manually.")
        return

    # Class distribution
    counts = df[cat_col].value_counts()
    print_info(f"Number of classes: {len(counts)}")
    print(f"\n  Class distribution:\n{counts.to_string()}\n")

    # Flag under-represented classes
    small_classes = counts[counts < MIN_SAMPLES_PER_CLASS]
    if len(small_classes) > 0:
        print_warning(
            f"{len(small_classes)} class(es) have < {MIN_SAMPLES_PER_CLASS} samples — "
            f"stratified CV may fail for these:\n{small_classes.to_string()}"
        )
    else:
        print_ok(f"All classes have ≥ {MIN_SAMPLES_PER_CLASS} samples.")

    # Check for text column
    text_col = None
    for candidate in ["Resume", "resume", "Resume_str", "text", "Text", "Clean_Resume"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col:
        avg_len = df[text_col].astype(str).str.len().mean()
        print_info(f"Avg resume text length: {avg_len:.0f} chars (column: '{text_col}')")
    else:
        print_warning("Could not find a text column. Check CSV structure manually.")


def validate_jobs_data(data_dir: Path) -> None:
    """Inspect jobs_dataset_with_features.csv for target leakage."""
    print_header("Validating: jobs_dataset_with_features.csv")

    path = data_dir / JOBS_CSV
    if not path.is_file():
        print_warning(f"File not found. Download from: {KAGGLE_URL}")
        return

    # Read a sample first (file can be very large)
    df_sample = pd.read_csv(path, nrows=5)
    print_info(f"Columns: {list(df_sample.columns)}")
    print(f"\n  First 3 rows:\n{df_sample.head(3).to_string()}\n")

    # Full read for leakage check (may take a moment for 1.6M rows)
    print_info("Loading full dataset for leakage analysis (this may take a moment)...")
    df = pd.read_csv(path)
    print_info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Identify role/target column
    role_col = None
    for candidate in ["Role", "role", "Job_Role", "job_role", "Title", "title"]:
        if candidate in df.columns:
            role_col = candidate
            break

    # Identify features column
    feat_col = None
    for candidate in ["Features", "features", "Description", "description"]:
        if candidate in df.columns:
            feat_col = candidate
            break

    if role_col is None or feat_col is None:
        print_warning(f"Could not identify role (found: {role_col}) or features (found: {feat_col}) column.")
        return

    unique_roles = df[role_col].nunique()
    print_info(f"Unique roles: {unique_roles}")
    print_info(f"Role distribution (top 10):\n{df[role_col].value_counts().head(10).to_string()}\n")

    # ── TARGET LEAKAGE DETECTION ──
    print_header("Target Leakage Detection")

    def check_leakage(row):
        """Check if the role name appears verbatim in the features text."""
        role = str(row[role_col]).lower().strip()
        features = str(row[feat_col]).lower()
        # Check if any word from the role appears in features
        role_words = set(role.split())
        features_words = set(features.split())
        overlap = role_words & features_words
        return len(overlap) / max(len(role_words), 1)

    # Sample for speed (check 10K rows)
    sample_size = min(10_000, len(df))
    sample = df.sample(n=sample_size, random_state=42)
    sample["_leakage_score"] = sample.apply(check_leakage, axis=1)

    avg_leakage = sample["_leakage_score"].mean()
    full_leakage_pct = (sample["_leakage_score"] >= 0.5).mean() * 100

    print_info(f"Avg role-word overlap in features: {avg_leakage:.2%}")
    print_info(f"Rows with ≥50% role words in features: {full_leakage_pct:.1f}%")

    if full_leakage_pct > 50:
        print_warning(
            f"TARGET LEAKAGE CONFIRMED — {full_leakage_pct:.1f}% of rows contain the role's "
            f"keywords in the features column.\n"
            f"  ❌ Do NOT train a classifier on this dataset.\n"
            f"  ✅ Use it as an embedding-based job database for similarity search only."
        )
    elif full_leakage_pct > 20:
        print_warning(
            f"POSSIBLE LEAKAGE — {full_leakage_pct:.1f}% overlap detected. Investigate manually."
        )
    else:
        print_ok(f"No significant leakage detected ({full_leakage_pct:.1f}% overlap).")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate training data before model training.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the CSV files (default: data/)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print_header("Smart Resume Analyser — Data Validation")

    if not data_dir.is_dir():
        print_warning(f"Data directory '{data_dir}' does not exist.")
        print_info(f"Create it and download CSVs from: {KAGGLE_URL}")
        sys.exit(1)

    # Step 1: Check files exist
    print_header("File Check")
    file_status = check_files_exist(data_dir)

    if not any(file_status.values()):
        print_info(f"\nNo data files found. Download both CSVs from:\n  {KAGGLE_URL}")
        print_info("Place them in the data/ directory, then re-run this script.")
        sys.exit(0)

    # Step 2: Validate each dataset
    if file_status.get(RESUME_CSV, False):
        validate_resume_data(data_dir)

    if file_status.get(JOBS_CSV, False):
        validate_jobs_data(data_dir)

    # Summary
    print_header("Validation Complete")
    print_info("Next steps:")
    print_info("  1. If clean_resume_data.csv is valid → run scripts/train_classifier.py")
    print_info("  2. jobs_dataset_with_features.csv → used as job embedding DB only (no classifier)")
    print_info("  3. The Streamlit app works without any CSV data (semantic matching is standalone)")


if __name__ == "__main__":
    main()
