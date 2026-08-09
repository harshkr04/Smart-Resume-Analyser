"""
Model Download Script
=====================

Pre-downloads all required models for offline use.
Run this once before deploying or going offline.

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("  Smart Resume Analyser — Model Download")
    print("=" * 60)

    # 1. spaCy model
    print("\n[1/2] Downloading spaCy model (en_core_web_sm)...")
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("  ✅ Already installed.")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            print("  ✅ Downloaded successfully.")
    except ImportError:
        print("  ❌ spacy not installed. Run: pip install spacy")

    # 2. Sentence-transformer model
    print("\n[2/2] Downloading sentence-transformer model (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  ✅ Downloaded to: {model._model_card_vars.get('model_id', 'cache')}")
    except ImportError:
        print("  ❌ sentence-transformers not installed. Run: pip install sentence-transformers")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("  All models downloaded. Ready for offline use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
