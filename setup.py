"""Smart Resume Analyser — Setup Configuration."""

from setuptools import setup, find_packages

setup(
    name="smart-resume-analyser",
    version="2.0.0",
    description="AI-powered resume analysis: semantic matching, ATS scoring, skill extraction, and job recommendations.",
    author="Harsh Kumar",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.30.0",
        "sentence-transformers>=2.2.0",
        "spacy>=3.7.0",
        "skillNer>=1.0.4",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "pdfplumber>=0.10.0",
        "python-docx>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "fpdf2>=2.7.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "ocr": ["pytesseract>=0.3.10", "Pillow>=10.0.0", "pdf2image>=1.16.0"],
        "training": ["transformers>=4.30.0", "torch>=2.0.0", "matplotlib>=3.7.0", "seaborn>=0.12.0"],
    },
)
