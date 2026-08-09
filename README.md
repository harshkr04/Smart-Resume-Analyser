# 📄 Smart Resume Analyser

AI-powered resume analysis tool that uses **semantic matching**, **ATS scoring**, **skill extraction**, and **job recommendations** to help candidates optimize their resumes for specific job descriptions.

> **Skill taxonomy sourced from [ESCO](https://esco.ec.europa.eu/) (European Skills, Competences, Qualifications and Occupations) via [skillNer](https://github.com/AnasAito/SkillNER).**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🎯 Semantic Matching** | Uses `all-MiniLM-L6-v2` sentence embeddings for true meaning-based resume-to-JD comparison |
| **📊 ATS Scoring** | Checks formatting, section coverage, bullet points, keyword density (0–100) |
| **🔍 Gap Analysis** | Identifies missing skills and keywords for the target role |
| **💼 Job Recommendations** | Ranked job matches using embedding similarity against 60+ roles |
| **🏷️ Resume Classification** | Categorizes resume into 24 job categories (TF-IDF + RandomForest) |
| **🤖 AI Suggestions** | Optional GPT-4o-mini powered improvement tips (requires API key) |
| **📥 PDF Report** | Downloadable analysis report with all scores and recommendations |
| **📄 Multi-format Parsing** | Supports PDF and DOCX resume uploads |

---

## 🏗️ Architecture

```
Smart-Resume-Analyser/
├── src/                        # Core ML/NLP modules
│   ├── parser.py               #   PDF/DOCX text extraction
│   ├── extractor.py            #   Skill, education, contact extraction (spaCy + ESCO)
│   ├── matcher.py              #   Semantic similarity (sentence-transformers)
│   ├── scorer.py               #   ATS scoring, gap analysis
│   ├── recommender.py          #   Embedding-based job recommendation
│   ├── classifier.py           #   Resume category classification
│   ├── llm_suggestions.py      #   Optional LLM suggestions (OpenAI)
│   ├── report.py               #   PDF report generation
│   └── config.py               #   Centralized configuration
├── app/
│   ├── streamlit_app.py        # Main Streamlit UI
│   ├── components/             # Reusable UI components
│   └── assets/style.css        # Custom dark theme
├── scripts/
│   ├── validate_data.py        # Data quality & leakage detection
│   └── train_classifier.py     # Classifier training with CV + metrics
├── data/
│   └── skills_taxonomy.json    # ESCO-sourced skill taxonomy
├── models/                     # Saved model artifacts (gitignored)
├── Dockerfile                  # HuggingFace Spaces deployment
└── requirements.txt
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.9+
- pip

### 1. Clone & Install

```bash
git clone https://github.com/harshkr04/Smart-Resume-Analyser.git
cd Smart-Resume-Analyser

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 3. (Optional) Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env to add your OpenAI API key (optional, for AI suggestions)
```

### 4. (Optional) Download Training Data

Download both CSVs from [Kaggle](https://www.kaggle.com/datasets/noorsaeed/resume-datasets) and place them in the `data/` directory:
- `clean_resume_data.csv` — for classifier training
- `jobs_dataset_with_features.csv` — for job recommendations

### 5. (Optional) Validate Data & Train Classifier

```bash
# Validate data quality and check for leakage
python scripts/validate_data.py --data-dir data/

# Train the resume category classifier
python scripts/train_classifier.py --data-dir data/ --output-dir models/

# Include DistilBERT comparison (requires torch + transformers)
python scripts/train_classifier.py --data-dir data/ --output-dir models/ --include-bert
```

### 6. Run the App

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

> **⏱️ First boot takes 30–60 seconds** to download the sentence-transformer model (~80MB) and spaCy model (~12MB). Subsequent runs use cached models instantly.

---

## 🌐 Deployment

### Primary: HuggingFace Spaces (Recommended)

HuggingFace Spaces provides **16GB RAM** on the free CPU tier — enough for the full model stack.

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** as the SDK
3. Push this repo to the Space:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/smart-resume-analyser
   git push hf main
   ```
4. The Dockerfile handles everything: model downloads, port config, etc.

### Fallback: Streamlit Community Cloud

> ⚠️ **RAM Limitation**: Streamlit Cloud has a **1GB RAM limit**. The full model stack (sentence-transformers + spaCy + PDF parsing) may cause OOM errors. Use this only for a lightweight demo.

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repo, set main file as `app/streamlit_app.py`
4. Add `OPENAI_API_KEY` to Streamlit secrets (optional)

### Docker (Self-hosted / Render)

```bash
docker build -t smart-resume-analyser .
docker run -p 7860:7860 smart-resume-analyser
```

---

## 🔧 Model Caching Strategy

| Model | Size | Download | Cache |
|-------|------|----------|-------|
| `all-MiniLM-L6-v2` | ~80MB | First run (auto) | `@st.cache_resource` — loads once per session |
| `en_core_web_sm` (spaCy) | ~12MB | `spacy download` | `@st.cache_resource` — loads once per session |
| TF-IDF + RF classifier | ~5MB | `train_classifier.py` | Disk (`models/`) |
| Job embeddings | ~1MB | First recommendation call | Disk (`models/job_embeddings.npy`) |

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | Enables LLM-powered resume improvement suggestions |
| `HF_HOME` | No | Override Hugging Face model cache directory |

---

## 📊 Training & Evaluation

The classifier is trained on `clean_resume_data.csv` (2,484 resumes, 24 categories) using stratified 5-fold cross-validation. The training script reports:

- Accuracy, F1 (macro), F1 (weighted) per model
- Per-class precision/recall/F1
- Confusion matrix
- Model size and training time comparison

> **Note on `jobs_dataset_with_features.csv`**: This dataset shows 100% classifier accuracy because the `Features` column restates each role's keywords verbatim (target leakage). We use it **exclusively as an embedding-based job database** for similarity search — no classifier is trained on it.

---

## 📦 Datasets

- **Resume Dataset**: [Kaggle — Resume Datasets](https://www.kaggle.com/datasets/noorsaeed/resume-datasets)
  - `clean_resume_data.csv` — 2,484 labeled resumes across 24 categories
  - `jobs_dataset_with_features.csv` — ~1.6M job entries across 60+ roles

---

## 🛠️ Tech Stack

- **NLP**: sentence-transformers, spaCy, skillNer (ESCO)
- **ML**: scikit-learn, (optional) transformers + PyTorch
- **UI**: Streamlit, Plotly
- **Parsing**: pdfplumber, python-docx
- **Reports**: fpdf2
- **LLM**: OpenAI GPT-4o-mini (optional)

---

## 📄 License

MIT
