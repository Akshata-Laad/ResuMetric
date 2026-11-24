# 🎯 Resume Scorer & Gap Analyzer — BERT-Powered

A comprehensive **Resume Scoring and Gap Analysis system** that uses **BERT (all-MiniLM-L6-v2)** semantic embeddings to deeply compare a resume against a job description. It identifies skill gaps, quantifies requirement alignment, and generates actionable recommendations to maximize ATS pass-through rates.

## ✨ Features

| Feature | Description |
|---|---|
| **BERT Semantic Scoring** | Uses `all-MiniLM-L6-v2` sentence embeddings with attention-weighted mean pooling for deep semantic similarity |
| **Granular Requirement Matching** | Each JD requirement is matched against every resume bullet using cosine similarity |
| **Hierarchical Skill Taxonomy** | 150+ skills across 7 categories (ML/AI, Cloud, Languages, Databases, etc.) with synonym normalization |
| **Experience & Education Gap Detection** | Automatically extracts years of experience and degree levels from both documents |
| **Section-Level Analysis** | Evaluates relevance of each resume section (Summary, Skills, Experience, Projects, Education) to the JD |
| **Actionable Recommendations** | Generates targeted suggestions: missing skills, weak bullets, ATS formatting tips |
| **PDF + TXT Support** | Parses PDF resumes (via pypdf) and plain text |
| **Supabase Authentication** | Built-in User Sign-up, Login, and Session management with personalized history tracking |
| **Streamlit Web UI** | Interactive drag-and-drop web interface with login sidebar, history tab, score cards, and JSON export |
| **FastAPI REST API** | Full REST backend (`/api/auth/*`, `/api/analyze/*`, `/api/history`) with Swagger UI |
| **Rich CLI** | Beautiful terminal output with tables, progress bars, and color-coded scores |

## 📁 Project Structure

```
ResumeProject/
├── analyzer/
│   ├── __init__.py              # Package exports
│   ├── models.py                # Pydantic data models & schemas
│   ├── parser.py                # Resume & JD text extraction and section parsing
│   ├── bert_engine.py           # BERT embedding engine (singleton, cached, batched)
│   ├── skill_extractor.py       # Skill taxonomy with 150+ keywords & synonym normalization
│   ├── gap_analyzer.py          # Semantic gap analysis (requirements, skills, experience, education)
│   └── scorer.py                # Composite scoring engine & recommendation generator
├── sample_data/
│   ├── sample_resume_ml_engineer.txt
│   ├── sample_jd_senior_ml_engineer.txt
│   └── sample_resume.pdf
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   └── test_skill_extractor.py
├── app.py                       # Streamlit Web UI
├── cli.py                       # Rich CLI
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the CLI

```bash
python cli.py <resume_path> <jd_path>
```

**Example:**
```bash
python cli.py sample_data/sample_resume_ml_engineer.txt sample_data/sample_jd_senior_ml_engineer.txt
```

**JSON output:**
```bash
python cli.py sample_data/sample_resume.pdf sample_data/sample_jd_senior_ml_engineer.txt --json
```

### 3. Run the Streamlit Web UI

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Upload or paste your resume and JD, click **Analyze Resume**, and view the full interactive report.

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## 🧠 How It Works

```
Resume (PDF/TXT)          Job Description (TXT)
       │                           │
       ▼                           ▼
  ┌─────────┐               ┌─────────────┐
  │ Parser  │               │  JD Parser  │
  │(sections│               │(requirements│
  │ bullets)│               │  keywords)  │
  └────┬────┘               └──────┬──────┘
       │                           │
       ▼                           ▼
  ┌──────────────────────────────────────┐
  │       BERT Engine (MiniLM-L6-v2)     │
  │  Embedding → Mean Pool → Normalize   │
  └──────────────────────────────────────┘
       │                           │
       ▼                           ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Semantic │  │  Skill   │  │Exp / Edu │
  │  Matrix  │  │  Overlap │  │   Gaps   │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │             │
       ▼              ▼             ▼
  ┌──────────────────────────────────────┐
  │   Composite Scorer (Weighted ATS)    │
  │  35% Semantic + 35% Skills           │
  │  + 15% Experience + 15% Education    │
  └──────────────────────────────────────┘
       │
       ▼
  Recommendations, Bullet Tips, ATS Tips
```

## 📊 Scoring Breakdown

| Dimension | Weight | What it measures |
|---|---|---|
| **Semantic Alignment** | 35% | BERT cosine similarity between resume content and JD requirements |
| **Skills Match** | 35% | Percentage of JD-required skills found in resume (taxonomy-based) |
| **Experience** | 15% | Required vs. candidate years of experience |
| **Education** | 15% | Required vs. candidate degree level |

### Match Tiers

| Score | Tier |
|---|---|
| ≥ 82% | 🟢 Strong Match |
| ≥ 68% | 🟡 Good Match |
| ≥ 50% | 🟠 Moderate Match |
| < 50% | 🔴 Low Match |

## 🔧 Architecture Highlights

- **Singleton BERT Engine** — model loads once and is reused across requests
- **Embedding Cache** — repeated text strings are not re-encoded
- **Batched Inference** — tokenizes and encodes in batches of 16 for throughput
- **TF-IDF Fallback** — if PyTorch/BERT cannot be loaded, falls back to TF-IDF cosine similarity
- **Local-First Model Loading** — discovers cached HuggingFace snapshots to avoid network calls
