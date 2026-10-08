# Veritas — Fake News & Credibility Detection System




## 📌 Project Overview & Framing

> **Important Academic Framing:** A text-only machine learning model cannot verify empirical facts in real time. Rather, it identifies **linguistic, stylistic, and structural patterns** that strongly correlate with misinformation (e.g. sensationalist phrasing, excessive punctuation, high ALL-CAPS density, lack of attribution, clickbait headline triggers). The predictions are presented as **"Likely Unreliable" vs. "Likely Reliable"** with a continuous confidence percentage.

---

## 🏗️ Architecture & Pipeline

```text
                        ┌───────────────────────────────┐
                        │   Input News (Title + Text)   │
                        └───────────────┬───────────────┘
                                        │
                         [ Data Preprocessing & Cleaning ]
                         • Strip dataset leakage strings
                         • URL and HTML removal
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
    [ TF-IDF Vectorizer ]                       [ Linguistic Feature Extractor ]
    • Word 1-2 n-grams                          • Readability (Flesch, Fog)
    • Sublinear term frequency                  • Orthography (ALL-CAPS, !, ?)
                                                • Headline clickbait triggers
                                                • Journalistic attribution cues
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        │
                          [ ColumnTransformer Union ]
                                        │
                        ┌───────────────▼───────────────┐
                        │   Hybrid Logistic Classifier   │
                        └───────────────┬───────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
     Predicted Credibility Class                    Confidence & Linguistic
  (Likely Reliable / Unreliable)                    Explainability Breakdown
```

---

## 📂 Project Directory Structure

```text
ML project 2/
├── data/
│   ├── raw/
│   │   └── news_dataset.csv         # Balanced curated dataset of real & fake news
│   ├── processed/
│   └── build_dataset.py             # Script to generate / expand dataset
├── src/
│   ├── __init__.py
│   ├── preprocess.py                # Cleaning, anti-leakage filters, train/test split
│   ├── features.py                  # Scikit-learn LinguisticFeatureExtractor
│   ├── train.py                     # Multi-model training & benchmark comparison
│   ├── evaluate.py                  # Accuracy, Precision, Recall, F1, ROC-AUC metrics
│   └── predict.py                   # Inference engine with explainability cues
├── models/
│   ├── best_model.joblib            # Production hybrid model pipeline
│   ├── tfidf_baseline.joblib        # Baseline TF-IDF model
│   ├── ling_rf_model.joblib         # Pure linguistic feature Random Forest
│   └── evaluation_metrics.json      # Benchmark comparison metrics
├── app/
│   └── app.py                       # Interactive Streamlit Web Application
├── fake-news-classifier-spec.md     # Project specification document
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

---

## 📊 Models Implemented & Compared

In adherence with the project specification, three distinct architectures are benchmarked:

1. **Model 1 (Baseline Benchmark):** TF-IDF Vectorizer (word n-grams 1-2) + Logistic Regression.
2. **Model 2 (Linguistic Stylometry):** 18 engineered linguistic features (lexical diversity, readability indices, punctuation, capitalization, attribution cues) + Random Forest.
3. **Model 3 (Hybrid Production Pipeline):** Feature Union (`ColumnTransformer`) combining TF-IDF and scaled linguistic feature vectors + Regularized Logistic Regression.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

### 2. Prepare / Generate Dataset

To generate or rebuild the curated balanced dataset:

```bash
python data/build_dataset.py
```

### 3. Train & Compare Models

Run the training pipeline to evaluate all models, compute classification metrics, and save the best pipeline:

```bash
python src/train.py
```

### 4. Run Single-Article Inference (CLI)

Test the inference script directly from the terminal:

```bash
python src/predict.py
```

### 5. Launch the Streamlit Web Application

Launch the interactive web UI:

```bash
streamlit run app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🌟 Web UI Features

- **Online URL Article Extraction:** Enter any live news article link (e.g., BBC, The Hindu, NDTV, Reuters, Substack); the system extracts clean journalistic text (stripping navigation, ads, and footers) using `trafilatura` and tests credibility instantly.
- **Preset Example Loader:** 1-click evaluation of genuine scientific reports, financial releases, or viral conspiracy/clickbait news.
- **Custom Input:** Paste any custom news headline and body.
- **Confidence Meter:** Color-coded status badge and probability distribution.
- **Explainability Highlights:**
  - 🚨 Sensational / Clickbait triggers detected
  - 📰 Journalistic attribution cues identified
- **Linguistic Metrics Display:** Word count, Lexical diversity (TTR), Flesch Reading Ease score, Gunning Fog index, ALL-CAPS ratio, and punctuation density.
- **Benchmarks Tab:** Comparison table of models, accuracy, precision, recall, F1, and confusion matrix.
- **External Fact-Checkers:** Direct links to PolitiFact, Alt News, BOOM FactCheck, and Snopes.

---

## ⚠️ Academic Limitations & Ethical Considerations

- **Style vs. Truth:** The model evaluates writing style, sentiment, and journalistic cues rather than checking real-world facts against ground truth.
- **Satire & Sarcasm:** Satirical articles (e.g. *The Onion*) may be flagged as unreliable due to exaggerated language.
- **Temporal & Domain Shift:** Model accuracy may vary across different regional domains or evolving misinformation styles.
