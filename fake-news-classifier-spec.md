# Fake News Detection using NLP and Machine Learning

**Type:** Mini Project, Semester 7 BE (Computer Science - Design), Mumbai University
**Goal:** Build a system that classifies a news article as *likely fake* or *likely genuine* using its text and linguistic features, with a simple web interface.

> **Important framing:** A text-only model cannot verify facts. It learns writing-style patterns that correlate with misinformation (sensational tone, clickbait, emotional language, no attribution). So the output should be presented as **"likely unreliable / likely reliable" with a confidence score**, not as absolute truth. Mention this in the report and the UI.

---

## 1. Scope

**Must have (core):**
- Train a classifier on a public fake news dataset
- Compare at least 2-3 models using proper metrics
- Linguistic feature engineering
- A simple UI where the user pastes article text and gets a prediction + confidence

**Good to have:**
- Fine-tuned transformer (RoBERTa / DistilBERT)
- Explainability (highlight words that influenced the prediction)
- URL input with automatic article extraction

**Out of scope:** Fact-checking against external sources, image/video analysis, social media propagation analysis.

---

## 2. Tech Stack

| Part | Tools |
|---|---|
| Language | Python 3.10+ |
| Data / ML | pandas, numpy, scikit-learn, XGBoost or LightGBM |
| NLP | NLTK, spaCy, TextBlob / VADER, textstat |
| Deep learning (optional) | PyTorch, Hugging Face `transformers`, `datasets` |
| Explainability | SHAP or LIME |
| Backend | FastAPI (or Flask) |
| Frontend | Streamlit (fastest) **or** React |
| Article extraction (optional) | `trafilatura` or `newspaper3k` |
| Environment | Google Colab (free GPU) for training, local for the app |

---

## 3. Datasets

Use at least one for training and, ideally, a different one for testing.

| Dataset | Notes |
|---|---|
| **FakeNewsNet** (PolitiFact + GossipCop) | Full articles with labels. Good main choice. |
| **LIAR** | ~12.8k short political statements, 6 truth labels (can merge into binary). |
| **ISOT Fake News** | Reuters vs. flagged fake sites. Very easy; models reach ~99% because of leakage (see section 4). |
| **Kaggle "Fake and Real News"** | Similar to ISOT, widely used. |

Optional: add a small hand-collected set of Indian news articles for a demo or test set, since most datasets are US-centric.

---

## 4. Critical Pitfall: Dataset Leakage

In datasets like ISOT, real articles begin with strings like `WASHINGTON (Reuters) -`, while fake ones have different formatting. A model can hit 99% by learning **source formatting** instead of deception. To avoid this:

1. Remove outlet names, datelines, bylines, URLs, "Reuters", "Image via", etc.
2. Split train/test by **source or time**, not just randomly.
3. **Train on one dataset, test on another.** The accuracy drop is the honest result; report it in the project.

---

## 5. Pipeline

### Step 1: Data loading and cleaning
- Merge title + body into one text field (keep the title separately too for headline features)
- Drop duplicates and empty rows
- Remove leakage strings (section 4)
- Lowercase, remove URLs/HTML; **keep punctuation and capitalization for the feature extraction step** (they are signals)
- Stratified train / validation / test split
- Check class balance

### Step 2: Baseline model
- TF-IDF (word 1-2 grams, plus optionally character n-grams), `max_features` around 50k
- Logistic Regression and Linear SVM
- Record accuracy, precision, recall, F1, ROC-AUC. This is the benchmark to beat.

### Step 3: Linguistic feature engineering
Compute these per article (compute before lowercasing where needed):

| Category | Features |
|---|---|
| Length and lexical | word count, avg word length, avg sentence length, type-token ratio |
| Readability | Flesch Reading Ease, Gunning Fog, SMOG (via `textstat`) |
| Style | ALL-CAPS word ratio, exclamation count, question mark count, quote count, ellipsis count |
| Headline | headline length, has number, clickbait phrases ("you won't believe", "shocking", "breaking"), caps ratio |
| Sentiment / emotion | polarity and subjectivity (TextBlob), VADER compound score, NRC emotion counts (anger, fear, etc.) |
| Syntax | POS tag ratios (noun, verb, adjective, adverb, pronoun) via spaCy |
| Entities | named-entity density, count of PERSON / ORG / GPE |
| Credibility cues | count of attribution phrases ("according to", "said", "officials"), hedging words, numeric/statistic count |
| Consistency (optional) | cosine similarity between headline and body embeddings |

### Step 4: Models to compare
1. TF-IDF + Logistic Regression (baseline)
2. Linguistic features only + Random Forest / XGBoost
3. TF-IDF + linguistic features combined + XGBoost or Logistic Regression
4. *(Optional)* Fine-tuned `distilroberta-base` or `roberta-base` (max length 512, 2-3 epochs, lr around 2e-5). Run on Colab GPU.
5. *(Optional)* Hybrid: transformer embedding concatenated with linguistic features

Use cross-validation and a small hyperparameter search (`GridSearchCV` / `RandomizedSearchCV`).

### Step 5: Evaluation
- Accuracy, **precision, recall, F1, ROC-AUC**, confusion matrix
- Cross-dataset test (train on A, test on B)
- Error analysis: read 20-30 misclassified articles and note patterns (satire, opinion pieces, and breaking news are often wrongly flagged)
- Feature importance plot (which linguistic features matter most)

### Step 6: Explainability
- Use SHAP (tree models) or LIME (any model) to show the top words/features behind each prediction
- In the UI, highlight influential words in the text

### Step 7: Save the model
- `joblib.dump` the sklearn pipeline (vectorizer + feature extractor + classifier), or `save_pretrained` for the transformer
- Make sure the **same preprocessing code** is used at training and inference time

---

## 6. Application

### Option A (quick): Streamlit
Single `app.py` that loads the model, accepts pasted text, shows prediction, confidence bar, and top contributing words.

### Option B (full stack): FastAPI + React
- **API:** `POST /predict` with `{ "title": "...", "text": "..." }`
  returns `{ "label": "likely_fake", "confidence": 0.87, "top_features": [...] }`
- **Frontend:** textarea (and optional URL input), "Analyze" button, result card with confidence, highlighted text, and a disclaimer.
- Enable CORS on FastAPI for the React dev server.

### UI must include
- Confidence score, not just a label
- Disclaimer: "This tool analyzes writing style and cannot verify facts. Cross-check with fact-checkers."
- Optional link to fact-checking sites (e.g., Alt News, BOOM, PolitiFact)

---

## 7. Suggested Folder Structure

```
fake-news-detector/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_features.ipynb
│   └── 04_transformer.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── app/
│   ├── api.py          # FastAPI (or app.py for Streamlit)
│   └── frontend/
├── reports/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## 8. Timeline (approx. 4-5 weeks)

| Week | Tasks |
|---|---|
| 1 | Finalize datasets, EDA, cleaning, leakage removal, baseline model |
| 2 | Linguistic feature extraction, classical model comparison |
| 3 | Transformer fine-tuning (optional), cross-dataset evaluation, error analysis |
| 4 | Explainability, save model, build API + UI |
| 5 | Testing, report writing, PPT, demo prep |

---

## 9. Deliverables (typical for a mini project)

- Working code repository with README and `requirements.txt`
- Trained model files
- Working demo (web app)
- Project report: abstract, introduction, literature survey, problem statement, methodology, dataset description, results with tables/graphs, limitations, conclusion and future scope, references
- PPT for presentation

> Check your department's report format and guide's requirements; these vary slightly.

---

## 10. Limitations to Mention in the Report

- Detects **style, not truth**: a well-written lie may pass; a poorly written true story may be flagged
- Bias towards the dataset's time period, topics, and country (mostly US politics)
- Satire and opinion pieces cause false positives
- Models become outdated as misinformation tactics change
- Text-only; no source credibility, author, or social propagation data

## 11. Future Scope

- Add metadata (source reputation, author, publish time)
- Multilingual support (Hindi, Marathi) using models like `xlm-roberta` or `MuRIL`
- Claim-level fact verification using retrieval (FEVER-style)
- Browser extension
- Social media propagation features

---

## 12. Key References

- Shu et al., *FakeNewsNet: A Data Repository with News Content, Social Context and Spatiotemporal Information* (2018)
- Wang, *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection* (2017)
- Ahmed, Traore, Saad, *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques* (2017), ISOT dataset
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach* (2019)
- Lundberg and Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP, 2017)

---

## 13. Quick-Start Checklist for the Builder

- [ ] Download FakeNewsNet or LIAR (plus ISOT for cross-dataset testing)
- [ ] Clean text and strip leakage artifacts
- [ ] TF-IDF + Logistic Regression baseline, record metrics
- [ ] Extract linguistic features, retrain, compare
- [ ] (Optional) Fine-tune DistilRoBERTa on Colab
- [ ] Cross-dataset evaluation + error analysis
- [ ] Add SHAP/LIME explanations
- [ ] Build Streamlit or FastAPI + React app with confidence and disclaimer
- [ ] Write report and prepare demo
