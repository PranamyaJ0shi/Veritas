"""
Training and model comparison pipeline for Fake News Detection.
Trains, benchmarks, and saves:
1. TF-IDF + Logistic Regression (Baseline)
2. Linguistic Features + Random Forest
3. Hybrid Pipeline (TF-IDF + Linguistic Features) + Logistic Regression / Ensemble
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import sys
# Allow running from project root or src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_data, get_train_test_splits
from src.features import LinguisticFeatureExtractor
from src.evaluate import evaluate_model, print_metrics_table

def train_and_compare(dataset_path: str = "data/raw/news_dataset.csv"):
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"[INFO] Dataset not found at {dataset_path}. Generating seed dataset...")
        from data.build_dataset import generate_augmented_dataset
        df_gen = generate_augmented_dataset(target_samples=350)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        df_gen.to_csv(dataset_path, index=False)
        print(f"[INFO] Generated dataset with {len(df_gen)} samples.")

    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = load_data(dataset_path)
    print(f"[INFO] Total clean samples: {len(df)} | Genuine: {(df['label'] == 0).sum()} | Fake: {(df['label'] == 1).sum()}")

    X_train_df, X_test_df, y_train, y_test = get_train_test_splits(df, test_size=0.2, random_state=42)
    print(f"[INFO] Training set: {len(X_train_df)} samples | Testing set: {len(X_test_df)} samples\n")

    results = []

    # =========================================================================
    # MODEL 1: TF-IDF + Logistic Regression (Baseline)
    # =========================================================================
    print("--> Training Model 1: Baseline TF-IDF + Logistic Regression...")
    tfidf_baseline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=42))
    ])
    tfidf_baseline.fit(X_train_df['clean_full_text'], y_train)

    y_pred_m1 = tfidf_baseline.predict(X_test_df['clean_full_text'])
    y_prob_m1 = tfidf_baseline.predict_proba(X_test_df['clean_full_text'])[:, 1]
    res_m1 = evaluate_model(y_test, y_pred_m1, y_prob_m1, model_name="1. TF-IDF + Logistic Regression")
    results.append(res_m1)

    # =========================================================================
    # MODEL 2: Linguistic Features + Random Forest
    # =========================================================================
    print("--> Training Model 2: Linguistic Features + Random Forest...")
    ling_extractor = LinguisticFeatureExtractor()
    X_train_ling = ling_extractor.transform(X_train_df)
    X_test_ling = ling_extractor.transform(X_test_df)

    scaler = StandardScaler()
    X_train_ling_scaled = scaler.fit_transform(X_train_ling)
    X_test_ling_scaled = scaler.transform(X_test_ling)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_model.fit(X_train_ling_scaled, y_train)

    y_pred_m2 = rf_model.predict(X_test_ling_scaled)
    y_prob_m2 = rf_model.predict_proba(X_test_ling_scaled)[:, 1]
    res_m2 = evaluate_model(y_test, y_pred_m2, y_prob_m2, model_name="2. Linguistic Features + Random Forest")
    results.append(res_m2)

    # =========================================================================
    # MODEL 3: Hybrid Combined Pipeline (TF-IDF + Linguistic Features)
    # =========================================================================
    print("--> Training Model 3: Hybrid (TF-IDF + Linguistic Features) + Logistic Regression...")
    
    # Feature union with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True), 'clean_full_text'),
            ('ling_features', Pipeline([
                ('extractor', LinguisticFeatureExtractor()),
                ('scaler', StandardScaler())
            ]), ['title', 'text'])
        ]
    )

    hybrid_pipeline = Pipeline([
        ('features', preprocessor),
        ('clf', LogisticRegression(C=1.5, max_iter=500, random_state=42))
    ])

    hybrid_pipeline.fit(X_train_df, y_train)

    y_pred_m3 = hybrid_pipeline.predict(X_test_df)
    y_prob_m3 = hybrid_pipeline.predict_proba(X_test_df)[:, 1]
    res_m3 = evaluate_model(y_test, y_pred_m3, y_prob_m3, model_name="3. Hybrid (TF-IDF + Linguistic) + LogReg")
    results.append(res_m3)

    # Print benchmark summary
    summary_table = print_metrics_table(results)

    # Select Best Model based on F1-Score and Accuracy
    best_res = max(results, key=lambda x: (x['F1-Score'], x['Accuracy']))
    print(f"[BEST MODEL SELECTED]: {best_res['Model']} (F1: {best_res['F1-Score']}, Acc: {best_res['Accuracy']})")

    # Save artifacts
    model_artifact_path = os.path.join("models", "best_model.joblib")
    joblib.dump(hybrid_pipeline, model_artifact_path)

    # Also save baseline and standalone linguistic model for flexibility
    joblib.dump(tfidf_baseline, os.path.join("models", "tfidf_baseline.joblib"))
    joblib.dump({
        "extractor": ling_extractor,
        "scaler": scaler,
        "model": rf_model,
        "feature_names": ling_extractor.feature_names_
    }, os.path.join("models", "ling_rf_model.joblib"))

    # Save evaluation metrics json
    metrics_path = os.path.join("models", "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[SAVED] Production model pipeline saved to: {model_artifact_path}")
    print(f"[SAVED] Evaluation metrics saved to: {metrics_path}")

    return hybrid_pipeline, results

if __name__ == "__main__":
    train_and_compare()
