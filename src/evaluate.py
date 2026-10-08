"""
Evaluation module for Fake News Detection.
Calculates and formats classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC,
and generates confusion matrix representations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model") -> dict:
    """
    Computes standard evaluation metrics.
    y_true: Ground truth binary labels (0 = reliable, 1 = fake/unreliable)
    y_pred: Predicted labels
    y_prob: Predicted probability for positive class (fake)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float('nan')
    else:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred)
    # [ [TN, FP], [FN, TP] ]
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    report = classification_report(y_true, y_pred, target_names=["Likely Genuine", "Likely Unreliable"], output_dict=True)

    results = {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1), 4),
        "ROC-AUC": round(float(auc), 4) if not np.isnan(auc) else "N/A",
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "Full Report": report
    }
    return results

def print_metrics_table(results_list: list):
    """
    Prints a markdown-style comparison table for the evaluated models.
    """
    summary_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy": f"{r['Accuracy']:.4f}",
            "Precision": f"{r['Precision']:.4f}",
            "Recall": f"{r['Recall']:.4f}",
            "F1-Score": f"{r['F1-Score']:.4f}",
            "ROC-AUC": f"{r['ROC-AUC']}" if r["ROC-AUC"] != "N/A" else "N/A"
        }
        for r in results_list
    ])
    print("\n" + "=" * 70)
    print("           MODEL BENCHMARK & COMPARISON SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")
    return summary_df
