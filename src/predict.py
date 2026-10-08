"""
Prediction and inference module for Fake News Detection.
Loads the trained model pipeline, processes new article titles and texts,
and outputs classification labels, confidence scores, linguistic metrics,
and explainability indicators.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_text_for_vectorizer, strip_leakage
from src.features import extract_article_features, CLICKBAIT_TRIGGERS, ATTRIBUTION_CUES
import re

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")

_MODEL_CACHE = None

def get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run src/train.py first!")
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE

# Satirical publishers and parody cues
SATIRE_TRIGGERS = [
    r"\bfauxy\b",
    r"\bthe onion\b",
    r"\bbabylon bee\b",
    r"\bclickhole\b",
    r"\bhumor\b",
    r"\bsatire\b",
    r"\bsatirical\b",
    r"\bparody\b",
    r"\bspoof\b"
]

def find_highlighted_cues(title: str, text: str) -> Dict[str, list]:
    """
    Finds specific trigger phrases, attribution patterns, and satire cues present in the article.
    """
    full_text = f"{title} {text}"
    lowered = full_text.lower()

    detected_clickbait = []
    for trigger in CLICKBAIT_TRIGGERS:
        clean_trig = trigger.replace(r"\b", "").replace(r"\B", "")
        matches = re.findall(trigger, lowered)
        if matches:
            detected_clickbait.append(clean_trig)

    detected_attribution = []
    for cue in ATTRIBUTION_CUES:
        clean_cue = cue.replace(r"\b", "").replace(r"\B", "")
        matches = re.findall(cue, lowered)
        if matches:
            detected_attribution.append(clean_cue)

    detected_satire = []
    for cue in SATIRE_TRIGGERS:
        clean_cue = cue.replace(r"\b", "").replace(r"\B", "")
        matches = re.findall(cue, lowered)
        if matches:
            detected_satire.append(clean_cue)

    return {
        "sensational_cues": list(set(detected_clickbait)),
        "attribution_cues": list(set(detected_attribution)),
        "satire_cues": list(set(detected_satire))
    }

def predict_article(title: str, text: str) -> Dict[str, Any]:
    """
    Inference function for a single news article.
    Returns:
    - label: "Likely Reliable" or "Likely Unreliable (Fake/Sensational)"
    - is_fake: bool (True for fake)
    - confidence: float (0.0 to 1.0)
    - raw_score: float (probability of fake)
    - metrics: dictionary of linguistic features
    - highlights: list of detected sensational / attribution cues
    """
    model = get_model()
    
    clean_title = strip_leakage(title)
    clean_body = strip_leakage(text)
    full_text = f"{clean_title} {clean_body}".strip()
    clean_full = clean_text_for_vectorizer(full_text)

    # Prepare DataFrame matching pipeline input schema
    input_df = pd.DataFrame([{
        "title": clean_title,
        "text": clean_body,
        "full_text": full_text,
        "clean_full_text": clean_full
    }])

    # Predict probability [prob_reliable, prob_fake]
    probs = model.predict_proba(input_df)[0]
    prob_fake = float(probs[1])
    prob_reliable = float(probs[0])

    is_fake = prob_fake >= 0.5
    confidence = prob_fake if is_fake else prob_reliable
    label = "Likely Unreliable" if is_fake else "Likely Reliable"

    linguistic_metrics = extract_article_features(title, text)
    highlights = find_highlighted_cues(title, text)

    is_satire = len(highlights["satire_cues"]) > 0
    if is_satire:
        is_fake = True
        label = "Likely Unreliable (Satirical Parody)"
        prob_fake = max(prob_fake, 0.95)
        prob_reliable = round(1.0 - prob_fake, 4)
        confidence = prob_fake

    return {
        "label": label,
        "is_fake": bool(is_fake),
        "is_satire": bool(is_satire),
        "confidence": round(float(confidence), 4),
        "confidence_percentage": round(float(confidence * 100), 2),
        "prob_fake": round(float(prob_fake), 4),
        "prob_reliable": round(float(prob_reliable), 4),
        "linguistic_metrics": linguistic_metrics,
        "highlights": highlights
    }

if __name__ == "__main__":
    sample_title = "SHOCKING BOMBSHELL: Miracle Lemon Cure Erases All Illness Overnight!"
    sample_text = "You won't believe what doctors just discovered! Corrupt hospitals are begging officials to ban this secret cure!"
    
    print("Testing sample prediction...")
    res = predict_article(sample_title, sample_text)
    print(f"Result: {res['label']} ({res['confidence_percentage']}%)")
    print(f"Sensational Cues Detected: {res['highlights']['sensational_cues']}")
