"""
Linguistic feature engineering module for Fake News Detection.
Implements lexical, stylistic, readability, headline, attribution,
and sensationalism features as a scikit-learn compatible transformer.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

# Clickbait keywords & sensational phrases
CLICKBAIT_TRIGGERS = [
    r"\byou won't believe\b",
    r"\bshocking\b",
    r"\bsecret\b",
    r"\bexposed\b",
    r"\bmiracle\b",
    r"\bbombshell\b",
    r"\bwake up\b",
    r"\bdeep state\b",
    r"\bbig pharma\b",
    r"\bbreaking\b",
    r"\bwhat happened next\b",
    r"\bhate this\b",
    r"\bshare this\b",
    r"\bunbelievable\b",
    r"\bfurious\b",
    r"\bterrifying\b",
    r"\bgreed\b",
    r"\bhoax\b",
    r"\bforbidden\b"
]

# Credibility and Attribution indicators commonly found in reliable journalism
ATTRIBUTION_CUES = [
    r"\baccording to\b",
    r"\bsaid\b",
    r"\bstated\b",
    r"\breported\b",
    r"\bofficials\b",
    r"\bspokesperson\b",
    r"\bconfirmed\b",
    r"\bpublished in\b",
    r"\bfindings\b",
    r"\bresearchers\b",
    r"\binvestigators\b",
    r"\banalysts\b",
    r"\bannounced\b",
    r"\btestified\b"
]

def calculate_readability(text: str):
    """
    Computes readability metrics using textstat if available,
    or robust heuristic fallbacks.
    """
    if not text or len(text.strip()) == 0:
        return 50.0, 10.0

    if HAS_TEXTSTAT:
        try:
            flesch = float(textstat.flesch_reading_ease(text))
            gunning_fog = float(textstat.gunning_fog(text))
            return flesch, gunning_fog
        except Exception:
            pass

    # Fallback heuristic
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    num_words = max(len(words), 1)
    num_sentences = max(len(sentences), 1)
    asl = num_words / num_sentences
    # Syllables heuristic (~1.4 per word avg)
    flesch_est = 206.835 - (1.015 * asl) - 50.0
    gunning_fog_est = 0.4 * (asl + 10.0)
    return max(0.0, min(100.0, flesch_est)), gunning_fog_est

def extract_article_features(title: str, text: str) -> dict:
    """
    Computes linguistic, stylistic, and structural features for a single article.
    """
    title = str(title) if title is not None else ""
    text = str(text) if text is not None else ""
    full_text = f"{title} {text}".strip()

    words = re.findall(r'\b\w+\b', full_text)
    num_words = len(words)
    title_words = re.findall(r'\b\w+\b', title)
    num_title_words = len(title_words)

    # Sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
    num_sentences = max(len(sentences), 1)

    # 1. Length & Lexical
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    avg_sentence_len = num_words / num_sentences
    unique_words = len(set([w.lower() for w in words]))
    ttr = (unique_words / num_words) if num_words > 0 else 0.0  # Type-Token Ratio

    # 2. Orthography & Punctuation
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio = (caps_words / num_words) if num_words > 0 else 0.0
    exclamation_count = full_text.count("!")
    question_count = full_text.count("?")
    quote_count = full_text.count('"') + full_text.count("'")
    ellipsis_count = full_text.count("...")

    # 3. Headline specifics
    title_caps_words = sum(1 for w in title_words if w.isupper() and len(w) > 1)
    title_caps_ratio = (title_caps_words / num_title_words) if num_title_words > 0 else 0.0
    title_has_num = 1.0 if bool(re.search(r'\d', title)) else 0.0
    title_exclamations = float(title.count("!"))

    # 4. Clickbait & Sensationalism triggers
    lowered_full = full_text.lower()
    lowered_title = title.lower()

    clickbait_matches = 0
    for trigger in CLICKBAIT_TRIGGERS:
        clickbait_matches += len(re.findall(trigger, lowered_full))
        if re.search(trigger, lowered_title):
            clickbait_matches += 2  # Double weight in headline

    # 5. Credibility / Attribution cues
    attribution_matches = 0
    for cue in ATTRIBUTION_CUES:
        attribution_matches += len(re.findall(cue, lowered_full))

    # 6. Readability
    flesch, gunning_fog = calculate_readability(full_text)

    # 7. Numeric & Statistical Density (genuine reports cite numbers/dates/percentages)
    numbers_count = len(re.findall(r'\b\d+(?:\.\d+)?%?|\$\d+\b', full_text))
    numbers_ratio = (numbers_count / num_words) if num_words > 0 else 0.0

    return {
        "word_count": float(num_words),
        "avg_word_length": float(avg_word_len),
        "avg_sentence_length": float(avg_sentence_len),
        "type_token_ratio": float(ttr),
        "caps_ratio": float(caps_ratio),
        "exclamation_count": float(exclamation_count),
        "question_count": float(question_count),
        "quote_count": float(quote_count),
        "ellipsis_count": float(ellipsis_count),
        "title_word_count": float(num_title_words),
        "title_caps_ratio": float(title_caps_ratio),
        "title_has_number": float(title_has_num),
        "title_exclamation_count": float(title_exclamations),
        "clickbait_cues_count": float(clickbait_matches),
        "attribution_cues_count": float(attribution_matches),
        "flesch_reading_ease": float(flesch),
        "gunning_fog_index": float(gunning_fog),
        "numbers_ratio": float(numbers_ratio)
    }

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that extracts linguistic feature vectors
    from a DataFrame or iterable with 'title' and 'text' columns.
    """
    def __init__(self):
        self.feature_names_ = [
            "word_count", "avg_word_length", "avg_sentence_length",
            "type_token_ratio", "caps_ratio", "exclamation_count",
            "question_count", "quote_count", "ellipsis_count",
            "title_word_count", "title_caps_ratio", "title_has_number",
            "title_exclamation_count", "clickbait_cues_count",
            "attribution_cues_count", "flesch_reading_ease",
            "gunning_fog_index", "numbers_ratio"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms input X (DataFrame with 'title' and 'text', or list of tuples)
        into a numpy 2D feature matrix.
        """
        if isinstance(X, pd.DataFrame):
            titles = X['title'].values if 'title' in X.columns else [""] * len(X)
            texts = X['text'].values if 'text' in X.columns else X['full_text'].values
        elif isinstance(X, list) or isinstance(X, np.ndarray):
            if len(X) > 0 and isinstance(X[0], (list, tuple)) and len(X[0]) >= 2:
                titles = [item[0] for item in X]
                texts = [item[1] for item in X]
            else:
                titles = [""] * len(X)
                texts = list(X)
        else:
            titles = [""]
            texts = [str(X)]

        feature_rows = []
        for title, text in zip(titles, texts):
            feats = extract_article_features(title, text)
            feature_rows.append([feats[name] for name in self.feature_names_])

        return np.array(feature_rows, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
