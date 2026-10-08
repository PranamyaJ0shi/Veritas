"""
Preprocessing module for Fake News Detection.
Implements dataset cleaning, leakage removal, and train/test data splitting.
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

# Known dataset leakage patterns (e.g. from ISOT, Reuters, wire services)
LEAKAGE_PATTERNS = [
    r'^[A-Z\s,]+(?:\(Reuters\)|—|–|-)\s*',
    r'\(Reuters\)',
    r'\(CNN\)',
    r'\(AP\)',
    r'Image via [^\n]+',
    r'Featured image via [^\n]+',
    r'By\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+for\s+[^\n]+)?',
    r'https?://\S+|www\.\S+',
    r'<.*?>',
]

def strip_leakage(text: str) -> str:
    """
    Strips known agency datelines, publisher tags, and URLs that introduce
    spurious correlation / data leakage.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text
    for pattern in LEAKAGE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def clean_text_for_vectorizer(text: str) -> str:
    """
    Standard text normalization for TF-IDF:
    lowercased, punctuation removed, excess whitespace collapsed.
    """
    if not isinstance(text, str):
        return ""
    # Strip URLs and leakage first
    text = strip_leakage(text)
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers if desired or collapse digits
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV dataset, performs deduplication and sanitization.
    Expects columns: 'title', 'text', 'label'
    """
    df = pd.read_csv(filepath)
    # Ensure mandatory columns exist
    df['title'] = df['title'].fillna("").astype(str)
    df['text'] = df['text'].fillna("").astype(str)
    df['label'] = df['label'].astype(int)

    # Filter out empty entries
    df = df[df['text'].str.strip().str.len() > 10].copy()
    df = df.drop_duplicates(subset=['title', 'text']).reset_index(drop=True)

    # Combined full text for unified processing
    df['full_text'] = df['title'] + " " + df['text']
    df['clean_full_text'] = df['full_text'].apply(clean_text_for_vectorizer)
    return df

def get_train_test_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Performs stratified train/test split.
    Returns: X_train_df, X_test_df, y_train, y_test
    """
    features = df[['title', 'text', 'full_text', 'clean_full_text']]
    labels = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    return X_train, X_test, y_train, y_test
