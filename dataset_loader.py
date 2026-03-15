"""
dataset_loader.py
-----------------
Downloads and caches the HuggingFace healthcare dataset, then exposes a
keyword search helper used by the Healthcare Knowledge Database Tool.

Dataset: harmesh95/healthcare-disease-knowledge
Columns (expected): Disease, Symptom1..SymptomN, Description (varies by split)
"""

import re
import pandas as pd
from typing import Optional
from datasets import load_dataset

_df: Optional[pd.DataFrame] = None  # module-level cache


def _load() -> pd.DataFrame:
    """Download (first run) or use cached dataset and return a DataFrame."""
    global _df
    if _df is not None:
        return _df

    print("[DatasetLoader] Loading healthcare dataset from HuggingFace …")
    raw = load_dataset("harmesh95/healthcare-disease-knowledge", trust_remote_code=True)

    # Combine all available splits into one DataFrame
    frames = []
    for split_name, split_data in raw.items():
        frames.append(split_data.to_pandas())

    _df = pd.concat(frames, ignore_index=True)

    # Normalise column names to lowercase for reliable access
    _df.columns = [c.strip().lower().replace(" ", "_") for c in _df.columns]

    print(f"[DatasetLoader] Loaded {len(_df)} records. Columns: {list(_df.columns)}")
    return _df


def _row_to_text(row: pd.Series, max_chars_per_col: int = 300) -> str:
    """Convert a dataset row into a readable text block."""
    parts = []
    for col in row.index:
        val = row[col]
        if pd.notna(val) and str(val).strip():
            val_str = str(val).strip()
            if len(val_str) > max_chars_per_col:
                val_str = val_str[:max_chars_per_col] + "..."
            parts.append(f"  {col.replace('_', ' ').title()}: {val_str}")
    return "\n".join(parts)


def search_symptoms(query: str, top_k: int = 2) -> str:
    """
    Keyword search over all text columns in the dataset.

    Parameters
    ----------
    query : str
        Symptom keywords entered by the agent (e.g. "fever headache").
    top_k : int
        Maximum number of records to return.

    Returns
    -------
    str
        Formatted text block of matching records, or a not-found message.
    """
    df = _load()
    keywords = [w.lower() for w in re.split(r"\W+", query) if len(w) > 2]

    if not keywords:
        return "No valid keywords found in the query."

    # Build a combined text column for searching
    text_cols = [c for c in df.columns]
    combined = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

    # Score each row by how many keywords it contains
    scores = combined.apply(lambda t: sum(kw in t for kw in keywords))
    ranked = df[scores > 0].copy()
    ranked["_score"] = scores[scores > 0]
    ranked = ranked.sort_values("_score", ascending=False).head(top_k)

    if ranked.empty:
        return (
            f"No records found for '{query}'. "
            "Try using different symptom keywords."
        )

    results = []
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        results.append(f"--- Result {i} ---\n{_row_to_text(row)}")

    return "\n\n".join(results)


# Allow direct testing of the loader
if __name__ == "__main__":
    print(search_symptoms("fever headache"))
