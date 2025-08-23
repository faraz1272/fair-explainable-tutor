# src/retrieval/rank_text.py
"""Ranking passages against an objective using TF–IDF and embeddings.

This module provides:
- `tfidf_scores`: *computing* cosine similarities via TF–IDF n‑grams.
- `embed_scores`: *computing* semantic similarities via SentenceTransformer embeddings.
- `rerank_by_objective`: *ranking* passages by relevance to the objective, with a
  fallback from TF–IDF to embeddings when lexical overlap is negligible.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


def tfidf_scores(objective: str, passages: List[str]) -> np.ndarray:
    """Return cosine similarities between *objective* and *passages* via TF–IDF.

    We *fit* a TF–IDF vectorizer (1–2 grams, English stopwords), *vectorize*
    objective + passages, and *compute* cosine similarities.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X = vec.fit_transform([objective] + passages)
    return cosine_similarity(X[0], X[1:]).ravel()


def embed_scores(objective: str, passages: List[str]) -> np.ndarray:
    """Return cosine similarities between *objective* and *passages* via embeddings.

    We *encode* objective + passages using a SentenceTransformer, *normalize*
    embeddings, and *compute* cosine similarities.
    """
    m = SentenceTransformer("all-MiniLM-L6-v2")
    embs = m.encode([objective] + passages, normalize_embeddings=True)
    return util.cos_sim(embs[0:1], embs[1:]).cpu().numpy().ravel()


def rerank_by_objective(objective: str, passages: List[str], top_k: int = 2) -> List[Tuple[int, float]]:
    """Rank passages by similarity to the objective using TF–IDF with embedding fallback.

    Args:
        objective: Query or objective string.
        passages: Candidate passage texts.
        top_k: Number of top results to return (≥1).

    Returns:
        A list of (index, similarity) pairs, sorted by similarity descending.
    """
    if not passages:
        return []

    # computing lexical similarities first
    sims = tfidf_scores(objective, passages)

    # falling back to semantic similarity if lexical signal is negligible
    if float(np.max(sims)) <= 1e-8:
        sims = embed_scores(objective, passages)

    # sorting and truncating results
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
    return [(i, float(s)) for i, s in ranked]
