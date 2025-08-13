# src/retrieval/rank_text.py
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

def tfidf_scores(objective: str, passages: List[str]) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
    X = vec.fit_transform([objective] + passages)
    return cosine_similarity(X[0], X[1:]).ravel()

def embed_scores(objective: str, passages: List[str]) -> np.ndarray:
    m = SentenceTransformer("all-MiniLM-L6-v2")
    embs = m.encode([objective] + passages, normalize_embeddings=True)
    return util.cos_sim(embs[0:1], embs[1:]).cpu().numpy().ravel()

def rerank_by_objective(objective: str, passages: List[str], top_k: int = 2) -> List[Tuple[int, float]]:
    if not passages:
        return []
    sims = tfidf_scores(objective, passages)
    if float(np.max(sims)) <= 1e-8:
        sims = embed_scores(objective, passages)
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:max(1, top_k)]
    return [(i, float(s)) for i, s in ranked]
