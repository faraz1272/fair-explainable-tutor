# src/generation/retrieval.py
"""Retrieval utilities for querying RDF passages and ranking by objective.

This module provides helpers to:
- build SPARQL filter clauses for topic and difficulty,
- query learning objects from a Turtle (TTL) corpus, and
- (optionally) re-rank passages using an external objective ranker.

All public functions return plain Python types for easy logging and UI display.
"""

from __future__ import annotations

from typing import List, Dict, Optional

from rdflib import Graph
from rdflib.namespace import Namespace, XSD

# declaring the LOM namespace once for reuse
LOM = Namespace("http://example.org/lom#")


def _topic_filter_clause(topic_csv: str) -> str:
    """Build a SPARQL FILTER for one or more comma-separated topics.

    Args:
        topic_csv: A comma-separated list of topics (e.g., "Physics, Optics").

    Returns:
        A SPARQL FILTER clause string or an empty string when no topics provided.
    """
    topics = [t.strip() for t in (topic_csv or "").split(",") if t.strip()]
    if not topics:
        return ""
    # building OR conditions using lowercase string equality on ?topic
    ors = " || ".join([f'(lcase(str(?topic)) = "{t.lower()}")' for t in topics])
    return f"FILTER ({ors})"


def _difficulty_filter_clause(difficulty: Optional[int], diff_band: int) -> str:
    """Build a SPARQL FILTER to constrain `educational_difficulty` to a band.

    Args:
        difficulty: The central difficulty value, or None to skip filtering.
        diff_band: The +/- integer band around the difficulty value.

    Returns:
        A SPARQL FILTER clause string or an empty string if *difficulty* is None.
    """
    if difficulty is None:
        return ""
    low, high = int(difficulty) - int(diff_band), int(difficulty) + int(diff_band)
    return (
        f"FILTER ( xsd:integer(?difficulty) >= {low} && xsd:integer(?difficulty) <= {high} )"
    )


def query_passages(
    topic: str,
    ttl_path: str = "rdf/corpus.ttl",
    difficulty: Optional[int] = None,
    diff_band: int = 1,
    min_words: int = 60,
    limit: int = 200,
) -> List[Dict]:
    """Query the RDF corpus for learning objects and return passage dicts.

    Args:
        topic: A topic or comma-separated topics to filter on (case-insensitive).
        ttl_path: Path to the Turtle (TTL) RDF corpus.
        difficulty: Optional difficulty anchor for filtering.
        diff_band: Half-width of the difficulty band around *difficulty*.
        min_words: Minimum number of words required for the text field.
        limit: Maximum number of results to return.

    Returns:
        A list of dicts with keys: `uri`, `title`, `difficulty`, `topic`, `text`.
    """
    # parsing the TTL graph from disk
    g = Graph().parse(ttl_path, format="turtle")

    # building the parameterized query
    q = f"""
    PREFIX lom: <{LOM}>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?uri ?title ?difficulty ?topic ?txt WHERE {{
      ?uri a lom:LearningObject ;
           lom:general_title ?title ;
           lom:educational_difficulty ?difficulty ;
           lom:topic ?topic ;
           lom:text ?txt .
      {_topic_filter_clause(topic)}
      {_difficulty_filter_clause(difficulty, diff_band)}
    }}
    LIMIT {int(limit)}
    """

    out: List[Dict] = []

    # executing the query and collecting results
    for row in g.query(q, initNs={"xsd": XSD}):
        uri, title, diff_raw, topic_val, txt = row
        try:
            # converting difficulty to int when possible
            diff = int(str(diff_raw))
        except Exception:
            diff = -1
        text = str(txt).strip()
        # skipping short texts to avoid weak contexts
        if len(text.split()) < int(min_words):
            continue
        out.append(
            {
                "uri": str(uri),
                "title": str(title),
                "difficulty": diff,
                "topic": str(topic_val),
                "text": text,
            }
        )

    return out


# optional adapter to your existing ranker (kept as-is for external importers)
from src.retrieval.rank_text import rerank_by_objective  # keep as-is
