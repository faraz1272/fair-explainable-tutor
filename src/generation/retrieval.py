# src/generation/retrieval.py
from typing import List, Dict
import rdflib
from rdflib import Graph
from rdflib.namespace import Namespace, XSD

LOM = Namespace("http://example.org/lom#")

def _topic_filter_clause(topic_csv: str) -> str:
    topics = [t.strip() for t in topic_csv.split(",") if t.strip()]
    if not topics:
        return ""
    ors = " || ".join([f'(lcase(str(?topic)) = "{t.lower()}")' for t in topics])
    return f"FILTER ({ors})"

def _difficulty_filter_clause(difficulty, diff_band) -> str:
    if difficulty is None: return ""
    low, high = int(difficulty) - int(diff_band), int(difficulty) + int(diff_band)
    return f"FILTER ( xsd:integer(?difficulty) >= {low} && xsd:integer(?difficulty) <= {high} )"

def query_passages(topic: str, ttl_path="rdf/corpus.ttl", difficulty=None, diff_band=1,
                   min_words=60, limit=200) -> List[Dict]:
    g = Graph().parse(ttl_path, format="turtle")
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
    LIMIT {limit}
    """
    out = []
    for row in g.query(q, initNs={"xsd": XSD}):
        uri, title, diff_raw, topic_val, txt = row
        try: diff = int(str(diff_raw))
        except: diff = -1
        text = str(txt).strip()
        if len(text.split()) < min_words: continue
        out.append({"uri": str(uri), "title": str(title), "difficulty": diff,
                    "topic": str(topic_val), "text": text})
    return out

# optional adapter to your existing ranker
from src.retrieval.rank_text import rerank_by_objective  # keep as-is