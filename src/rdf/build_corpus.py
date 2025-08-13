# src/data/build_corpus.py
import csv, uuid
from rdflib import Graph, Namespace, Literal, RDF, XSD

LOM = Namespace("http://example.org/lom#")

def add_obj(g, uri, title, topic, difficulty, text, source="Projetct Gutenberg"):
    g.add((uri, RDF.type, LOM.LearningObject))
    g.add((uri, LOM.general_title, Literal(title)))
    g.add((uri, LOM.topic, Literal(topic)))
    try:
        diff = int(difficulty)
        g.add((uri, LOM.educational_difficulty, Literal(diff, datatype=XSD.integer)))
    except Exception:
        g.add((uri, LOM.educational_difficulty, Literal(-1, datatype=XSD.integer)))
    g.add((uri, LOM.text, Literal(text)))
    g.add((uri, LOM.source, Literal(source)))

def build_from_csv(csv_path, out_ttl="rdf/corpus.ttl",
                   title_col="title", topic_col="topic",
                   diff_col="difficulty", text_col="text",
                   source="Projetct Gutenberg"):
    g = Graph()
    g.bind("lom", LOM)
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            title = (row.get(title_col) or "").strip()
            topic = (row.get(topic_col) or "").strip()
            diff  = (row.get(diff_col) or "7").strip()
            text  = (row.get(text_col) or "").strip()
            if not text or not topic:
                continue
            uri = LOM[f"obj-{uuid.uuid4().hex}"]
            add_obj(g, uri, title or "Untitled", topic, diff, text, source=source)
    g.serialize(destination=out_ttl, format="turtle")
    print(f"Wrote {out_ttl} with {len(g)} triples.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="rdf/corpus.ttl")
    p.add_argument("--title_col", default="title")
    p.add_argument("--topic_col", default="topic")
    p.add_argument("--diff_col", default="difficulty")
    p.add_argument("--text_col", default="text")
    p.add_argument("--source", default="Projetct Gutenberg")
    args = p.parse_args()
    build_from_csv(args.csv, out_ttl=args.out,
                   title_col=args.title_col, topic_col=args.topic_col,
                   diff_col=args.diff_col, text_col=args.text_col,
                   source=args.source)
