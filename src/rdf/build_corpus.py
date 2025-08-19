# src/rdf/build_corpus.py
import os, csv, re, uuid, hashlib, argparse
from typing import Optional, Dict, Tuple, List
from rdflib import Graph, Namespace, Literal, RDF, XSD, URIRef

LOM = Namespace("http://example.org/lom#")

# ---------------------------
# Common helpers
# ---------------------------

def add_obj(
    g: Graph,
    uri: URIRef,
    title: str,
    topic: str,
    difficulty,
    text: str,
    source: str = "Project Gutenberg",
    extras: Optional[Dict[str, str]] = None,
):
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
    # optional extra triples (work_id, work_title, author, chunk_index, etc.)
    if extras:
        for k, v in extras.items():
            if v is None:
                continue
            g.add((uri, getattr(LOM, k), Literal(v)))

def _hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def chunk_text(text: str, target_words: int = 240, overlap_words: int = 40, min_words: int = 120) -> List[str]:
    """
    Greedy paragraph packer with overlap; falls back to whole text if needed.
    """
    text = text.replace("\r\n", "\n").strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text]

    chunks, buf, wc = [], [], 0
    for p in paras:
        w = p.split()
        if wc + len(w) <= target_words or wc < min_words:
            buf.append(p); wc += len(w)
        else:
            if wc >= min_words:
                combined = " ".join(buf).strip()
                chunks.append(combined)
                # overlap tail
                if overlap_words > 0:
                    tail = " ".join(combined.split()[-overlap_words:])
                    buf, wc = ([tail, p], len(tail.split()) + len(w))
                else:
                    buf, wc = ([p], len(w))
            else:
                buf.append(p); wc += len(w)
    if buf and wc >= min_words:
        chunks.append(" ".join(buf).strip())
    if not chunks:
        chunks = [text]
    return chunks

def guess_topic(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["river", "forest", "sea", "storm", "whale", "mountain", "wind", "boat", "island"]):
        return "Nature"
    if any(k in t for k in ["king", "queen", "war", "battle", "emperor", "castle"]):
        return "History"
    if any(k in t for k in ["school", "teacher", "student", "class"]):
        return "Education"
    return "General"

def estimate_difficulty(text: str) -> int:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s for s in sents if s.strip()]
    if not sents:
        return 5
    avg = sum(len(s.split()) for s in sents) / max(1, len(sents))
    score = 2 + int((min(max(avg, 10), 40) - 10) / 30 * 7)  # map ~10–40 words/sent → 2..9
    return max(1, min(10, score))

def read_metadata_csv(path: Optional[str]) -> Dict[str, Dict]:
    """
    CSV columns: filename,title,topic,difficulty,author,source
    Keyed by LOWERCASED filename.
    """
    metas = {}
    if not path or not os.path.isfile(path):
        return metas
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip().lower()
            if not fn:
                continue
            metas[fn] = {
                "title": (row.get("title") or "").strip(),
                "topic": (row.get("topic") or "").strip(),
                "difficulty": int(row["difficulty"]) if (row.get("difficulty") or "").strip().isdigit() else None,
                "author": (row.get("author") or "").strip(),
                "source": (row.get("source") or "Project Gutenberg").strip(),
            }
    return metas

def parse_filename_meta(filename: str) -> Tuple[str, Optional[str], Optional[int]]:
    """
    Optional pattern: Title__Topic__Difficulty.txt
    e.g. 'Moby_Dick__Nature__9.txt'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("__")
    title = parts[0].replace("_", " ").strip() or "Untitled"
    topic = parts[1].strip() if len(parts) > 1 else None
    difficulty = int(parts[2]) if len(parts) > 2 and (parts[2] or "").isdigit() else None
    return title, topic, difficulty

# ---------------------------
# Mode 1: build from CSV (your original flow)
# ---------------------------

def build_from_csv(csv_path, out_ttl="rdf/corpus.ttl",
                   title_col="title", topic_col="topic",
                   diff_col="difficulty", text_col="text",
                   source="Project Gutenberg"):
    g = Graph()
    g.bind("lom", LOM)
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            title = (row.get(title_col) or "").strip() or "Untitled"
            topic = (row.get(topic_col) or "").strip() or "General"
            diff  = (row.get(diff_col) or "7").strip()
            text  = (row.get(text_col) or "").strip()
            if not text:
                continue
            uri = LOM[f"obj-{uuid.uuid4().hex}"]
            add_obj(g, uri, title, topic, diff, text, source=source)
    g.serialize(destination=out_ttl, format="turtle")
    print(f"✅ Wrote {out_ttl} with {len(g)} triples (CSV mode).")

# ---------------------------
# Mode 2: build from .txt stories + metadata, with chunking
# ---------------------------

def build_from_texts(
    input_dir: str,
    out_ttl: str = "rdf/corpus.ttl",
    metadata_csv: Optional[str] = None,
    target_words: int = 240,
    overlap_words: int = 40,
    min_words: int = 120,
):
    os.makedirs(os.path.dirname(out_ttl), exist_ok=True)
    metas = read_metadata_csv(metadata_csv)
    g = Graph()
    g.bind("lom", LOM)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read().strip()
        if not full_text:
            continue

        meta = metas.get(fname.lower(), {})
        title_fn, topic_fn, diff_fn = parse_filename_meta(fname)

        title = meta.get("title") or title_fn
        topic = meta.get("topic") or topic_fn or guess_topic(full_text)
        difficulty = meta.get("difficulty") if meta.get("difficulty") is not None \
                     else (diff_fn if diff_fn is not None else estimate_difficulty(full_text))
        author = meta.get("author") or ""
        source = meta.get("source") or "Project Gutenberg"

        work_id = _hash(fname)  # stable id per story file
        chunks = chunk_text(full_text, target_words=target_words, overlap_words=overlap_words, min_words=min_words)

        for idx, chunk in enumerate(chunks):
            uri = LOM[f"obj-{work_id}-{idx:03d}"]
            extras = {
                "work_id": work_id,
                "work_title": title,
                "author": author or None,
                "chunk_index": idx,
            }
            add_obj(
                g, uri,
                title=f"{title} (chunk {idx+1})",
                topic=topic,
                difficulty=difficulty,
                text=chunk,
                source=source,
                extras=extras
            )

    g.serialize(destination=out_ttl, format="turtle")
    print(f"✅ Wrote {out_ttl} with {len(g)} triples (texts mode).")

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--csv", help="CSV with text column (legacy mode).")
    mode.add_argument("--texts", help="Folder of .txt stories (new mode).")
    p.add_argument("--out", default="rdf/corpus.ttl", help="Output TTL path.")
    # CSV options
    p.add_argument("--title_col", default="title")
    p.add_argument("--topic_col", default="topic")
    p.add_argument("--diff_col", default="difficulty")
    p.add_argument("--text_col", default="text")
    p.add_argument("--source", default="Project Gutenberg")
    # Texts mode options
    p.add_argument("--metadata", default=None, help="Optional metadata CSV (filename,title,topic,difficulty,author,source).")
    p.add_argument("--target_words", type=int, default=240)
    p.add_argument("--overlap_words", type=int, default=40)
    p.add_argument("--min_words", type=int, default=120)
    args = p.parse_args()

    if args.csv:
        build_from_csv(
            args.csv,
            out_ttl=args.out,
            title_col=args.title_col,
            topic_col=args.topic_col,
            diff_col=args.diff_col,
            text_col=args.text_col,
            source=args.source
        )
    else:
        build_from_texts(
            input_dir=args.texts,
            out_ttl=args.out,
            metadata_csv=args.metadata,
            target_words=args.target_words,
            overlap_words=args.overlap_words,
            min_words=args.min_words
        )



# # src/rdf/build_corpus.py
# import csv, uuid
# from rdflib import Graph, Namespace, Literal, RDF, XSD

# LOM = Namespace("http://example.org/lom#")

# def add_obj(g, uri, title, topic, difficulty, text, source="Projetct Gutenberg"):
#     g.add((uri, RDF.type, LOM.LearningObject))
#     g.add((uri, LOM.general_title, Literal(title)))
#     g.add((uri, LOM.topic, Literal(topic)))
#     try:
#         diff = int(difficulty)
#         g.add((uri, LOM.educational_difficulty, Literal(diff, datatype=XSD.integer)))
#     except Exception:
#         g.add((uri, LOM.educational_difficulty, Literal(-1, datatype=XSD.integer)))
#     g.add((uri, LOM.text, Literal(text)))
#     g.add((uri, LOM.source, Literal(source)))

# def build_from_csv(csv_path, out_ttl="rdf/corpus.ttl",
#                    title_col="title", topic_col="topic",
#                    diff_col="difficulty", text_col="text",
#                    source="Projetct Gutenberg"):
#     g = Graph()
#     g.bind("lom", LOM)
#     with open(csv_path, newline="", encoding="utf-8") as f:
#         r = csv.DictReader(f)
#         for row in r:
#             title = (row.get(title_col) or "").strip()
#             topic = (row.get(topic_col) or "").strip()
#             diff  = (row.get(diff_col) or "7").strip()
#             text  = (row.get(text_col) or "").strip()
#             if not text or not topic:
#                 continue
#             uri = LOM[f"obj-{uuid.uuid4().hex}"]
#             add_obj(g, uri, title or "Untitled", topic, diff, text, source=source)
#     g.serialize(destination=out_ttl, format="turtle")
#     print(f"Wrote {out_ttl} with {len(g)} triples.")

# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--csv", required=True)
#     p.add_argument("--out", default="rdf/corpus.ttl")
#     p.add_argument("--title_col", default="title")
#     p.add_argument("--topic_col", default="topic")
#     p.add_argument("--diff_col", default="difficulty")
#     p.add_argument("--text_col", default="text")
#     p.add_argument("--source", default="Projetct Gutenberg")
#     args = p.parse_args()
#     build_from_csv(args.csv, out_ttl=args.out,
#                    title_col=args.title_col, topic_col=args.topic_col,
#                    diff_col=args.diff_col, text_col=args.text_col,
#                    source=args.source)
