# src/generate/generate_lesson.py
import rdflib
from rdflib import Graph
from rdflib.namespace import Namespace, XSD
from typing import List, Dict
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import json
from functools import lru_cache


from src.retrieval.rank_text import rerank_by_objective

load_dotenv()  # load OpenAI API key from .env file

FORCE_CPU = False

LOM = Namespace("http://example.org/lom#")

# --- Heuristic cause→effect extraction from raw text ---
def _sentences(txt: str):
    # super light sentence split
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt) if s.strip()]

def _norm(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip(" ;,.-–—")
    return s

def _mk_pair(cause: str, effect: str) -> str:
    cause = _norm(cause)
    effect = _norm(effect)
    if not cause or not effect:
        return ""
    pair = f"{cause} -> {effect}"
    # keep pairs readable
    return pair if 8 <= len(pair) <= 160 else ""

def extract_pairs_from_texts(texts, max_pairs=5):
    """
    Extract cause→effect pairs from a list of text passages.
    Uses simple regex patterns to catch common causal connectors.
    Returns a list of strings in the form 'cause -> effect'.
    """
    pairs = []
    seen = set()

    def _norm(s: str) -> str:
        return re.sub(r'\s+', ' ', s).strip(" ;,.-–—")

    def _mk_pair(cause: str, effect: str) -> str:
        cause = _norm(cause)
        effect = _norm(effect)
        if cause and effect:
            pair = f"{cause} -> {effect}"
            if 8 <= len(pair) <= 160 and pair.lower() not in seen:
                seen.add(pair.lower())
                return pair
        return ""

    # Common causal connectors and their split direction
    cue_patterns = [
        (r'^(.*?)\bbecause\b\s+(.*)$', False),         # cause because effect → swap
        (r'^\s*because\s+(.*?),\s+(.*)$', True),       # because cause, effect
        (r'^(.*?)\bdue to\b\s+(.*)$', False),
        (r'^\s*due to\s+(.*?),\s+(.*)$', True),
        (r'^(.*?)\bowing to\b\s+(.*)$', False),
        (r'^\s*owing to\s+(.*?),\s+(.*)$', True),
        (r'^(.*?),\s+so that\s+(.*)$', False),
        (r'^(.*?),\s+so\s+(.*)$', False),
        (r'^(.*?)\bled to\b\s+(.*)$', False),
        (r'^(.*?)\bresulted in\b\s+(.*)$', False),
        (r'^(.*?)\bcauses\b\s+(.*)$', False),
        (r'^(.*?)\bcaused\b\s+(.*)$', False),
        (r'^(.*?)\bcausing\b\s+(.*)$', False),
        (r'^(.*?)\bforcing\b\s+(.*)$', False),
        (r'^(.*?)\btherefore\b\s+(.*)$', False),
        (r'^(.*?)\bthus\b\s+(.*)$', False),
        (r'^(.*?)\bconsequently\b\s+(.*)$', False),
        (r'^(.*?)\bhence\b\s+(.*)$', False),
    ]

    for txt in texts:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt) if s.strip()]
        for s in sentences:
            for pattern, swap in cue_patterns:
                m = re.match(pattern, s, flags=re.IGNORECASE)
                if m:
                    if swap:
                        cause, effect = m.group(1), m.group(2)
                    else:
                        cause, effect = m.group(1), m.group(2)
                    pair = _mk_pair(cause, effect)
                    if pair:
                        pairs.append(pair)
                    break  # stop after first matching pattern
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    return pairs[:max_pairs]

def _topic_filter_clause(topic_csv: str) -> str:
    """
    Build a SPARQL FILTER that matches any of the comma-separated topics (case-insensitive).
    Example: "Nature, Fantasy" -> FILTER( lcase(str(?topic)) = "nature" || lcase(str(?topic)) = "fantasy" )
    """
    topics = [t.strip() for t in topic_csv.split(",") if t.strip()]
    if not topics:
        return ""  # no filter -> match all topics
    ors = " || ".join([f'(lcase(str(?topic)) = "{t.lower()}")' for t in topics])
    return f"FILTER ({ors})"

def _difficulty_filter_clause(difficulty: int | None, diff_band: int) -> str:
    if difficulty is None:
        return ""
    low = int(difficulty) - int(diff_band)
    high = int(difficulty) + int(diff_band)
    return f"FILTER ( xsd:integer(?difficulty) >= {low} && xsd:integer(?difficulty) <= {high} )"

def query_passages(
    topic: str,
    ttl_path: str = "data/rdf/corpus.ttl",
    difficulty: int | None = None,
    diff_band: int = 1,
    min_words: int = 60,
    limit: int = 200,
) -> List[Dict]:
    """
    Query RDF for learning objects matching topic(s) and optional difficulty band.
    - Case-insensitive topic match; supports comma-separated topics.
    - Pulls lom:text in the same query.
    - Skips passages shorter than min_words.
    """
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
    # initNs gives xsd prefix to rdflib’s parser/evaluator
    for row in g.query(q, initNs={"xsd": XSD}):
        uri, title, diff_raw, topic_val, txt = row
        # safe difficulty cast
        try:
            diff = int(str(diff_raw))
        except Exception:
            diff = -1
        text = str(txt).strip()
        if len(text.split()) < min_words:
            continue
        out.append({
            "uri": str(uri),
            "title": str(title),
            "difficulty": diff,
            "topic": str(topic_val),
            "text": text,
        })
    return out

def build_prompt(objective: str, context_items: List[Dict]) -> str:
    snippets = "\n\n".join(
        [f"Title: {x['title']}\nText: {x['text']}" for x in context_items]
    )
    return (
        "You are a fair and explainable tutoring assistant.\n"
        f"Learning objective: {objective}\n\n"
        "Use the context texts below to produce a short lesson (120–180 words) and 3 comprehension questions. "
        "Avoid stereotypes; be inclusive.\n\n"
        f"Context:\n{snippets}\n\n"
        "Output format:\n"
        "Lesson:\n"
        "Q1:\nQ2:\nQ3:\n"
    )

def simple_fairness_scan(text: str) -> Dict:
    """Toy checker for obviously problematic terms; extend later."""
    flagged = []
    bad_terms = ["stupid", "lazy", "crazy"]  # placeholder demo list
    lower = text.lower()
    for t in bad_terms:
        if t in lower:
            flagged.append(t)
    return {"flagged_terms": flagged, "passed": len(flagged) == 0}

def explain_selection(topic: str, difficulty: int, sim_score: float) -> str:
    return (
        f"Selected because: topic='{topic}', difficulty={difficulty}, "
        f"objective_similarity={sim_score:.2f}."
    )

def _device_map():
    if FORCE_CPU:
        return {"": "cpu"}
    # Let HF pick (will choose mps:0 if available)
    return "auto"

@lru_cache(maxsize=2)
def _load_pipe(model_id: str):
    """Load and cache model/tokenizer/pipeline once per model_id."""
    print(f">>> Loading model (cached): {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=_device_map(),
        # torch_dtype=torch.float32  # Uncomment to force CPU-friendly precision
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tok)
    return tok, pipe

def hf_chat_generate(messages,
                     model_id: str = "microsoft/Phi-3.5-mini-instruct",
                     max_new_tokens: int = 260,   # trimmed for speed
                     temperature: float = 0.2) -> str:
    tok, pipe = _load_pipe(model_id)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        eos_token_id=tok.eos_token_id,
    )[0]["generated_text"]
    return out[len(prompt):].lstrip() if out.startswith(prompt) else out

def build_context_snippets(context_items):
    return "\n\n".join([f"Title: {x['title']}\nText:\n{x['text']}" for x in context_items])

def postprocess_lesson(lesson_text: str) -> str:
    text = lesson_text.strip()

    # Keep from 'Lesson:' onward if model prepended stuff
    if "Lesson:" in text:
        text = text[text.index("Lesson:"):]

    # Remove stray headings (Lecture/Objective/etc.)
    text = re.sub(r'(?im)^\s*(lecture|objective|duration|materials)\s*:.*\n', '', text)

    # De-dup consecutive sentences
    sents = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in sents:
        if s and (not cleaned or s.lower() != cleaned[-1].lower()):
            cleaned.append(s)
    text = " ".join(cleaned)

    # Ensure order: Lesson then Q1..Q3
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].strip().lower().startswith("q1:"):
        qs = [ln for ln in lines if re.match(r'(?i)^q[1-9]:', ln)]
        body = "\n".join([ln for ln in lines if ln not in qs])
        text = f"{body}\n" + "\n".join(qs)

    # Exactly 3 questions
    lines = [ln for ln in text.splitlines() if ln.strip()]
    qs_idx = [i for i,l in enumerate(lines) if re.match(r'(?i)^q[1-9]:', l)]
    if len(qs_idx) < 3:
        need = 3 - len(qs_idx)
        start = len(qs_idx) + 1
        lines += [f"Q{start+i}: Identify one cause and its effect described in the lesson." for i in range(need)]
    elif len(qs_idx) > 3:
        kept, n = [], 0
        for ln in lines:
            if re.match(r'(?i)^q[1-9]:', ln):
                if n < 3:
                    kept.append(ln); n += 1
            else:
                kept.append(ln)
        lines = kept

    # Clamp lesson to 160 words max
# Clamp to ~160 words, keeping whole sentences
    try:
        li = lines.index("Lesson:")
        body_lines = []
        for j in range(li+1, len(lines)):
            if re.match(r'(?i)^q[1-9]:', lines[j]): break
            body_lines.append(lines[j])
        body = " ".join(body_lines).strip()

        words = body.split()
        if len(words) > 175:
            # split into sentences, rebuild until ≤160 words
            sents = re.split(r'(?<=[.!?])\s+', body)
            keep, wc = [], 0
            for s in sents:
                w = len(s.split())
                if wc + w > 160: break
                keep.append(s)
                wc += w
            body = " ".join(keep).rstrip(",;:") + ("" if body.endswith(('.', '!', '?')) else ".")

            # rebuild full text with questions
            rest = [ln for ln in lines if re.match(r'(?i)^q[1-9]:', ln)]
            lines = lines[:li+1] + [body] + rest

        text = "\n".join(lines)
    except ValueError:
        pass
    return text


def clean_pairs(pairs_raw):
    out = []
    for p in pairs_raw:
        if not isinstance(p, str):
            continue
        # reject proverbs/quotes/JSON-like
        if any(ch in p for ch in ['{', '}', '"']):
            continue
        if "Old saying" in p or "’" in p or "“" in p or "”" in p:
            continue
        p = re.sub(r"\s+", " ", p).strip()
        p = re.sub(r"[→⇒=>]", "->", p)
        # keep only strings that look like "X -> Y"
        if "->" in p:
            left, right = p.split("->", 1)
            if left.strip() and right.strip() and len(p) <= 120:
                out.append(f"{left.strip()} -> {right.strip()}")
    # cap 5 items
    return out[:5]

def generate(
    objective: str,
    topic: str,
    top_k: int = 2,
    model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    ttl_path: str = "rdf/corpus.ttl",
    difficulty: int | None = None,
    diff_band: int = 1,
    min_words: int = 60,
    limit: int = 50,
):
    # 1) Retrieve by topic
    items = query_passages(
    topic=topic,
    ttl_path=ttl_path,
    difficulty=difficulty,
    diff_band=diff_band,
    min_words=min_words,
    limit=limit,
    )
    if not items:
        print(f"No passages found for topic '{topic}' in {ttl_path}.")
        return

    # 2) Re-rank by objective similarity
    print(">>> Retrieving and re-ranking passages...")
    passages = [x["text"] for x in items]
    ranked = rerank_by_objective(objective, passages, top_k=top_k)
    chosen = [items[i] | {"similarity": score} for i, score in ranked]

    # 3) PLAN: extract cause→effect pairs from chosen context
    plan_msgs = [
        {"role": "system", "content": "You are a precise teaching assistant. Output ONLY valid JSON exactly as specified."},
        {"role": "user", "content": (
            "Learning objective: Teach cause and effect in short stories.\n"
            "From ONLY the context passages below, extract 3–5 concise cause→effect pairs that literally occur or are strongly implied in the text. "
            "Do NOT invent proverbs, quotes, place names, or external references. Do NOT output JSON objects inside the list; each item must be a simple string in the form 'cause -> effect'.\n"
            "Return JSON exactly in this schema:\n"
            '{\"pairs\": [\"<cause> -> <effect>\", \"<cause> -> <effect>\"]}\n\n'
            "Context:\n" + build_context_snippets(chosen)
        )}
    ]

    # 3) PLAN: extract cause→effect pairs from chosen context
    print(">>> Mining cause→effect pairs heuristically from retrieved passages...")
    heur_pairs = extract_pairs_from_texts([x["text"] for x in chosen], max_pairs=5)
    print(">>> Heuristic pairs:", heur_pairs)

    pairs = list(heur_pairs)

    # If we didn't get at least 3, back off to a small LLM plan step to fill gaps
    if len(pairs) < 3:
        plan_msgs = [
            {"role": "system", "content": "You are a precise teaching assistant. Output ONLY valid JSON exactly as specified."},
            {"role": "user", "content": (
                "From ONLY the context passages below, extract up to 5 concise cause→effect pairs that literally occur or are strongly implied in the text. "
                "Use the exact format 'cause -> effect' for each item. Do NOT invent external references.\n"
                'Return JSON exactly as: {"pairs": ["<cause> -> <effect>", "..."]}\n\n'
                "Context:\n" + build_context_snippets(chosen)
            )}
        ]
        print(">>> Heuristic <3; backing off to model planning...")
        plan_raw = hf_chat_generate(plan_msgs, model_id=model_id, max_new_tokens=180, temperature=0.15)
        m = re.search(r'\{.*\}', plan_raw, flags=re.S)
        if m:
            try:
                llm_pairs = json.loads(m.group(0)).get("pairs", [])
            except Exception:
                llm_pairs = []
            llm_pairs = clean_pairs(llm_pairs)
            # merge & dedupe while preserving order
            for p in llm_pairs:
                if p.lower() not in [q.lower() for q in pairs]:
                    pairs.append(p)

    # LAST RESORT: make *content-aware* fallback, not generic rain
    if not pairs:
        # naive: use first two sentences to form a pair
        for t in [x["text"] for x in chosen]:
            sents = _sentences(t)
            if len(sents) >= 2:
                guess = _mk_pair(sents[0][:120], sents[1][:120])
                if guess:
                    pairs.append(guess)
            if len(pairs) >= 3:
                break

    print(">>> Pairs:", pairs)


    # 4) WRITE: produce lesson + 3 questions in strict format
    write_msgs = [
        {"role": "system", "content": "You are a fair and explainable tutoring assistant."},
        {"role": "user", "content": (
            "Write a short lesson (120–160 words) that teaches CAUSE and EFFECT using ONLY the notes below. "
            "Do not invent place names, quotes, or external stories. Use neutral phrasing like 'a town', 'a river'. "
            "Use inclusive, stereotype‑free language. Then write exactly 3 comprehension questions about the cause→effect relationships.\n\n"
            f"Learning objective: {objective}\n"
            "Cause→Effect notes:\n- " + "\n- ".join(pairs) + "\n\n"
            "STRICT OUTPUT FORMAT (follow exactly):\n"
            "Lesson:\n"
            "<ONE paragraph, 120–160 words. NO headings, NO bullet points, NO numbered sections.>\n"
            "Q1:\nQ2:\nQ3:\n"
        )}
    ]

    print(">>> Writing lesson...")
    lesson = hf_chat_generate(write_msgs, model_id=model_id, max_new_tokens=260, temperature=0.25)
    print(">>> Write step complete.")

    lesson = postprocess_lesson(lesson)

    # 5) Fairness scan (toy)
    fairness = simple_fairness_scan(lesson)

    # 6) Print result + explanations
    print("=== LESSON ===")
    print(lesson.strip(), "\n")

    print("=== EXPLANATION ===")
    for x in chosen:
        print(explain_selection(topic, x["difficulty"], x["similarity"]))
        print(f"URI: {x['uri']} | Title: {x['title']}")
    print("\nFairness check:", "passed" if fairness["passed"] else f"flagged {fairness['flagged_terms']}")
    print()
    return lesson, chosen, fairness

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, help="e.g., 'Teach cause and effect in short stories'")
    p.add_argument("--topic", required=True, help="e.g., 'Nature'")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--model_id", default="microsoft/Phi-3-mini-4k-instruct", help="Hugging Face model ID")
    p.add_argument("--ttl", default="rdf/corpus.ttl", help="Path to RDF corpus (.ttl)")
    p.add_argument("--difficulty", type=int, default=None, help="Target difficulty (optional)")
    p.add_argument("--diff_band", type=int, default=1, help="+/- band around difficulty")
    p.add_argument("--min_words", type=int, default=60, help="Minimum words per passage")
    p.add_argument("--limit", type=int, default=50, help="Max items to fetch before reranking")

    args = p.parse_args()

    generate(
    args.objective,
    args.topic,
    top_k=args.top_k,
    model_id=args.model_id,
    ttl_path=args.ttl,
    difficulty=args.difficulty,
    diff_band=args.diff_band,
    min_words=args.min_words,
    limit=args.limit,
    )
