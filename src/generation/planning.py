# src/generation/planning.py
"""Planning utilities for mining and preparing cause→effect pairs.

This module provides lightweight heuristics to *extract*, *tidy*, and *filter*
"cause -> effect" pairs from context passages, plus a small LLM-backed
fallback (`plan_with_llm`) when the heuristics yield too few good pairs.

Key functions
-------------
- extract_pairs_from_texts: *scanning* texts and *extracting* candidate pairs.
- clean_pairs: *normalizing* arrow format, *deduplicating*, and *casing*.
- is_good_pair / refine_pair_text / tidy_pair: *validating* and *polishing* pairs.
- plan_with_llm: *asking* a chat model to propose pairs from provided contexts.
"""

from __future__ import annotations

import json
import re
from typing import List

from .utils import chat_generate


# ------------------------------
# Clause and text normalization
# ------------------------------

def _normalize_clause(s: str) -> str:
    """Return a compact, sentence-cased clause.

    Steps: *removing* quotes/punctuation, *dropping* discourse markers,
    *keeping* text before heavy qualifiers, *collapsing* whitespace, and
    *applying* sentence case.
    """
    # removing quotes/backticks
    s = re.sub(r'[“”"\'`]+', '', s)
    # trimming leading/trailing punctuation
    s = s.strip(" .,:;–—-")
    # dropping leading discourse markers (and/but/so/then/...)
    s = re.sub(r'^(and|but|so|then|thus|therefore|however)\b[:,\s-]*', '', s, flags=re.I)
    # keeping main clause before heavy qualifiers
    s = s.split(",", 1)[0]
    s = re.split(r'\b(which|that|who)\b', s, maxsplit=1, flags=re.I)[0]
    # collapsing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # applying sentence case
    if s:
        s = s[0].upper() + s[1:]
    return s


def _normalize_text(t: str) -> str:
    """Return a whitespace-normalized string suitable for regex scanning."""
    t = t.replace("—", " - ").replace("–", " - ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ------------------------------
# Pair shaping helpers
# ------------------------------

def refine_pair_text(p: str) -> str:
    """Return a refined pair with normalized clauses around an arrow.

    If no arrow is present, *returning* a stripped version of the input.
    """
    if "->" not in p:
        return p.strip()
    left, right = [x.strip() for x in p.split("->", 1)]
    left, right = _normalize_clause(left), _normalize_clause(right)
    return f"{left} -> {right}"


def tidy_pair(p: str) -> str:
    """Return a tidied pair with consistent spacing, spelling, and casing."""
    # normalizing arrow spacing
    p = re.sub(r'\s*-\s*->', ' -> ', p)
    # fixing common OCR/typo variants
    p = re.sub(r'\blifeboot\b', 'lifeboat', p, flags=re.I)
    p = re.sub(r'\blifeboats\b', 'lifeboat', p, flags=re.I)
    if "->" in p:
        l, r = [x.strip() for x in p.split("->", 1)]
        p = f"{l[:1].upper()+l[1:]} -> {r[:1].upper()+r[1:]}"
    return p.strip()


def is_good_pair(p: str) -> bool:
    """Return True if *p* looks like a concise, non-generic "cause -> effect" pair."""
    if "->" not in p:
        return False
    left, right = [x.strip(" '\"“”‘’.,;:-") for x in p.split("->", 1)]

    # rejecting super-generic starts
    bad_starts = ("there is nothing", "there’s nothing", "it is fascinating", "fascinating,")
    if left.lower().startswith(bad_starts) or right.lower().startswith(bad_starts):
        return False

    # enforcing reasonable word lengths
    if not (3 <= len(left.split()) <= 16 and 3 <= len(right.split()) <= 16):
        return False

    return True


def clean_pairs(pairs: List[str]) -> List[str]:
    """Return a cleaned, deduplicated list of pairs (max 5 preserved).

    Steps: *converting* alternate arrows to `->`, *stripping* quotes/braces,
    *trimming* sides, *collapsing* whitespace, *casing* first letters,
    and *deduplicating* case-insensitively.
    """
    out, seen = [], set()
    for p in pairs:
        if not isinstance(p, str):
            continue
        # converting arrow symbols
        p = re.sub(r"[→⇒=>]", "->", p)
        # removing braces/quotes
        p = p.replace("{", "").replace("}", "")
        p = re.sub(r'[“”"]', "", p)
        p = p.strip()
        if "->" not in p:
            continue
        L, R = [x.strip() for x in p.split("->", 1)]
        if not L or not R:
            continue
        # collapsing spaces and casing
        L = re.sub(r"\s+", " ", L)
        R = re.sub(r"\s+", " ", R)
        p = f"{L[:1].upper()+L[1:]} -> {R[:1].upper()+R[1:]}"
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out[:5]


# ------------------------------
# Extraction from raw texts
# ------------------------------

def extract_pairs_from_texts(texts: List[str], max_pairs: int = 5) -> List[str]:
    """Extract up to *max_pairs* cause→effect pairs from a list of texts.

    Strategy: *splitting* texts into sentence-like chunks, *searching* for cue
    patterns ("because", "led to", etc.), *building* pairs with directionality,
    and *collecting* unique candidates.
    """
    pairs, seen = [], set()

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip(" ;,.-–—")

    def mk(c: str, e: str):
        c, e = norm(c), norm(e)
        if not c or not e:
            return None
        p = f"{c} -> {e}"
        if 8 <= len(p) <= 160 and p.lower() not in seen:
            seen.add(p.lower())
            return p
        return None

    # cue patterns: list of (regex, direction)
    cue_patterns = [
        # because / due to / owing to (reverse direction: cause appears after cue)
        (r"\bbecause\b\s+(.*)", "rev"),      # X because Y  => Y -> X
        (r"\bdue to\b\s+(.*)", "rev"),
        (r"\bowing to\b\s+(.*)", "rev"),
        # so/so that (forward direction)
        (r"(.*?),\s+so that\s+(.*)", "fwd"),  # X, so that Y => X -> Y
        (r"(.*?),\s+so\s+(.*)", "fwd"),
        # led to / resulted in / causes / caused / causing / forcing (forward)
        (r"(.*)\bled to\b\s+(.*)", "fwd"),
        (r"(.*)\bresulted in\b\s+(.*)", "fwd"),
        (r"(.*)\bcauses\b\s+(.*)", "fwd"),
        (r"(.*)\bcaused\b\s+(.*)", "fwd"),
        (r"(.*)\bcausing\b\s+(.*)", "fwd"),
        (r"(.*)\bforcing\b\s+(.*)", "fwd"),
        # therefore / thus / consequently / hence (forward)
        (r"(.*?)(?:;|,)?\s*\b(therefore|thus|consequently|hence)\b[, ]+(.*)", "fwd"),
    ]

    for raw in texts:
        txt = _normalize_text(raw)
        # splitting into sentence-like chunks
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # searching cue patterns (case-insensitive)
            matched = False
            for pat, direction in cue_patterns:
                m = re.search(pat, s, flags=re.IGNORECASE)
                if not m:
                    continue
                if direction == "rev" and m.lastindex:
                    # X because Y  ->  Y -> X
                    pre = re.split(pat, s, flags=re.IGNORECASE)[0].strip()
                    cause, effect = m.group(1), pre
                elif direction == "fwd" and m.lastindex and m.lastindex >= 2:
                    cause, effect = m.group(1), m.group(2)
                else:
                    # fallback: split once by pattern
                    parts = re.split(pat, s, flags=re.IGNORECASE, maxsplit=1)
                    if len(parts) == 2:
                        cause, effect = parts[0], parts[1]
                    else:
                        continue
                p = mk(cause, effect)
                if p:
                    pairs.append(p)
                matched = True
                break
            if matched and len(pairs) >= max_pairs:
                break

        if len(pairs) >= max_pairs:
            break

        # fallback: split on obvious tokens when nothing matched
        if len(pairs) == 0:
            for token in ["causing", "led to", "resulted in"]:
                if token in txt.lower():
                    parts = re.split(token, txt, flags=re.IGNORECASE, maxsplit=1)
                    if len(parts) == 2:
                        p = mk(parts[0], parts[1])
                        if p:
                            pairs.append(p)
                            break

    return pairs[:max_pairs]


# ------------------------------
# LLM-backed planning fallback
# ------------------------------

def plan_with_llm(chosen_items: List[dict], model_id: str, backend: str) -> List[str]:
    """Ask a chat model to propose up to five cause→effect pairs from contexts.

    The function *building* a prompt from `chosen_items`, *calling* `chat_generate`,
    and then *parsing* either JSON or plain-text output lines containing arrows.
    """
    ctx = "\n\n".join([f"Title: {x['title']}\nText:\n{x['text']}" for x in chosen_items])
    msgs = [
        {"role": "system", "content": "You are a precise teaching assistant."},
        {
            "role": "user",
            "content": (
                "From ONLY the context below, extract up to 5 concise cause→effect pairs in the form 'cause -> effect'. "
                "Prefer short paraphrases over quotes. Do not invent external details.\n\n"
                "Return EITHER:\n"
                '1) JSON as {"pairs": ["<cause> -> <effect>", ...]}\n'
                "OR\n"
                "2) Plain lines, one per pair, each containing '->'.\n\n"
                f"{ctx}"
            ),
        },
    ]

    raw = chat_generate(
        msgs, backend=backend, model_id=model_id, temperature=0.1, max_new_tokens=220
    )

    # printing a short sample for debugging CLI runs
    print(
        ">>> [plan raw sample]:",
        (raw[:220].replace("\n", " ") + ("..." if len(raw) > 220 else "")) if isinstance(raw, str) else raw,
    )

    # trying to parse JSON first
    pairs: List[str] = []
    m = re.search(r"\{.*\}", raw or "", flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                pairs = obj.get("pairs", []) or obj.get("notes", []) or []
        except Exception:
            pairs = []

    # falling back to plain lines containing '->'
    if not pairs and isinstance(raw, str):
        lines = [ln.strip() for ln in raw.splitlines()]
        pairs = [ln for ln in lines if "->" in ln and len(ln.split("->", 1)[0].strip()) > 0]

    return clean_pairs(pairs)
