# src/generation/planning.py
import re, json
from typing import List

import os
from openai import OpenAI
from .utils import chat_generate
import re

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# planning.py

# planning.py
import re

def _normalize_clause(s: str) -> str:
    # strip quotes/punctuation, collapse spaces
    s = re.sub(r'[“”"\'`]+', '', s)
    s = s.strip(" .,:;–—-")
    # drop leading discourse markers
    s = re.sub(r'^(and|but|so|then|thus|therefore|however)\b[:,\s-]*', '', s, flags=re.I)
    # keep the main clause before heavy qualifiers
    s = s.split(",", 1)[0]
    s = re.split(r'\b(which|that|who)\b', s, maxsplit=1, flags=re.I)[0]
    s = re.sub(r'\s+', ' ', s).strip()
    # sentence case
    if s:
        s = s[0].upper() + s[1:]
    return s

def refine_pair_text(p: str) -> str:
    if "->" not in p:
        return p.strip()
    L, R = [x.strip() for x in p.split("->", 1)]
    L, R = _normalize_clause(L), _normalize_clause(R)
    return f"{L} -> {R}"

def is_good_pair(p: str) -> bool:
    if "->" not in p:
        return False
    L, R = [x.strip(" '\"“”‘’.,;:-") for x in p.split("->", 1)]

    # reject super-generic starts
    bad_starts = ("there is nothing", "there’s nothing", "it is fascinating", "fascinating,")
    if L.lower().startswith(bad_starts) or R.lower().startswith(bad_starts):
        return False

    # reasonable lengths
    if not (3 <= len(L.split()) <= 16 and 3 <= len(R.split()) <= 16):
        return False

    # cues are a plus, not mandatory
    return True

def _normalize_text(t: str) -> str:
    # normalize punctuation that can break regexes
    t = t.replace("—", " - ").replace("–", " - ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def extract_pairs_from_texts(texts: List[str], max_pairs=5) -> List[str]:
    pairs, seen = [], set()

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip(" ;,.-–—")

    def mk(c: str, e: str):
        c, e = norm(c), norm(e)
        if not c or not e:
            return None
        p = f"{c} -> {e}"
        if 8 <= len(p) <= 160 and p.lower() not in seen:
            seen.add(p.lower()); 
            return p
        return None

    cue_patterns = [
        # because / due to / owing to (both orders)
        (r"\bbecause\b\s+(.*)", "rev"),         # X because Y  => Y -> X
        (r"\bdue to\b\s+(.*)", "rev"),
        (r"\bowing to\b\s+(.*)", "rev"),
        (r"(.*?),\s+so that\s+(.*)", "fwd"),    # X, so that Y => X -> Y
        (r"(.*?),\s+so\s+(.*)", "fwd"),
        # led to / resulted in / causes / caused / causing / forcing (forward)
        (r"(.*)\bled to\b\s+(.*)", "fwd"),
        (r"(.*)\bresulted in\b\s+(.*)", "fwd"),
        (r"(.*)\bcauses\b\s+(.*)", "fwd"),
        (r"(.*)\bcaused\b\s+(.*)", "fwd"),
        (r"(.*)\bcausing\b\s+(.*)", "fwd"),
        (r"(.*)\bforcing\b\s+(.*)", "fwd"),
        # therefore / thus / consequently / hence
        (r"(.*?)(?:;|,)?\s*\b(therefore|thus|consequently|hence)\b[, ]+(.*)", "fwd"),
    ]

    for raw in texts:
        txt = _normalize_text(raw)
        # simple sentence-ish split (keep long sentences intact)
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # try cues (case-insensitive, search anywhere)
            matched = False
            for pat, direction in cue_patterns:
                m = re.search(pat, s, flags=re.IGNORECASE)
                if not m:
                    continue
                if direction == "rev" and m.lastindex:
                    # X because Y  ->  Y -> X
                    # we need both sides; try to split remaining part
                    # find the left side as "everything before the cue"
                    pre = re.split(pat, s, flags=re.IGNORECASE)[0].strip()
                    cause, effect = m.group(1), pre
                elif direction == "fwd" and m.lastindex and m.lastindex >= 2:
                    cause, effect = m.group(1), m.group(2)
                else:
                    # fallback for single-group patterns: split by cue word
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

        # content-aware last-ditch fallback: split on obvious causal tokens
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

def tidy_pair(p: str) -> str:
    p = re.sub(r'\s*-\s*->', ' -> ', p)
    p = re.sub(r'\blifeboot\b', 'lifeboat', p, flags=re.I)
    p = re.sub(r'\blifeboats\b', 'lifeboat', p, flags=re.I)
    if "->" in p:
        l, r = [x.strip() for x in p.split("->", 1)]
        p = f"{l[:1].upper()+l[1:]} -> {r[:1].upper()+r[1:]}"
    return p.strip()

def clean_pairs(pairs: list[str]) -> list[str]:
    out, seen = [], set()
    for p in pairs:
        if not isinstance(p, str):
            continue
        # normalize arrows and strip quotes/braces
        p = re.sub(r"[→⇒=>]", "->", p)
        p = p.replace("{", "").replace("}", "")
        p = re.sub(r'[“”"]', "", p)

        p = p.strip()
        if "->" not in p:
            continue
        L, R = [x.strip() for x in p.split("->", 1)]
        if not L or not R:
            continue

        # normalize spacing/casing a touch
        L = re.sub(r"\s+", " ", L)
        R = re.sub(r"\s+", " ", R)
        p = f"{L[:1].upper()+L[1:]} -> {R[:1].upper()+R[1:]}"
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out[:5]


# def plan_with_llm(chosen_items, model_id: str, backend) -> List[str]:
#     ctx = "\n\n".join([f"Title: {x['title']}\nText:\n{x['text']}" for x in chosen_items])

#     base_prompt = (
#         "From ONLY the context passages below, extract 3–5 concise cause→effect pairs in the form 'cause -> effect'.\n"
#         "STRICT RULES:\n"
#         "- Only use events literally described or strongly implied in the text.\n"
#         "- DO NOT invent author names, book titles, teaching materials, or proverbs.\n"
#         "- DO NOT include generic moral lessons or global commentary.\n"
#         "- Output must be valid JSON exactly as: {\"pairs\": [\"<cause> -> <effect>\"]}\n\n"
#         f"Context:\n{ctx}"
#     )

#     pairs = []
#     for attempt in range(3):  # retry loop
#         msgs = [
#             {"role": "system", "content": "You are a precise teaching assistant. Output ONLY valid JSON."},
#             {"role": "user", "content": base_prompt}
#         ]
#         raw = chat_generate(msgs, backend=backend, model_id=model_id, temperature=0.1, max_new_tokens=220)
#         m = re.search(r'\{.*\}', raw, flags=re.S)
#         if m:
#             try:
#                 parsed_pairs = json.loads(m.group(0)).get("pairs", [])
#                 pairs = clean_pairs(parsed_pairs)
#             except Exception:
#                 pairs = []
#         if pairs:  # break early if we got something valid
#             break

#     # Final fallback to heuristic if still empty
#     if not pairs:
#         print(">>> LLM failed to return valid pairs; falling back to heuristic extraction.")
#         text_blocks = [x["text"] for x in chosen_items]
#         pairs = extract_pairs_from_texts(text_blocks, max_pairs=3)
#         pairs = clean_pairs(pairs)

#     return pairs

def plan_with_llm(chosen_items, model_id: str, backend) -> list[str]:
    ctx = "\n\n".join([f"Title: {x['title']}\nText:\n{x['text']}" for x in chosen_items])
    msgs = [
        {"role": "system", "content": "You are a precise teaching assistant."},
        {"role": "user", "content": (
            "From ONLY the context below, extract up to 5 concise cause→effect pairs in the form 'cause -> effect'. "
            "Prefer short paraphrases over quotes. Do not invent external details.\n\n"
            "Return EITHER:\n"
            '1) JSON as {"pairs": ["<cause> -> <effect>", ...]}\n'
            "OR\n"
            "2) Plain lines, one per pair, each containing '->'.\n\n"
            f"{ctx}"
        )}
    ]
    raw = chat_generate(msgs, backend=backend, model_id=model_id, temperature=0.1, max_new_tokens=220)
    print(">>> [plan raw sample]:", raw[:220].replace("\n"," ") + ("..." if len(raw) > 220 else ""))

    # try JSON first
    pairs = []
    m = re.search(r'\{.*\}', raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                pairs = obj.get("pairs", []) or obj.get("notes", []) or []
        except Exception:
            pairs = []

    # fallback: grab lines with '->'
    if not pairs:
        lines = [ln.strip() for ln in raw.splitlines()]
        pairs = [ln for ln in lines if "->" in ln and len(ln.split("->", 1)[0].strip()) > 0]

    return clean_pairs(pairs)


