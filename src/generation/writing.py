# src/generation/writing.py
import re
from typing import List
from .utils import chat_generate

from src.generation.utils import chat_generate
from src.generation.fairness import simple_fairness_scan

from src.generation.utils import chat_generate
from src.generation.fairness import simple_fairness_scan

def pairs_to_questions(pairs):
    qs, used = [], set()
    for i, p in enumerate(pairs[:3], start=1):
        if "->" in p:
            cause, _ = [t.strip() for t in p.split("->", 1)]
            # trim trailing punctuation/hyphens to avoid “...paper -”
            cause = re.sub(r'\s*[-–—:]*\s*$', '', cause)
            q = f"Q{i}: What effect follows from: {cause}?"
        else:
            q = f"Q{i}: Identify one cause and its effect from the lesson."
        if q.lower() not in used:
            used.add(q.lower()); qs.append(q)
    while len(qs) < 3:
        filler = f"Q{len(qs)+1}: Identify one cause and its effect from the lesson."
        if filler.lower() not in used:
            used.add(filler.lower()); qs.append(filler)
    return qs[:3]

def write_lesson(objective, pairs, model_id, backend):
    """
    Writes a lesson and 3 comprehension questions using the given pairs.
    Ensures the output is one paragraph (120–160 words) with no headings, bullets, or numbered steps.
    """
    msgs = [
        {"role": "system", "content": "You are a fair and explainable tutoring assistant."},
        {"role": "user", "content": (
            f"Write exactly ONE paragraph (120–160 words) that teaches {objective} "
            "using ONLY the cause→effect notes below. "
            "Avoid headings, bullet points, numbered lists, or section labels. "
            "Do not invent place names, quotes, or external stories. Use neutral, inclusive, stereotype-free language. "
            "Then write exactly 3 unique comprehension questions about the cause→effect relationships, each starting with 'Q1:', 'Q2:', 'Q3:'.\n\n"
            f"Cause→Effect notes:\n- " + "\n- ".join(pairs) + "\n\n"
            "STRICT OUTPUT FORMAT (follow exactly):\n"
            "Lesson:\n"
            "<ONE paragraph only, 120–160 words. No headings, bullets, or numbered steps.>\n"
            "Q1:\nQ2:\nQ3:\n"
        )}
    ]

    raw_lesson = chat_generate(
        msgs,
        backend=backend,
        model_id=model_id,
        temperature=0.25,
        max_new_tokens=260
    )

    cleaned_lesson = postprocess_lesson(raw_lesson)

    # overwrite Q1–Q3 with pair-derived questions
    qs = pairs_to_questions(pairs)
    lines = [ln for ln in cleaned_lesson.splitlines() if ln.strip()]
    try:
        li = lines.index("Lesson:")
    except ValueError:
        li = 0

    body = []
    for j in range(li+1, len(lines)):
        if re.match(r'(?i)^q[1-9]:', lines[j]):
            break
        body.append(lines[j])

    final_lines = lines[:li+1] + [" ".join(body).strip()] + qs
    cleaned_lesson = "\n".join(final_lines)

    fairness = simple_fairness_scan(cleaned_lesson)

    return cleaned_lesson, fairness

def build_write_prompt_with_context(objective: str, passage_text: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a fair and explainable tutoring assistant."},
        {"role": "user", "content": (
            "Write a short lesson (120–160 words) that teaches CAUSE and EFFECT using ONLY the excerpt below. "
            "Keep the lesson tightly focused on this excerpt; do not introduce unrelated topics or global commentary. "
            "Use inclusive, stereotype‑free language. Then write exactly 3 unique comprehension questions.\n\n"
            f"Learning objective: {objective}\n\n"
            "Excerpt:\n"
            f"\"\"\"\n{passage_text}\n\"\"\"\n\n"
            "STRICT OUTPUT FORMAT:\n"
            "Lesson:\n"
            "<ONE paragraph, 120–160 words. NO headings, NO bullets.>\n"
            "Q1:\nQ2:\nQ3:\n"
        )}
    ]

def postprocess_lesson(lesson_text: str, default_q: str = "Identify one cause and its effect from the lesson.") -> str:
    text = lesson_text.strip()

    if "Lesson:" in text:
        text = text[text.index("Lesson:"):]
    text = re.sub(r'(?im)^\s*(lecture|objective|duration|materials)\s*:.*\n', '', text)

    # Remove duplicate sentences
    sents = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in sents:
        if s and (not cleaned or s.lower() != cleaned[-1].lower()):
            cleaned.append(s)
    text = " ".join(cleaned)

    lines = [ln for ln in text.splitlines() if ln.strip()]

    # ✅ NEW: handle "Lesson: <inline text>" so the clamp finds a standalone header
    if lines and lines[0].lower().startswith("lesson:"):
        first = lines[0]
        after = first.split(":", 1)[1].strip() if ":" in first else ""
        lines = ["Lesson:"] + ([after] if after else []) + lines[1:]

    # If lesson starts directly with questions
    if lines and lines[0].lower().startswith("q1:"):
        qs = [ln for ln in lines if re.match(r'(?i)^q[1-9]:', ln)]
        body = "\n".join([ln for ln in lines if ln not in qs])
        text = f"{body}\n" + "\n".join(qs)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    qs_idx = [i for i, l in enumerate(lines) if re.match(r'(?i)^q[1-9]:', l)]

    # Fill missing questions
    if len(qs_idx) < 3:
        need = 3 - len(qs_idx)
        start = len(qs_idx) + 1
        lines += [f"Q{start + i}: {default_q}" for i in range(need)]
    elif len(qs_idx) > 3:
        kept, n = [], 0
        for ln in lines:
            if re.match(r'(?i)^q[1-9]:', ln):
                if n < 3:
                    kept.append(ln); n += 1
            else:
                kept.append(ln)
        lines = kept

    # Clamp to ~160 words
    try:
        li = lines.index("Lesson:")
        body, after = [], []
        for j in range(li + 1, len(lines)):
            if re.match(r'(?i)^q[1-9]:', lines[j]):
                after = lines[j:]; break
            body.append(lines[j])
        body = " ".join(body).strip()
        words = body.split()
        if len(words) > 175:
            sents = re.split(r'(?<=[.!?])\s+', body)
            keep, wc = [], 0
            for s in sents:
                w = len(s.split())
                if wc + w > 160: break
                keep.append(s); wc += w
            body = " ".join(keep).rstrip(",;:") + "."
        lines = lines[:li + 1] + [body] + (after if after else [l for l in lines if re.match(r'(?i)^q[1-9]:', l)])
    except ValueError:
        pass

    # Deduplicate repeated questions
    seen, deduped = set(), []
    for ln in lines:
        if re.match(r'(?i)^q[1-9]:', ln):
            norm = ln.lower().strip()
            if norm not in seen:
                seen.add(norm); deduped.append(ln)
        else:
            deduped.append(ln)
    lines = deduped

    return "\n".join(lines)

def strip_answers(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        if re.match(r'(?i)^q[1-3]:', ln):
            ln = re.sub(r'(?i)\s*[-–—]?\s*answer.*$', '', ln).rstrip()
        cleaned.append(ln)
    return "\n".join(cleaned)
