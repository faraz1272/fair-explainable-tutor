# src/generation/writing.py
"""Lesson writing utilities.

This module provides helpers to:
- generate a lesson and questions from cause→effect *pairs* (`write_lesson`),
- build a writing prompt for a single *context passage* (`build_write_prompt_with_context`),
- *postprocess* raw model output into our strict "Lesson + Q1..Q3" format,
- *strip* any inline answers from question lines,
- *derive* diverse questions from input pairs.

Notes:
- Fairness checks are handled upstream in `generate.py` via `fairness_ext`.
- We keep action-verb comments to clarify each transformation step.
"""

from __future__ import annotations

import re
from typing import Dict, List

from .utils import chat_generate


def pairs_to_questions(pairs: List[str]) -> List[str]:
    """Return up to three question prompts derived from cause→effect *pairs*.

    We *extract* the cause (left side of "->"), *trim* trailing punctuation,
    and *build* distinct question stems. If a pair is malformed, we *fallback*
    to a generic prompt.
    """
    qs: List[str] = []
    used: set[str] = set()
    for i, p in enumerate(pairs[:3], start=1):
        if "->" in p:
            cause, _ = [t.strip() for t in p.split("->", 1)]
            # trimming trailing punctuation/hyphens to avoid “… paper -”
            cause = re.sub(r"\s*[-–—:]*\s*$", "", cause)
            q = f"Q{i}: What effect follows from: {cause}?"
        else:
            q = f"Q{i}: Identify one cause and its effect from the lesson."
        if q.lower() not in used:
            used.add(q.lower())
            qs.append(q)

    # backfilling to ensure exactly three
    while len(qs) < 3:
        filler = f"Q{len(qs) + 1}: Identify one cause and its effect from the lesson."
        if filler.lower() not in used:
            used.add(filler.lower())
            qs.append(filler)
    return qs[:3]


def write_lesson(
    objective: str,
    pairs: List[str],
    model_id: str,
    backend: str,
) -> tuple[str, None]:
    """Generate a one-paragraph lesson + 3 questions from *pairs*.

    Steps:
    1) *building* a strict-format prompt for the model,
    2) *generating* candidate text via `chat_generate`,
    3) *postprocessing* it to our required format,
    4) *replacing* the questions with pair-derived ones (deduped and numbered).

    Returns:
        (cleaned_lesson, None). Fairness is handled by the caller later.
    """
    msgs = [
        {"role": "system", "content": "You are a fair and explainable tutoring assistant."},
        {
            "role": "user",
            "content": (
                "Write a single paragraph (120–160 words) that teaches CAUSE and EFFECT using ONLY the cause→effect notes below. "
                "Keep the lesson tied to the implied scene; do not invent classroom activities, lists, or headings. "
                "Use inclusive, stereotype-free language. "
                "Then write exactly 3 comprehension questions about the cause→effect relationships. "
                "Each question must be phrased differently (no repetition) and must start with 'Q1:', 'Q2:', 'Q3:'.\n\n"
                f"Learning objective: {objective}\n"
                "Cause→Effect notes:\n- "
                + "\n- ".join(pairs)
                + "\n\nSTRICT OUTPUT FORMAT (follow exactly):\nLesson:\n<ONE paragraph only, 120–160 words. No headings, bullets, or numbered steps.>\nQ1:\nQ2:\nQ3:\n"
            ),
        },
    ]

    # generating model output
    raw_lesson = chat_generate(
        msgs, backend=backend, model_id=model_id, temperature=0.25, max_new_tokens=260
    )

    # normalizing the model output and ensuring a sensible fallback question text
    cleaned_lesson = postprocess_lesson(
        raw_lesson, default_q=f"Identify one example related to: {objective}"
    )

    # --- Overwrite questions using pair-derived prompts, then enforce uniqueness ---
    qs = pairs_to_questions(pairs)

    # *normalizing* keys for dedupe
    def norm_key(q: str) -> str:
        q = re.sub(r"(?i)^q[1-9]:\s*", "", q).strip()
        q = re.sub(r"\s+", " ", q.lower())
        return q.rstrip("?.")

    # *stripping* any inline answers and *ensuring* a trailing question mark
    base_qs: List[str] = []
    for q in qs:
        q = re.sub(r"(?i)\s*[-–—]?\s*answer.*$", "", q).strip()
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        base_qs.append(q)

    # *deduping* while preserving order
    seen: set[str] = set()
    uniq_qs: List[str] = []
    for q in base_qs:
        k = norm_key(q)
        if k not in seen:
            seen.add(k)
            uniq_qs.append(q)

    # *backfilling* with diverse templates
    templates = [
        "Identify one cause and its effect from the lesson.",
        "Explain how one action led to a result in the lesson.",
        "Find one example of a cause that produced an effect, and name both parts.",
        "Describe a cause–effect link you noticed in the lesson.",
    ]
    for t in templates:
        if len(uniq_qs) >= 3:
            break
        k = norm_key(t)
        if k not in seen:
            uniq_qs.append(t if t.endswith("?") else t + "?")
            seen.add(k)

    # *padding* if still short (extreme edge)
    while len(uniq_qs) < 3:
        uniq_qs.append(
            (templates[-1] if templates else f"Identify one example related to: {objective}") + "?"
        )

    # *renumbering* as Q1/Q2/Q3
    qprefix_re = re.compile(r"(?i)^q[1-9]:\s*")
    final_qs: List[str] = []
    for i in range(3):
        qtxt = qprefix_re.sub("", uniq_qs[i]).strip()
        final_qs.append(f"Q{i + 1}: {qtxt}")

    # *rebuilding* output: keep the lesson body, replace Qs with our final_qs
    lines = [ln for ln in cleaned_lesson.splitlines() if ln.strip()]
    try:
        li = lines.index("Lesson:")
    except ValueError:
        lines = ["Lesson:"] + lines
        li = 0

    body: List[str] = []
    for j in range(li + 1, len(lines)):
        if re.match(r"(?i)^q[1-9]:", lines[j]):
            break
        body.append(lines[j])

    final_lines = lines[: li + 1] + [" ".join(body).strip()] + final_qs
    cleaned_lesson = "\n".join(final_lines)

    # returning lesson; fairness is handled later in generate() via fairness_ext
    return cleaned_lesson, None


def build_write_prompt_with_context(objective: str, passage_text: str) -> List[Dict[str, str]]:
    """Build a strict-format prompt for writing from a single *passage* context."""
    return [
        {"role": "system", "content": "You are a fair and explainable tutoring assistant."},
        {
            "role": "user",
            "content": (
                "Write a short lesson (120–160 words) that teaches CAUSE and EFFECT using ONLY the excerpt below. "
                "Keep the lesson tightly focused on this excerpt; do not introduce unrelated topics or global commentary. "
                "Use inclusive, stereotype‑free language. Then write exactly 3 unique comprehension questions.\n\n"
                f"Learning objective: {objective}\n\nExcerpt:\n\"\"\"\n{passage_text}\n\"\"\"\n\n"
                "STRICT OUTPUT FORMAT:\nLesson:\n<ONE paragraph, 120–160 words. NO headings, NO bullets.>\nQ1:\nQ2:\nQ3:\n"
            ),
        },
    ]


def postprocess_lesson(
    lesson_text: str,
    default_q: str = "Identify one cause and its effect from the lesson.",
) -> str:
    """Clean raw model output and guarantee exactly 3 DISTINCT questions.

    The function *normalizes* the section order, *dedupes* repeated sentences,
    *clamps* lesson length, and *repairs* or *fills* questions.
    """
    text = (lesson_text or "").strip()

    # *normalizing* starting point at 'Lesson:'
    if "Lesson:" in text:
        text = text[text.index("Lesson:") :]
    # *dropping* common hallucinated headings
    text = re.sub(
        r"(?im)^\s*(lecture|objective|duration|materials|introduction|procedure)\s*:.*\n",
        "",
        text,
    )

    # *deduping* consecutive identical sentences
    sents = re.split(r"(?<=[.!?])\s+", text)
    cleaned: List[str] = []
    for s in sents:
        if s and (not cleaned or s.strip().lower() != cleaned[-1].strip().lower()):
            cleaned.append(s.strip())
    text = " ".join(cleaned)

    # *splitting* into non-empty lines
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # *moving* questions under Lesson when output starts with Q1
    if lines and lines[0].lower().startswith("q1:"):
        qs = [ln for ln in lines if re.match(r"(?i)^q[1-9]:", ln)]
        body = "\n".join([ln for ln in lines if ln not in qs])
        text = f"Lesson:\n{body.strip()}\n" + "\n".join(qs)
        lines = [ln for ln in text.splitlines() if ln.strip()]

    # *locating* the lesson block and any existing Qs
    try:
        li = lines.index("Lesson:")
    except ValueError:
        lines = ["Lesson:"] + lines
        li = 0

    # *separating* body and questions
    body_parts: List[str] = []
    question_lines: List[str] = []
    for j in range(li + 1, len(lines)):
        if re.match(r"(?i)^q[1-9]:", lines[j]):
            question_lines = lines[j:]
            break
        body_parts.append(lines[j])

    body = " ".join(body_parts).strip()

    # *clamping* lesson body to ~160 words while preserving sentence boundaries
    words = body.split()
    if len(words) > 175:
        sents = re.split(r"(?<=[.!?])\s+", body)
        keep: List[str] = []
        wc = 0
        for s in sents:
            w = len(s.split())
            if wc + w > 160:
                break
            keep.append(s.strip())
            wc += w
        body = " ".join(keep).rstrip(",;:") + "."

    # *extracting* existing Qs (and *stripping* any inline answers)
    raw_qs: List[str] = []
    for ln in question_lines:
        if re.match(r"(?i)^q[1-9]:", ln):
            ln = re.sub(r"(?i)\s*[-–—]?\s*answer.*$", "", ln).strip()
            if not ln.rstrip().endswith("?"):
                ln = ln.rstrip().rstrip(".") + "?"
            raw_qs.append(ln)

    # *building* a diverse question set
    templates = [
        "Identify one cause and its effect from the lesson.",
        "Explain how one action led to a result in the lesson.",
        "Find one example of a cause that produced an effect, and name both parts.",
    ]

    # *normalizing* and *deduping* existing questions
    seen: set[str] = set()
    uniq_qs: List[str] = []
    for q in raw_qs:
        nq = re.sub(r"\s+", " ", q.lower()).strip()
        nq = re.sub(r"(?i)^q[1-9]:\s*", "", nq)
        if nq not in seen:
            seen.add(nq)
            uniq_qs.append(q)

    # *filling* to ensure exactly 3 distinct questions
    def make_q(n: int, txt: str) -> str:
        return f"Q{n}: {txt.rstrip('?')}?"

    idx = 0
    while len(uniq_qs) < 3 and idx < len(templates):
        t = templates[idx]
        key = re.sub(r"\s+", " ", t.lower()).strip()
        if key not in seen:
            uniq_qs.append(make_q(len(uniq_qs) + 1, t))
            seen.add(key)
        idx += 1

    # *padding* if still short (rare)
    alt = "Describe a cause–effect link you noticed in the lesson."
    while len(uniq_qs) < 3:
        uniq_qs.append(make_q(len(uniq_qs) + 1, alt if len(uniq_qs) == 2 else default_q))

    # *renumbering* Q1/Q2/Q3 and *ensuring* distinctness by value (last pass)
    final_qs: List[str] = []
    seen2: set[str] = set()
    for i, q in enumerate(uniq_qs[:3], start=1):
        qtxt = re.sub(r"(?i)^q[1-9]:\s*", "", q).strip()
        key2 = re.sub(r"\s+", " ", qtxt.lower())
        if key2 in seen2:
            for cand in templates + [alt, default_q]:
                k = re.sub(r"\s+", " ", cand.lower())
                if k not in seen2:
                    qtxt = cand
                    key2 = k
                    break
        seen2.add(key2)
        final_qs.append(make_q(i, qtxt))

    # *rebuilding* full text
    out_lines = ["Lesson:", body] + final_qs
    return "\n".join([ln for ln in out_lines if ln.strip()])


def strip_answers(text: str) -> str:
    """Remove any trailing "Answer…" content from Q-lines while preserving text."""
    lines = text.splitlines()
    cleaned: List[str] = []
    for ln in lines:
        if re.match(r"(?i)^q[1-3]:", ln):
            ln = re.sub(r"(?i)\s*[-–—]?\s*answer.*$", "", ln).rstrip()
        cleaned.append(ln)
    return "\n".join(cleaned)
