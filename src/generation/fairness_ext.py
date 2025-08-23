# src/generation/fairness_ext.py
from __future__ import annotations
"""Fairness, inclusivity, and readability checks for generated text.

This module provides:
- Lightweight heuristics for non-inclusive language and gender–occupation patterns.
- Pronoun balance and readability (Flesch–Kincaid grade) checks.
- Detoxify-based toxicity scores using the "unbiased" checkpoint.
- An optional one-shot rewrite utility that calls a provided `chat_generate` function
  to reduce issues when they are detected.

All functions return JSON-serialisable structures to ease logging and UI display.
"""

import re
from typing import Dict, Any, List, Tuple

from detoxify import Detoxify
import textstat

# ------------------------------
# Model loading utilities
# ------------------------------
_TOX = None

def _tox() -> Detoxify:
    """Return a cached Detoxify model instance.

    Loading the model is relatively expensive; we cache it the first time
    and reuse it across subsequent calls to reduce latency and memory overhead.
    """
    global _TOX
    if _TOX is None:
        # Loading "unbiased" variant to reduce demographic correlations
        _TOX = Detoxify("unbiased")
    return _TOX


# ------------------------------
# Pattern definitions
# ------------------------------
# Extending these lists allows tailoring the checks to your product domain.
NON_INCLUSIVE: Dict[str, List[str]] = {
    "ableist": [
        r"\bcrazy\b", r"\binsane\b", r"\bpsycho\b", r"\bmadman\b", r"\blame\b",
        r"\bmidget\b", r"\bcripple\b", r"\bdumb\b", r"\bstupid\b",
    ],
    "gendered": [
        r"\bmanpower\b", r"\bchairman\b", r"\bpoliceman\b", r"\bfireman\b",
    ],
}

# Occupation list for a simple gender–stereotype heuristic
OCCUPATIONS: List[str] = [
    "nurse", "secretary", "teacher", "assistant", "engineer", "programmer", "scientist",
    "pilot", "doctor", "surgeon", "professor", "manager", "director", "ceo", "founder",
    "janitor", "driver", "chef", "cook", "writer", "artist", "musician", "athlete",
]

PRONOUNS_M = r"\b(he|him|his)\b"
PRONOUNS_F = r"\b(she|her|hers)\b"


def _count_regex(text: str, pattern: str) -> int:
    """Count the number of case-insensitive regex matches in *text* for *pattern*."""
    return len(re.findall(pattern, text, flags=re.I))


# ------------------------------
# Individual checks
# ------------------------------

def toxicity_scores(text: str) -> Dict[str, float]:
    """Compute Detoxify scores for *text*.

    Returns a mapping of score names to floats, e.g.,
    {"toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"}.
    """
    scores = _tox().predict(text or "")
    # Converting numpy types to plain floats for JSON serialisation
    return {k: float(v) for k, v in scores.items()}


def check_non_inclusive_language(text: str) -> Dict[str, Any]:
    """Scan *text* for non-inclusive terms and return hits.

    Returns
    -------
    dict with keys:
      - "flagged": List[Tuple[category, term]] of matched terms.
      - "passed": bool indicating whether no terms were found.
    """
    flagged: List[Tuple[str, str]] = []
    for bucket, patterns in NON_INCLUSIVE.items():
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.I):
                # Recording matched term with category
                flagged.append((bucket, m.group(0)))
    return {"flagged": flagged, "passed": len(flagged) == 0}


def check_gendered_occupations(text: str, window: int = 6) -> Dict[str, Any]:
    """Flag potential gender–occupation stereotypes within a token window.

    We flag patterns like "he … engineer" or "she … nurse" if a gendered pronoun
    appears within *window* tokens around an occupation.

    Notes
    -----
    This is a heuristic. It does not assert identity; it highlights potentially
    stereotyped pairings for human review.
    """
    # Tokenising by words and punctuation to maintain positions
    tokens = re.findall(r"\w+|\S", text.lower())
    occ_set = set(OCCUPATIONS)
    hits: List[str] = []

    # Scanning tokens and collecting nearby pronouns
    for i, tok in enumerate(tokens):
        if tok in occ_set:
            left = " ".join(tokens[max(0, i - window): i])
            right = " ".join(tokens[i + 1: i + 1 + window])
            if re.search(PRONOUNS_M, left) or re.search(PRONOUNS_M, right):
                hits.append(f"male_pronoun→{tok}")
            if re.search(PRONOUNS_F, left) or re.search(PRONOUNS_F, right):
                hits.append(f"female_pronoun→{tok}")

    # Assessing balance: if only one side appears, we consider it imbalanced
    male = [h for h in hits if h.startswith("male_pronoun")]
    female = [h for h in hits if h.startswith("female_pronoun")]
    return {
        "pairs": hits,
        "balanced": not (bool(male) ^ bool(female)),
        "passed": len(hits) == 0 or (bool(male) and bool(female)),
    }


def pronoun_balance(text: str) -> Dict[str, Any]:
    """Compute gendered and neutral pronoun counts and assess rough balance.

    The check passes when no gendered pronouns are used, or when male/female counts
    are roughly balanced (within 33%–67% split). Neutral pronouns (they/them) are
    reported but do not affect the pass criterion directly.
    """
    m = _count_regex(text, PRONOUNS_M)
    f = _count_regex(text, PRONOUNS_F)
    they = _count_regex(text, r"\bthey|them|their|theirs\b")
    total = m + f + they
    return {
        "counts": {"male": m, "female": f, "they": they},
        "ratio_m_f": (m / f) if f else None,
        "passed": total == 0 or (0.33 <= (m / (m + f)) <= 0.67 if (m + f) else True),
    }


def reading_level(text: str) -> Dict[str, Any]:
    """Compute the Flesch–Kincaid grade level and check audience suitability.

    The default pass range is 2.0–9.0 and should be tuned to your audience.
    When computation fails, we return `passed=True` with `fk_grade=None`.
    """
    try:
        grade = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        grade = None
    return {"fk_grade": grade, "passed": True if grade is None else (2.0 <= grade <= 9.0)}


# ------------------------------
# Aggregate scoring
# ------------------------------

def score_fairness(text: str) -> Dict[str, Any]:
    """Aggregate all fairness-related checks for *text*.

    Returns
    -------
    dict with keys: "passed", "toxicity", "non_inclusive", "gendered_occupations",
    "pronoun_balance", and "readability".
    """
    tox = toxicity_scores(text)
    noninc = check_non_inclusive_language(text)
    gender_occ = check_gendered_occupations(text)
    pronouns = pronoun_balance(text)
    level = reading_level(text)

    # Applying thresholds (tune as needed)
    tox_flag = max(tox.values()) > 0.30 or tox.get("toxicity", 0.0) > 0.20
    passed = (not tox_flag) and noninc["passed"] and gender_occ["passed"] and pronouns["passed"]

    return {
        "passed": passed,
        "toxicity": tox,
        "non_inclusive": noninc,
        "gendered_occupations": gender_occ,
        "pronoun_balance": pronouns,
        "readability": level,
    }


# ------------------------------
# Optional rewrite
# ------------------------------

def rewrite_to_reduce_issues(
    text: str,
    issues: Dict[str, Any],
    backend: str,
    model_id: str,
    chat_generate,
) -> str:
    """Rewrite *text* once to reduce detected issues.

    Parameters
    ----------
    text: The original text to revise.
    issues: The output from :func:`score_fairness` (or similarly-shaped dict).
    backend: Backend identifier passed through to `chat_generate` (e.g., "openai" or "huggingface").
    model_id: Model identifier/name for the chosen backend.
    chat_generate: Callable(messages, backend, model_id, ...) -> str that performs generation.

    Returns
    -------
    str: A revised text when problems are found; otherwise returns the original *text*.
    """
    problems: List[str] = []

    # Collecting actionable guidance from detected issues
    if max(issues["toxicity"].values()) > 0.30:
        problems.append("reduce toxicity and harsh wording")
    if not issues["non_inclusive"]["passed"]:
        found = [w for _, w in issues["non_inclusive"]["flagged"]]
        problems.append("replace non-inclusive terms: " + ", ".join(sorted(set(found))[:6]))
    if not issues["gendered_occupations"]["passed"]:
        problems.append("avoid stereotyping genders with occupations")
    if not issues["pronoun_balance"]["passed"]:
        problems.append("balance or neutralize gendered pronouns")

    if not problems:
        # Returning original when nothing needs fixing
        return text

    # Building a concise editing prompt
    msgs = [
        {"role": "system", "content": "You are a careful editor for inclusive, age-appropriate educational text."},
        {
            "role": "user",
            "content": (
                "Rewrite the following lesson to address these concerns: "
                + "; ".join(problems)
                + ". Keep meaning the same, keep ~140 words, and keep exactly 3 comprehension questions "
                + "labeled Q1..Q3 at the end.\n\nLesson to fix:\n" + text
            ),
        },
    ]

    # Generating a single-pass rewrite
    new_text = chat_generate(
        msgs,
        backend=backend,
        model_id=model_id,
        temperature=0.2,
        max_new_tokens=260,
    )
    return new_text.strip() if new_text else text
