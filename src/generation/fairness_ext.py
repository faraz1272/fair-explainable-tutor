# src/generation/fairness_ext.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple
from detoxify import Detoxify
import textstat

# ---------- helpers ----------
_TOX = None
def _tox():
    global _TOX
    if _TOX is None:
        # 'unbiased' is the Detoxify variant trained to reduce demographic correlations
        _TOX = Detoxify('unbiased')
    return _TOX

# Non-inclusive / ableist / stigmatizing words (extendable)
NON_INCLUSIVE = {
    "ableist": [
        r"\bcrazy\b", r"\binsane\b", r"\bpsycho\b", r"\bmadman\b", r"\blame\b",
        r"\bmidget\b", r"\bcripple\b", r"\bdumb\b", r"\bstupid\b"
    ],
    "gendered": [
        r"\bmanpower\b", r"\bchairman\b", r"\bpoliceman\b", r"\bfireman\b"
    ],
}

# Occupation list for simple gender stereotype heuristic
OCCUPATIONS = [
    "nurse","secretary","teacher","assistant","engineer","programmer","scientist",
    "pilot","doctor","surgeon","professor","manager","director","ceo","founder",
    "janitor","driver","chef","cook","writer","artist","musician","athlete"
]
PRONOUNS_M = r"\b(he|him|his)\b"
PRONOUNS_F = r"\b(she|her|hers)\b"

def _count_regex(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))

# ---------- individual checks ----------
def toxicity_scores(text: str) -> Dict[str, float]:
    """
    Detoxify scores: toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit
    """
    scores = _tox().predict(text or "")
    # ensure plain floats (for JSON)
    return {k: float(v) for k, v in scores.items()}

def check_non_inclusive_language(text: str) -> Dict[str, Any]:
    flagged: List[Tuple[str, str]] = []
    for bucket, patterns in NON_INCLUSIVE.items():
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.I):
                flagged.append((bucket, m.group(0)))
    return {
        "flagged": flagged,
        "passed": len(flagged) == 0
    }

def check_gendered_occupations(text: str, window: int = 6) -> Dict[str, Any]:
    """
    Flags patterns like 'he ... engineer' or 'she ... nurse' within a small window.
    This is a heuristic; it doesn't assert identity, just highlights potential stereotypes.
    """
    tokens = re.findall(r"\w+|\S", text.lower())
    occ_set = set(OCCUPATIONS)
    hits: List[str] = []
    for i, tok in enumerate(tokens):
        if tok in occ_set:
            left = " ".join(tokens[max(0, i-window):i])
            right = " ".join(tokens[i+1:i+1+window])
            if re.search(PRONOUNS_M, left) or re.search(PRONOUNS_M, right):
                hits.append(f"male_pronoun→{tok}")
            if re.search(PRONOUNS_F, left) or re.search(PRONOUNS_F, right):
                hits.append(f"female_pronoun→{tok}")
    # imbalance if only one side appears with “stereotyped” pairings
    male = [h for h in hits if h.startswith("male_pronoun")]
    female = [h for h in hits if h.startswith("female_pronoun")]
    return {
        "pairs": hits,
        "balanced": not (bool(male) ^ bool(female)),
        "passed": len(hits) == 0 or (bool(male) and bool(female))
    }

def pronoun_balance(text: str) -> Dict[str, Any]:
    m = _count_regex(text, PRONOUNS_M)
    f = _count_regex(text, PRONOUNS_F)
    they = _count_regex(text, r"\bthey|them|their|theirs\b")
    total = m + f + they
    return {
        "counts": {"male": m, "female": f, "they": they},
        "ratio_m_f": (m / f) if f else None,
        "passed": total == 0 or (0.33 <= (m/(m+f)) <= 0.67 if (m+f) else True)
    }

def reading_level(text: str) -> Dict[str, Any]:
    try:
        grade = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        grade = None
    return {
        "fk_grade": grade,
        "passed": True if grade is None else (2.0 <= grade <= 9.0)  # tune for your audience
    }

# ---------- aggregate ----------
def score_fairness(text: str) -> Dict[str, Any]:
    tox = toxicity_scores(text)
    noninc = check_non_inclusive_language(text)
    gender_occ = check_gendered_occupations(text)
    pronouns = pronoun_balance(text)
    level = reading_level(text)

    # thresholds (tune as needed)
    tox_flag = max(tox.values()) > 0.30 or tox.get("toxicity", 0.0) > 0.20
    passed = (not tox_flag) and noninc["passed"] and gender_occ["passed"] and pronouns["passed"]

    return {
        "passed": passed,
        "toxicity": tox,
        "non_inclusive": noninc,
        "gendered_occupations": gender_occ,
        "pronoun_balance": pronouns,
        "readability": level
    }

# ---------- optional: rewrite on fail ----------
def rewrite_to_reduce_issues(text: str, issues: Dict[str, Any],
                             backend: str, model_id: str,
                             chat_generate) -> str:
    """
    One-shot gentle rewrite prompt using your existing chat_generate backend.
    """
    problems = []
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
        return text  # nothing to fix

    msgs = [
        {"role": "system", "content": "You are a careful editor for inclusive, age-appropriate educational text."},
        {"role": "user", "content":
            "Rewrite the following lesson to address these concerns: "
            + "; ".join(problems)
            + ". Keep meaning the same, keep ~140 words, and keep exactly 3 comprehension questions labeled Q1..Q3 at the end.\n\n"
              "Lesson to fix:\n" + text}
    ]
    new_text = chat_generate(msgs, backend=backend, model_id=model_id, temperature=0.2, max_new_tokens=260)
    return new_text.strip() if new_text else text
