# src/generation/generate.py
"""Generate a lesson with retrieval, planning, writing, and fairness checks.

This module orchestrates the pipeline:
1) Retrieving and re-ranking candidate passages for a given *topic* and *objective*.
2) Mining, cleaning, and selecting cause→effect pairs (or backing off to LLM planning).
3) Writing a short lesson grounded on selected pairs or a chosen passage.
4) Scoring fairness/toxicity/readability and attempting a single-pass rewrite if needed.
5) Saving artifacts and appending provenance entries.

It is designed to be called both programmatically (via :func:`generate`) and as
an executable module (`python -m src.generation.generate` or running this file).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .retrieval import query_passages, rerank_by_objective
from .planning import (
    clean_pairs,
    extract_pairs_from_texts,
    is_good_pair,
    plan_with_llm,
    refine_pair_text,
    tidy_pair,
)
from .writing import (
    build_write_prompt_with_context,
    postprocess_lesson,
    strip_answers,
    write_lesson,
)
from .utils import chat_generate
from .fairness_ext import score_fairness, rewrite_to_reduce_issues
from src.provenance.prov import add_provenance_multi


def _print_fairness_summary(f: Dict[str, Any]) -> None:
    """Print a compact fairness summary for CLI runs.

    Args:
        f: A dict returned by :func:`score_fairness`.
    """
    status = "passed ✅" if f.get("passed") else "needs attention ❗"
    print("\nFairness check:", status)

    # Printing overall toxicity if present
    tox = f.get("toxicity", {}).get("toxicity")
    if isinstance(tox, (int, float)):
        try:
            print("  toxicity:", f"{tox:.3f}")
        except Exception:
            print("  toxicity:", tox)

    # Printing Flesch–Kincaid grade level
    fk = f.get("readability", {}).get("fk_grade")
    if fk is not None:
        print("  FK grade:", fk)

    # Printing unique non-inclusive terms (if any)
    ni = f.get("non_inclusive", {}).get("flagged", [])
    if ni:
        terms = ", ".join(sorted({t for _, t in ni}))
        print("  non-inclusive terms:", terms)

    # Noting gender–occupation heuristic flag
    go = f.get("gendered_occupations", {})
    if go and not go.get("passed", True):
        print("  note:", "gender–occupation heuristic flagged potential stereotypes")


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
    backend: str = "huggingface",
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Run the end-to-end pipeline and return the lesson, contexts, and fairness.

    Args:
        objective: The learning objective guiding passage ranking and lesson writing.
        topic: The topic to retrieve from the RDF corpus.
        top_k: The number of passages to select after re-ranking.
        model_id: The model identifier for the specified *backend*.
        ttl_path: The RDF Turtle path for the corpus.
        difficulty: Optional difficulty anchor (exact or within *diff_band*).
        diff_band: The band around *difficulty* to include in retrieval.
        min_words: Minimum word count for retrieved passages.
        limit: Maximum number of candidate passages to retrieve.
        backend: "huggingface" or "openai" for generation.

    Returns:
        (lesson, chosen_contexts, fairness_dict).
    """
    # Retrieving candidate passages and guarding empty results
    items = query_passages(topic, ttl_path, difficulty, diff_band, min_words, limit)
    if not items:
        print(f"No passages found for topic '{topic}' in {ttl_path}.")
        return "", [], {"passed": True}

    print(">>> Retrieving and re-ranking passages...")
    passages = [x["text"] for x in items]

    # Re-ranking by objective similarity and collecting chosen contexts
    ranked = rerank_by_objective(objective, passages, top_k=top_k)
    chosen = [items[i] | {"similarity": score} for i, score in ranked]

    print(">>> Chosen contexts:")
    for x in chosen:
        print(f" - {x['title']} (diff={x['difficulty']}, sim={x['similarity']:.2f})")

    # Mining cause→effect pairs heuristically and cleaning them
    print(">>> Mining cause→effect pairs heuristically...")
    pairs = extract_pairs_from_texts([x["text"] for x in chosen], max_pairs=5)
    pairs = [tidy_pair(p) for p in pairs]
    pairs = clean_pairs(pairs)

    # Keeping only good pairs (or backing off to LLM planning if few)
    good = [p for p in pairs if is_good_pair(p)]
    if len(good) < 2:
        print(">>> Heuristic <2 good pairs; backing off to model planning...")
        llm_pairs = plan_with_llm(chosen, model_id, backend)
        pairs = clean_pairs(good + llm_pairs)
    else:
        pairs = good[:5]

    # Refining texts and limiting to top 5
    pairs = [refine_pair_text(p) for p in pairs]
    pairs = clean_pairs(pairs)[:5]

    print(">>> Pairs:", pairs if pairs else "[none]")

    # Deciding pair-mode when at least two good pairs exist
    use_pair_mode = len(pairs) >= 2

    # Writing the lesson either from pairs or a single chosen passage
    print(">>> Writing lesson...")
    if use_pair_mode:
        # write_lesson returns (cleaned_lesson, fairness); we’ll run our fairness scorer instead
        lesson, _ = write_lesson(objective, pairs, model_id, backend)
    else:
        passage_text = chosen[0]["text"]
        msgs = build_write_prompt_with_context(objective, passage_text)
        raw_lesson = chat_generate(
            msgs,
            backend=backend,
            model_id=model_id,
            temperature=0.25,
            max_new_tokens=260,
        )
        lesson = postprocess_lesson(
            raw_lesson,
            default_q=f"Identify one example related to: {objective}",
        )
        lesson = strip_answers(lesson)

    print(">>> Write step complete.")

    # Scoring fairness and attempting a single rewrite if needed
    fairness = score_fairness(lesson)
    if not fairness["passed"]:
        print(">>> Fairness issues detected; attempting one rewrite...")
        fixed = rewrite_to_reduce_issues(
            text=lesson,
            issues=fairness,
            backend=backend,
            model_id=model_id,
            chat_generate=chat_generate,
        )
        fixed = postprocess_lesson(
            fixed, default_q=f"Identify one example related to: {objective}"
        )
        fixed = strip_answers(fixed)
        fairness2 = score_fairness(fixed)
        if fairness2["passed"]:
            lesson = fixed
            fairness = fairness2

    # Creating outputs directory and stamping filenames
    os.makedirs("outputs", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # Building a minimal run record for later inspection
    record: Dict[str, Any] = {
        "objective": objective,
        "topic": topic,
        "model_id": model_id,
        "chosen": [
            {
                "uri": x["uri"],
                "title": x["title"],
                "difficulty": x["difficulty"],
                "similarity": x["similarity"],
            }
            for x in chosen
        ],
        "pairs": pairs,
        "fairness": fairness,
    }

    # Writing artifacts
    with open(f"outputs/run-{stamp}.json", "w", encoding="utf-8") as fjson:
        json.dump(record, fjson, ensure_ascii=False, indent=2)
    with open(f"outputs/run-{stamp}.txt", "w", encoding="utf-8") as ftxt:
        ftxt.write(lesson)

    # Appending provenance entries
    prov_graph = Path("rdf/provenance.ttl")
    activity_id = f"run-{stamp}"
    agent_name = os.environ.get("PROV_AGENT", "Faraz Ahmed")
    used_paths = [ttl_path, __file__]
    generated_paths = [f"outputs/run-{stamp}.json", f"outputs/run-{stamp}.txt"]

    try:
        add_provenance_multi(
            graph_path=prov_graph,
            activity_id=activity_id,
            agent_name=agent_name,
            used_paths=used_paths,
            generated_paths=generated_paths,
            started=datetime.now(timezone.utc),
            ended=datetime.now(timezone.utc),
        )
        print(f"[prov] Appended provenance to {prov_graph}")
    except Exception as e:  # noqa: BLE001
        print(f"[prov] WARN: failed to append provenance: {e}")

    # Printing outputs for CLI usage
    print("=== LESSON ===")
    print(lesson.strip(), "\n")
    print("=== EXPLANATION ===")
    for x in chosen:
        print(
            f"Selected because: topic='{topic}', difficulty={x['difficulty']}, "
            f"objective_similarity={x['similarity']:.2f}."
        )
        print(f"URI: {x['uri']} | Title: {x['title']}")
    _print_fairness_summary(fairness)

    return lesson, chosen, fairness


if __name__ == "__main__":
    # Parsing CLI arguments and running the generator
    p = argparse.ArgumentParser(description="Generate a fairness-checked lesson for a topic/objective.")
    p.add_argument("--objective", required=True, help="Learning objective guiding the lesson")
    p.add_argument("--topic", required=True, help="Topic to retrieve from the RDF corpus")
    p.add_argument("--top_k", type=int, default=2, help="How many contexts to keep after re-ranking")
    p.add_argument("--model_id", default="microsoft/Phi-3-mini-4k-instruct", help="Model id/name for generation")
    p.add_argument("--ttl", default="rdf/corpus.ttl", help="Path to the RDF corpus (TTL)")
    p.add_argument("--difficulty", type=int, default=None, help="Optional difficulty anchor")
    p.add_argument("--diff_band", type=int, default=1, help="Difficulty band around the anchor")
    p.add_argument("--min_words", type=int, default=60, help="Minimum passage word count")
    p.add_argument("--limit", type=int, default=50, help="Max number of candidate passages to retrieve")
    p.add_argument(
        "--backend",
        default="huggingface",
        choices=["huggingface", "openai"],
        help="Which backend to use for text generation",
    )

    args = p.parse_args()
    generate(
        args.objective,
        args.topic,
        args.top_k,
        args.model_id,
        args.ttl,
        args.difficulty,
        args.diff_band,
        args.min_words,
        args.limit,
        args.backend,
    )
