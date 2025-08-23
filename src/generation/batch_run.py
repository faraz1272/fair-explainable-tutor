# src/generation/batch_run.py
"""Batch runner for generating lessons across multiple topics.

This module:
- parses CLI arguments,
- iterates topics,
- calls `generate()` once per topic,
- computes simple summary metrics (word count, number of questions, fairness pass), and
- writes a CSV summary with paths to the latest run artifacts.

Notes
-----
- We *search* for the most recent `outputs/run-*.json|.txt` created after each topic's start time
  to *associate* artifacts with that topic's run.
- The CSV destination can be overriden with `--out_csv`; otherwise we *create* one in `outputs_new/`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time
from typing import List, Tuple, Optional

from src.generation.generate import generate


def _latest_run_files(since_ts: float) -> Tuple[Optional[str], Optional[str]]:
    """Return the most recent `outputs/run-*.json` and `.txt` created after *since_ts*.

    We *scan* the `outputs/` directory, *filter* by modification time, and *pick* the latest
    matching JSON and TXT files separately.
    """

    def latest(globpat: str) -> Optional[str]:
        files = [p for p in glob.glob(globpat) if os.path.getmtime(p) >= since_ts]
        return max(files, key=os.path.getmtime) if files else None

    return latest("outputs/run-*.json"), latest("outputs/run-*.txt")


def _word_count_from_lesson(lesson_text: str) -> int:
    """Return a word count computed from the body following the `Lesson:` header."""
    if not lesson_text:
        return 0
    body = re.sub(r"(?is).*?Lesson:\s*", "", lesson_text)
    return len(re.findall(r"\b\w+\b", body))


def _q_count_from_lesson(lesson_text: str) -> int:
    """Return the number of question lines starting with Q1..Q3 (case-insensitive)."""
    if not lesson_text:
        return 0
    return len(re.findall(r"(?im)^q[1-3]:", lesson_text))


def main() -> None:
    """Parse arguments, run generation per topic, and write a summary CSV."""
    ap = argparse.ArgumentParser(description="Batch-generate lessons across multiple topics and summarize results.")
    ap.add_argument("--ttl", default="rdf/corpus.ttl", help="Path to the RDF corpus in Turtle format")
    ap.add_argument("--backend", default="huggingface", choices=["huggingface", "openai"], help="Generation backend")
    ap.add_argument("--model_id", default="microsoft/Phi-3-mini-4k-instruct", help="Model id/name for the chosen backend")
    ap.add_argument(
        "--objective",
        default="Teach cause and effect in short stories",
        help="Learning objective that guides retrieval and writing",
    )
    ap.add_argument(
        "--topics",
        nargs="+",
        default=["Nature", "Society", "Ethics", "War", "Psychology"],
        help="One or more topics to query from the RDF corpus",
    )
    ap.add_argument("--top_k", type=int, default=2, help="How many contexts to keep after re-ranking")
    ap.add_argument("--min_words", type=int, default=1, help="Minimum words required in a passage to be considered")
    ap.add_argument("--difficulty", type=int, default=None, help="Optional difficulty anchor for retrieval filter")
    ap.add_argument("--diff_band", type=int, default=1, help="Band around difficulty to include (±)")
    ap.add_argument("--limit", type=int, default=50, help="Max number of candidate passages to retrieve")
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Path for summary CSV (default: outputs_new/summary-<timestamp>.csv)",
    )
    args = ap.parse_args()

    # Creating output directory for summaries (artifacts remain in `outputs/`)
    os.makedirs("outputs_new", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_csv = args.out_csv or f"outputs_new/summary-{stamp}.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Writing CSV header
        w.writerow(
            [
                "topic",
                "objective",
                "backend",
                "model_id",
                "chosen_titles",
                "num_chosen",
                "lesson_words",
                "questions_count",
                "fairness_passed",
                "toxicity_score",
                "readability_fk_grade",
                "run_json",
                "run_txt",
            ]
        )

        for topic in args.topics:
            print(f"\n=== Topic: {topic} ===")
            start = time.time()

            # Running one generation and capturing artifacts
            lesson, chosen, fairness = generate(
                objective=args.objective,
                topic=topic,
                top_k=args.top_k,
                model_id=args.model_id,
                ttl_path=args.ttl,
                difficulty=args.difficulty,
                diff_band=args.diff_band,
                min_words=args.min_words,
                limit=args.limit,
                backend=args.backend,
            )

            # Locating latest run files produced after this topic's start time
            json_path, txt_path = _latest_run_files(start)

            # Deriving simple metrics
            chosen_titles = "; ".join(sorted({c.get("title", "") for c in chosen}))
            num_chosen = len(chosen)
            words = _word_count_from_lesson(lesson)
            qcount = _q_count_from_lesson(lesson)
            tox = ""
            fk = ""
            if isinstance(fairness, dict):
                tox = f"{fairness.get('toxicity', {}).get('toxicity', '')}"
                fk = fairness.get("readability", {}).get("fk_grade", "")

            # Writing a row into the CSV
            w.writerow(
                [
                    topic,
                    args.objective,
                    args.backend,
                    args.model_id,
                    chosen_titles,
                    num_chosen,
                    words,
                    qcount,
                    fairness.get("passed", False) if isinstance(fairness, dict) else False,
                    tox,
                    fk,
                    json_path or "",
                    txt_path or "",
                ]
            )

    print(f"\n✅ Wrote {out_csv}")


if __name__ == "__main__":
    main()
