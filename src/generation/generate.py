# src/generation/generate.py
import argparse, re
from .retrieval import query_passages, rerank_by_objective
from .planning import extract_pairs_from_texts, clean_pairs, plan_with_llm, tidy_pair
from .writing import write_lesson, postprocess_lesson, strip_answers, build_write_prompt_with_context
from .utils import chat_generate
from .fairness import simple_fairness_scan
import json, os, time

import os
from openai import OpenAI

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate(objective, topic, top_k=2, model_id="microsoft/Phi-3-mini-4k-instruct",
             ttl_path="rdf/corpus.ttl", difficulty=None, diff_band=1, min_words=60, limit=50, backend: str = "huggingface"):
    items = query_passages(topic, ttl_path, difficulty, diff_band, min_words, limit)
    if not items:
        print(f"No passages found for topic '{topic}' in {ttl_path}."); return
    print(">>> Retrieving and re-ranking passages...")
    passages = [x["text"] for x in items]
    ranked = rerank_by_objective(objective, passages, top_k=top_k)
    chosen = [items[i] | {"similarity": score} for i, score in ranked]
    print(">>> Chosen contexts:")
    for x in chosen: print(f" - {x['title']} (diff={x['difficulty']}, sim={x['similarity']:.2f})")

    print(">>> Mining cause→effect pairs heuristically...")
    pairs = extract_pairs_from_texts([x["text"] for x in chosen], max_pairs=5)
    pairs = [tidy_pair(p) for p in pairs]

    if len(pairs) < 3:
        print(">>> Heuristic <3; backing off to model planning...")
        llm_pairs = plan_with_llm(chosen, model_id, backend)
        for p in llm_pairs:
            if p.lower() not in [q.lower() for q in pairs]:
                pairs.append(p)

    print(">>> Pairs:", pairs if pairs else "[none]")

    print(">>> Writing lesson...")
    if pairs:
        lesson, fairness = write_lesson(objective, pairs, model_id, backend)
    else:
        passage_text = chosen[0]["text"]
        msgs = [
            {"role": "system", "content": "You are a fair and explainable tutoring assistant."},
            {"role": "user", "content": (
                f"Learning objective: {objective}\n\n"
                "Output must strictly follow this format:\n"
                "Lesson:\n<ONE paragraph, 120–160 words, no headings, no bullet points>\n"
                "Q1:\nQ2:\nQ3:\n"
                "Do not repeat questions. Do not add any other text or labels outside this format.\n\n"
                "Text:\n" + passage_text
            )}
        ]
        raw_lesson = chat_generate(
            msgs,
            backend=backend,
            model_id=model_id,
            temperature=0.25,
            max_new_tokens=260
        )
        lesson = postprocess_lesson(raw_lesson, default_q=f"Identify one example related to: {objective}")
        lesson = strip_answers(lesson)
        fairness = simple_fairness_scan(lesson)

    print(">>> Write step complete.")

    os.makedirs("outputs", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    record = {
    "objective": objective,
    "topic": topic,
    "model_id": model_id,
    "chosen": [{"uri": x["uri"], "title": x["title"], "difficulty": x["difficulty"], "similarity": x["similarity"]} for x in chosen],
    "pairs": pairs,
    "fairness": fairness,
    }
    with open(f"outputs/run-{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    with open(f"outputs/run-{stamp}.txt", "w", encoding="utf-8") as f:
        f.write(lesson)

    print("=== LESSON ==="); print(lesson.strip(), "\n")
    print("=== EXPLANATION ===")
    for x in chosen:
        print(f"Selected because: topic='{topic}', difficulty={x['difficulty']}, objective_similarity={x['similarity']:.2f}.")
        print(f"URI: {x['uri']} | Title: {x['title']}")
    print("\nFairness check:", "passed" if fairness["passed"] else f"flagged {fairness['flagged_terms']}")

    return lesson, chosen, fairness

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--model_id", default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--ttl", default="rdf/corpus.ttl")
    p.add_argument("--difficulty", type=int, default=None)
    p.add_argument("--diff_band", type=int, default=1)
    p.add_argument("--min_words", type=int, default=60)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument(
    "--backend",
    default="huggingface",
    choices=["huggingface", "openai"],
    help="Which backend to use for text generation"
    )
    args = p.parse_args()
    generate(args.objective, args.topic, args.top_k, args.model_id,
             args.ttl, args.difficulty, args.diff_band, args.min_words, args.limit, args.backend)
