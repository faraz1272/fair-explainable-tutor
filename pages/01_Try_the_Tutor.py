# 01_Try_the_Tutor.py
"""Streamlit front‑end for the Fair & Explainable Tutor.

This app:
- *loading* topics from an RDF/Turtle corpus,
- *running* the lesson generator with fairness checks,
- *rendering* summaries, contexts, and fairness metrics,
- *recording* provenance into PROV‑O TTL,
- *offering* downloads of artifacts (lesson text, JSON, provenance).
"""

import os, sys, time, json, glob, re
from typing import Optional, Tuple
from pathlib import Path

# ensure repo modules are importable
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
from rdflib import Graph, Namespace

from src.generation.generate import generate
from src.provenance.prov import add_provenance

LOM = Namespace("http://example.org/lom#")

st.set_page_config(page_title="Fair & Explainable Tutor", layout="wide")


# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def list_topics_from_ttl(ttl_path: str) -> list[str]:
    """Return available topics from the TTL corpus."""
    if not os.path.exists(ttl_path):
        return []
    g = Graph().parse(ttl_path, format="turtle")
    # collecting topics
    topics = sorted({str(o) for _, _, o in g.triples((None, LOM.topic, None))})
    return topics


def _latest_run_files(since_ts: float) -> Tuple[Optional[str], Optional[str]]:
    """Return most recent JSON and TXT artifacts created after given timestamp."""
    def latest(globpat: str) -> Optional[str]:
        files = [p for p in glob.glob(globpat) if os.path.getmtime(p) >= since_ts]
        return max(files, key=os.path.getmtime) if files else None

    return latest("outputs/run-*.json"), latest("outputs/run-*.txt")


def _split_questions(lesson: str) -> list[str]:
    """Return lines beginning with Q1–3: from the lesson text."""
    qs = []
    for ln in lesson.splitlines():
        if re.match(r"(?i)^q[1-3]:", ln.strip()):
            qs.append(ln.strip())
    return qs


def _fairness_summary(f: dict) -> dict:
    """Return a compact fairness summary (toxicity, grade, non-inclusive terms)."""
    tox = f.get("toxicity", {}).get("toxicity", None)
    fk = f.get("readability", {}).get("fk_grade", None)
    noninc = f.get("non_inclusive", {}).get("flagged", [])
    return {
        "passed": f.get("passed", False),
        "toxicity": tox,
        "fk_grade": fk,
        "noninclusive_terms": sorted({t for _, t in noninc}) if noninc else [],
    }


# ----- FAIRNESS RENDERING -----
def _fairness_summary_dict(f: dict) -> dict:
    """Return a dict of key fairness indicators for display."""
    f = f or {}
    return {
        "passed": f.get("passed", False),
        "toxicity": f.get("toxicity", {}).get("toxicity", None),
        "fk_grade": f.get("readability", {}).get("fk_grade", None),
        "noninclusive_terms": sorted(
            {t for _, t in f.get("non_inclusive", {}).get("flagged", [])}
        )
        if f.get("non_inclusive")
        else [],
        "gendered_flag": not f.get("gendered_occupations", {}).get("passed", True),
        "pronouns": f.get("pronoun_balance", {}),
    }


def render_fairness(f: dict) -> None:
    """Render a formatted fairness panel."""
    s = _fairness_summary_dict(f)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall", "Passed" if s["passed"] else "Needs attention")
    with col2:
        st.metric(
            "Toxicity",
            f"{s['toxicity']:.3f}" if isinstance(s["toxicity"], (int, float)) else "–",
        )
    with col3:
        st.metric(
            "FK Grade",
            f"{s['fk_grade']:.1f}" if isinstance(s["fk_grade"], (int, float)) else "–",
        )

    if s["noninclusive_terms"]:
        st.markdown("**Non-inclusive terms flagged**")
        st.write(", ".join(s["noninclusive_terms"]))
    else:
        st.markdown("**Non-inclusive terms flagged:** _none_")

    if s["gendered_flag"]:
        st.warning("Potential gender–occupation stereotypes flagged (heuristic).")

    if s["pronouns"]:
        pr = s["pronouns"]
        st.markdown(
            f"**Pronoun balance:** she={pr.get('she',0)}, he={pr.get('he',0)}, they={pr.get('they',0)}"
        )


# ----- PROVENANCE RENDERING -----
def infer_activity_id_from_path(json_path: Optional[str]) -> Optional[str]:
    """Infer activity ID from JSON filename (e.g. run-YYYYMMDD-HHMMSS)."""
    if not json_path:
        return None
    base = os.path.basename(json_path)
    name, _ = os.path.splitext(base)
    return name


def render_provenance(
    ttl_path: str,
    activity_json: Optional[str],
    activity_txt: Optional[str],
    backend: str,
    model_id: str,
    objective: str,
    topic: str,
) -> None:
    """Render provenance panel and optionally record provenance triples."""
    st.markdown("### Provenance")
    activity_id = infer_activity_id_from_path(activity_json)

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**Activity**")
        st.write(f"- ID: `{activity_id or 'n/a'}`")
        st.write(f"- Objective: {objective}")
        st.write(f"- Topic: {topic}")
        st.write(f"- Backend/Model: `{backend} / {model_id}`")
        st.write(f"- Corpus (TTL): `{ttl_path}`")

    with rcol:
        st.markdown("**Artifacts**")
        st.write(f"- JSON: `{activity_json or 'n/a'}`")
        st.write(f"- Lesson: `{activity_txt or 'n/a'}`")
        st.write(f"- PROV graph: `rdf/provenance.ttl`")

        if os.path.exists("rdf/provenance.ttl"):
            with open("rdf/provenance.ttl", "rb") as fh:
                st.download_button(
                    "Download provenance.ttl", data=fh.read(), file_name="provenance.ttl"
                )

        if st.button(
            "Record this run to provenance.ttl",
            key="prov_record",
            disabled=activity_id is None,
        ):
            try:
                triples = add_provenance(
                    graph_path=Path("rdf/provenance.ttl"),
                    activity_id=activity_id or "unknown-activity",
                    agent_name="author",
                    used_path=ttl_path,
                    generated_path=activity_txt or activity_json,
                )
                st.success(f"Provenance updated (added {triples} triples).")
            except Exception as e:
                st.error(f"Failed to record provenance: {e}")

    with st.expander("View provenance.ttl"):
        if os.path.exists("rdf/provenance.ttl"):
            with open("rdf/provenance.ttl", "r", encoding="utf-8") as fh:
                st.code(fh.read(), language="turtle")
        else:
            st.info("No provenance file found yet. It will be created after the first run.")


def render_context_rationale(
    chosen: list[dict], objective: str, topic: str, top_k: int, min_words: int
) -> None:
    """Explain rationale behind context selection."""
    st.markdown("### Why these contexts?")
    if not chosen:
        st.info("No contexts to explain.")
        return

    st.write(
        f"We selected **{min(top_k, len(chosen))}** passages most relevant to the objective “{objective}” and topic **{topic}**. "
        f"Passages are ranked by semantic similarity and filtered to roughly **{min_words}+** words where possible."
    )

    bullets = []
    for c in chosen:
        title = c.get("title", "")
        diff = c.get("difficulty", "")
        sim = c.get("similarity", None)
        uri = c.get("uri", "")
        sim_txt = (
            f"{sim:.2f}"
            if isinstance(sim, (int, float))
            else (str(sim) if sim is not None else "n/a")
        )
        bullets.append(
            f"- **{title}** (difficulty: {diff}, similarity: {sim_txt}) — selected for topical fit; source: `{uri}`"
        )

    st.markdown("\n".join(bullets))


# ---------- sidebar ----------
st.sidebar.header("Controls")

ttl_path = st.sidebar.text_input("TTL path", value="data/rdf/corpus.ttl")
topics = list_topics_from_ttl(ttl_path)
objective = st.sidebar.text_input(
    "Learning objective", value="Teach cause and effect in short stories"
)

if topics:
    topic = st.sidebar.selectbox("Topic", topics, index=0)
else:
    topic = st.sidebar.text_input("Topic (free text)", value="Society")

# --- Model & Backend selection (OpenAI default) ---
provider = st.sidebar.radio(
    "Provider",
    ["OpenAI", "Hugging Face"],
    index=0,            # OpenAI by default
    horizontal=True
)
backend = "openai" if provider == "OpenAI" else "huggingface"

if provider == "OpenAI":
    model_oa = st.sidebar.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0
    )
    model_hf = None
    # Optional: nudge if API key is missing
    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning(
            "OPENAI_API_KEY is not set. Add it to your environment or .env before running.",
            icon="⚠️",
        )
else:
    st.sidebar.warning(
        "Hugging Face models can be slow to load the first time due to large downloads. "
        "On Apple Silicon (MPS), they may also run in float32 for stability.",
        icon="⏳",
    )
    model_hf = st.sidebar.selectbox(
        "Hugging Face model",
        ["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-7B-Instruct"],
        index=0
    )
    model_oa = None

top_k = st.sidebar.slider("Top-K passages", 1, 3, 2)
min_words = st.sidebar.slider("Min words per passage", 1, 120, 60)
limit = st.sidebar.slider("Max results from TTL", 10, 200, 50)

run_btn = st.sidebar.button("Generate lesson", use_container_width=True)

st.title("📚 Fair & Explainable Tutor")
st.caption("Builds a short lesson + 3 questions from your RDF corpus, with fairness checks.")


# ---------- main interaction ----------
if run_btn:
    model_id = model_hf if backend == "huggingface" else model_oa
    start_ts = time.time()

    with st.spinner("Running generator…"):
        result = generate(
            objective=objective,
            topic=topic,
            top_k=top_k,
            model_id=model_id,
            ttl_path=ttl_path,
            difficulty=None,
            diff_band=1,
            min_words=min_words,
            limit=limit,
            backend=backend,
        )

    if not result:
        st.error("No result (likely no passages found for that topic). Check TTL path/topic.")
    else:
        lesson, chosen, fairness = result

        # top row summary
        cols = st.columns([2, 1, 1, 1])
        with cols[0]:
            st.subheader("Lesson")
        summary = _fairness_summary(fairness if isinstance(fairness, dict) else {})
        with cols[1]:
            st.metric("Fairness", "Passed" if summary["passed"] else "Needs attention")
        with cols[2]:
            st.metric(
                "Toxicity",
                f"{summary['toxicity']:.3f}"
                if isinstance(summary["toxicity"], (int, float))
                else "–",
            )
        with cols[3]:
            st.metric(
                "FK Grade",
                f"{summary['fk_grade']:.1f}"
                if isinstance(summary["fk_grade"], (int, float))
                else "–",
            )

        st.text_area("Lesson + Questions", value=lesson, height=280)

        qs = _split_questions(lesson)
        if qs:
            st.markdown("**Comprehension Questions**")
            st.write("\n".join([f"- {q}" for q in qs]))

        st.markdown("**Chosen Contexts**")
        if chosen:
            df = pd.DataFrame(
                [
                    {
                        "title": c.get("title", ""),
                        "difficulty": c.get("difficulty", ""),
                        "similarity": f"{c.get('similarity',0):.2f}"
                        if isinstance(c.get("similarity", 0), (float, int))
                        else c.get("similarity", ""),
                        "uri": c.get("uri", ""),
                    }
                    for c in chosen
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No contexts returned.")

        with st.expander("Why these contexts?"):
            render_context_rationale(chosen or [], objective, topic, top_k, min_words)

        with st.expander("Fairness report"):
            render_fairness(fairness if isinstance(fairness, dict) else {})

        json_path, txt_path = _latest_run_files(start_ts)
        with st.expander("Artifacts"):
            if json_path and os.path.exists(json_path):
                st.caption(f"Run JSON: `{json_path}`")
                meta = json.load(open(json_path, encoding="utf-8"))
                pairs = meta.get("pairs", [])
                if pairs:
                    st.markdown("**Extracted cause→effect pairs**")
                    st.write("\n".join([f"- {p}" for p in pairs]))
                st.download_button(
                    "Download JSON",
                    data=json.dumps(meta, indent=2),
                    file_name=os.path.basename(json_path),
                )
            else:
                st.caption("Run JSON not found.")

            if txt_path and os.path.exists(txt_path):
                st.caption(f"Run Text: `{txt_path}`")
                txt = open(txt_path, encoding="utf-8").read()
                st.download_button(
                    "Download Lesson (.txt)", data=txt, file_name=os.path.basename(txt_path)
                )
            else:
                st.caption("Run text not found.")

        render_provenance(
            ttl_path=ttl_path,
            activity_json=json_path,
            activity_txt=txt_path,
            backend=backend,
            model_id=model_id,
            objective=objective,
            topic=topic,
        )
else:
    st.info("Set TTL path and topic, then click **Generate lesson**.")
