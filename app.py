# app.py — Landing page
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Fair & Explainable Tutor",
    page_icon="🎓",
    layout="wide"
)

# ---------- HERO ----------
st.markdown("""
# 🎓 Fair & Explainable AI-Powered Tutoring

Generate **grounded** mini-lessons that teach **cause and effect** from classic short stories.
Every run includes a **fairness panel** and **PROV-O provenance** for traceability.
""")

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Objective", "Cause ↔ Effect")
with col_b:
    st.metric("Grounding", "RDF + Retrieval")
with col_c:
    st.metric("Governance", "Fairness Signals")
with col_d:
    st.metric("Traceability", "PROV-O Logging")

st.divider()

# ---------- CALL TO ACTION ----------
st.subheader("Try it now")
st.page_link("pages/01_Try_the_Tutor.py", label="▶️ Open the Tutor", icon=":material/rocket:")

st.caption("The interactive page lets you pick a topic, see retrieved passages, mined pairs, "
           "the generated lesson with three distinct questions, the fairness report, and provenance info.")

st.divider()

# ---------- WHY THIS PROJECT ----------
st.subheader("Why this project?")
st.markdown("""
- **Ungrounded outputs** can drift from the source text and reduce educational value.  
- **Repetitive, vague questions** don’t help learners practice reasoning.  
- **Opaque generation** makes review and audit difficult.  

This project narrows scope to a single learning goal — **cause and effect** — and builds a **transparent pipeline**:
**retrieve → re-rank → mine pairs → write → post-process → fairness → provenance**.
""")

# ---------- HOW IT WORKS ----------
st.subheader("How it works (at a glance)")
steps = [
    ("Corpus (RDF/TTL)", "Chunks (150–300 words) of public-domain stories with title, topic, difficulty, and text."),
    ("Retrieval & Re-rank", "Filter by topic/min length, then rank by semantic similarity to the objective."),
    ("Pair Mining", "Heuristics for causal cues; fallback to a small LLM constrained to the selected context."),
    ("Writer + Post-process", "120–160 word paragraph + exactly 3 distinct questions; de-dupe and trim."),
    ("Fairness Panel", "Toxicity (Detoxify), FK Grade, non-inclusive lexicon, pronoun balance, stereotype heuristic."),
    ("Provenance (PROV-O)", "Log inputs, outputs, model/params, and timestamps for auditability.")
]
for i, (title, desc) in enumerate(steps, 1):
    st.markdown(f"**{i}. {title}.** {desc}")

st.divider()

# ---------- ARCHITECTURE IMAGE (optional) ----------
left, right = st.columns([1.2, 1])
with left:
    st.subheader("Architecture")
    st.markdown("The pipeline ties together semantic retrieval, explainable pair mining, and governance layers.")
    arch = Path("assets/system_architecture.png")
    if arch.exists():
        st.image(str(arch), use_column_width=True, caption="System architecture")
    else:
        st.info("Add an image at `assets/system_architecture.png` to show the pipeline diagram.")

with right:
    st.subheader("Key Features")
    st.markdown("""
- **Grounded generation** from classic stories  
- **Explainable** cause→effect notes  
- **Format guarantees**: 1 paragraph + 3 unique questions  
- **Fairness signals** at-a-glance  
- **PROV-O records** for each run  
- **Streamlit UI** for demo
""")

st.divider()

# ---------- ETHICS / GOVERNANCE ----------
st.subheader("Ethics & Governance")
st.markdown("""
- Designed for **transparency** and **auditability**: every generation run is recorded.  
- Fairness panel provides quick indicators; not a definitive assessment but a **useful early signal**.  
- The corpus uses **public-domain texts**; no personal data are stored.
""")

# ---------- FOOTER ----------
st.divider()
st.caption("Built with Streamlit 1.36 • Hugging Face small models • RDF via rdflib • Detoxify for toxicity scoring")