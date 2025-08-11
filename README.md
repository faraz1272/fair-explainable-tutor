# Fair & Explainable AI Tutor (MSc Dissertation)

This is the implementation of my MSc dissertation project — an AI-powered tutoring system
that generates personalized, fair, and explainable learning content using a Large Language Model (LLM).

## Project Structure
- `data_raw/` — raw datasets (e.g., CommonLit, Gutenberg)
- `data_processed/` — cleaned and tagged datasets
- `rdf/` — RDF metadata and provenance graphs
- `src/` — main Python modules
- `notebooks/` — Jupyter notebooks for experiments
- `outputs/` — generated lessons and explanations

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt