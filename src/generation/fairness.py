# src/generation/fairness.py
def simple_fairness_scan(text: str):
    flagged, bad_terms = [], ["stupid", "lazy", "crazy"]
    low = text.lower()
    for t in bad_terms:
        if t in low: flagged.append(t)
    return {"flagged_terms": flagged, "passed": len(flagged) == 0}
