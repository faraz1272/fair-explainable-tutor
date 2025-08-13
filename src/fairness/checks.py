# src/fairness/simple_checks.py
from typing import Dict, List

BAD_TERMS = ["stupid", "lazy", "crazy"]  # extend later

def scan(text: str) -> Dict:
    lower = text.lower()
    flagged: List[str] = [t for t in BAD_TERMS if t in lower]
    return {"flagged_terms": flagged, "passed": len(flagged) == 0}
