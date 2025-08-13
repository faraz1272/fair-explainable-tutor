from pathlib import Path
import pandas as pd

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")

def load_commonlit_sample() -> pd.DataFrame:
    """
    Placeholder: replace with a real CSV later.
    For now, we mock 5 passages so downstream code works.
    """
    rows = [
        {"id": "p1", "title": "A River Journey", "text": "The river bends and flows...", "topic": "Nature"},
        {"id": "p2", "title": "Courage in Silence", "text": "She stood firm as the crowd hushed...", "topic": "Character"},
        {"id": "p3", "title": "Turning Gears", "text": "Engines hummed as the city awoke...", "topic": "Technology"},
        {"id": "p4", "title": "Seeds of Change", "text": "Planting ideas like seeds in spring...", "topic": "Society"},
        {"id": "p5", "title": "Night Lanterns", "text": "Lanterns drifted above the lake...", "topic": "Tradition"},
    ]
    return pd.DataFrame(rows)