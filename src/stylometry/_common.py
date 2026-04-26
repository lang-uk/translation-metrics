"""Shared utilities for stylometric analysis modules."""

import json
import re
from pathlib import Path

from tokenize_uk import tokenize_words

ROOT = Path(__file__).resolve().parent.parent.parent
TRANSLATIONS_PATH = ROOT / "results" / "unite-models" / "parsed_translations.json"
RESULTS_DIR = ROOT / "results" / "stylometry"

EN_KEY = "EN_1945_George_Orwell_Animal_Farm"

HUMAN_SYSTEMS = [
    "UK_1947_Ivan_Cherniatynskyi_Kolhosp_tvaryn",
    "UK_1984_Iryna_Dybko_Khutir_tvaryn",
    "UK_1991_Oleksii_Drozdovskyi_Skotoferma",
    "UK_1991_Yurii_Shevchuk_Ferma_rai_dlia_tvaryn",
    "UK_1992_Natalia_Okolitenko_Skotokhutir",
    "UK_2020_Bohdana_Nosenok_Kolhosp_tvaryn",
    "UK_2021_Viacheslav_Stelmakh_Kolhosp_tvaryn",
]
AI_SYSTEMS = ["gpt_5_2", "lapa_translations_combine", "deepl"]
ALL_SYSTEMS = HUMAN_SYSTEMS + AI_SYSTEMS

DYBKO_KEY = "UK_1984_Iryna_Dybko_Khutir_tvaryn"
RANKED_SYSTEMS = [s for s in ALL_SYSTEMS if s != DYBKO_KEY]


def short_name(system: str) -> str:
    return (
        system.replace("UK_", "")
        .replace("_Kolhosp_tvaryn", "")
        .replace("_Khutir_tvaryn", "")
        .replace("_Skotoferma", "")
        .replace("_Ferma_rai_dlia_tvaryn", "")
        .replace("_Skotokhutir", "")
        .replace("_translations_combine", "")
    )


def is_valid_segment(text: str) -> bool:
    t = text.strip()
    return bool(t) and t != "---" and len(t) > 10


def uk_words(text: str) -> list[str]:
    """Tokenize with tokenize-uk, return word tokens only."""
    return [t for t in tokenize_words(text) if re.match(r"\w", t)]


def build_full_text(segments: list[str], en_segments: list[str]) -> str:
    """Concatenate all valid segments into one string."""
    parts = []
    for i, seg in enumerate(segments):
        if i < len(en_segments) and is_valid_segment(en_segments[i]):
            parts.append(seg.strip())
    return " ".join(parts)


def load_translations() -> dict:
    with open(TRANSLATIONS_PATH) as f:
        return json.load(f)


def save_results(data: dict, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved → {path}")
    return path
