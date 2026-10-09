"""Shared config/path helpers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.load(open(ROOT / "config.json"))
if (ROOT / "config.local.json").exists():      # git-ignored personal overrides (e.g. your real e-mail for OpenAlex)
    CFG.update(json.load(open(ROOT / "config.local.json")))
BIB = (ROOT / CFG["bibliography_data_dir"]).resolve()
DATA = ROOT / "data"
OUT = ROOT / "output"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

STANCES = ["critical", "mixed", "supportive", "neutral_use", "not_discussed"]
COLOURS = {"critical": "#d1495b", "mixed": "#edae49", "supportive": "#2e86ab",
           "neutral_use": "#8d99ae", "not_discussed": "#d9d9d9"}
