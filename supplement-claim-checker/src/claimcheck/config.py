"""Single source of truth for paths and settings.

Every tunable lives here so that later experiments (chunk size, top-k, model
names) can be compared by changing one file and re-running the evals.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "pubmed"
PROCESSED_DIR = DATA_DIR / "processed"

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY") or None

MAX_PER_TOPIC = int(os.getenv("MAX_PER_TOPIC", "1500"))
# Below this, an "abstract" is usually a stub or a one-line note with no
# findings. Short but real abstracts (~250 chars) must survive.
MIN_ABSTRACT_CHARS = 200
