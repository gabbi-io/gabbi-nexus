from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv('GABBI_DATA_DIR', BASE_DIR / 'data'))
CASES_DIR = DATA_DIR / 'cases'
STATIC_DIR = BASE_DIR / 'app' / 'static'
MAX_TEXT_PREVIEW = 12000
TOP_K_CHUNKS = 6
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
DEFAULT_LLM_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
