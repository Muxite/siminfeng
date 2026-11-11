"""Allow both pytest and python -m pytest from the campaign root."""
from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
