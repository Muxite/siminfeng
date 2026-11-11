"""Support the literal `pytest cc/ladder` entry point without installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
