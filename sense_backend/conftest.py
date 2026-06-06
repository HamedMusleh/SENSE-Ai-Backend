"""
Pytest configuration: ensure the project root is importable so that
`from backend.main import app` works regardless of where pytest is invoked.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
