"""
conftest.py — makes src/ importable from tests/ without each test file
needing its own sys.path hack. pytest auto-discovers this file from the
project root and applies it to the whole test session.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
