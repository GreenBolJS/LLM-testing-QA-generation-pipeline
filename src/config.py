"""
Shared config loading + small utilities used across the pipeline.

Every other module in src/ imports CONFIG from here rather than re-parsing
config.yaml itself, so there is exactly one place that knows the on-disk
layout of the config file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of src/
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml into a plain dict. Raises if the file is missing,
    since every downstream module assumes config values exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. Copy/create it before running the pipeline."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"config.yaml at {path} is empty or invalid.")
    return cfg


CONFIG: dict[str, Any] = load_config()


def get_path(key: str) -> Path:
    """Resolve a *_dir / *_path config entry relative to ROOT_DIR."""
    raw = CONFIG["paths"][key]
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p)


def setup_logging(name: str) -> logging.Logger:
    """Consistent logger setup for every module — same format, same level,
    controlled by config.yaml's logging.level so verbosity is one knob."""
    level_name = CONFIG.get("logging", {}).get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def require_env(var_name: str) -> str:
    """Fetch a required env var, raising a clear error pointing at .env.example
    rather than letting a confusing KeyError/None bubble up from deep inside
    an API client."""
    val = os.environ.get(var_name)
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: {var_name}. "
            f"Copy .env.example to .env, fill it in, and load it "
            f"(e.g. `export $(cat .env | xargs)` or python-dotenv) before running."
        )
    return val
