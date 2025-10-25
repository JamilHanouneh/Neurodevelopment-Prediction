import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load YAML configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    config = load_config()
    log_level = level or config["project"].get("logging_level", "INFO")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger


def ensure_dir(path: Path) -> Path:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_subjects(
    raw_dir: Path, subject_prefix: str, max_subjects: Optional[int] = None
) -> List[Path]:
    """Return sorted subject directories."""
    subjects = sorted(
        [
            p
            for p in raw_dir.iterdir()
            if p.is_dir() and p.name.startswith(subject_prefix)
        ]
    )
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    return subjects


def save_json(data: Dict, path: Path) -> None:
    """Persist dictionary to JSON."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def load_json(path: Path) -> Dict:
    """Load JSON safely."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)

