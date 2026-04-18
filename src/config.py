"""
FedAcuity config loader.
Usage: from src.config import cfg
"""
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


cfg = load_config()
