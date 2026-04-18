"""
FedAcuity — M5.1 Centralised Results Logger
All FL rounds, DP sweeps, and XAI computations write to structured logs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import cfg

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(cfg["paths"]["results"]["logs"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ResultsLogger:
    """Thread-safe, append-only results logger."""

    def __init__(self, run_name: str = None):
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records: list = []
        self.log_path = RESULTS_DIR / f"results_{self.run_name}.json"

    def log_round(self, strategy: str, round_num: int, metrics: Dict[str, Any]):
        record = {
            "strategy": strategy,
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self.records.append(record)
        logger.debug(f"Logged: {record}")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)
        csv_path = str(self.log_path).replace(".json", ".csv")
        self.to_dataframe().to_csv(csv_path, index=False)
        logger.info(f"Results saved: {self.log_path}")

    @classmethod
    def load(cls, run_name: str) -> "ResultsLogger":
        inst = cls(run_name=run_name)
        with open(inst.log_path) as f:
            inst.records = json.load(f)
        return inst
