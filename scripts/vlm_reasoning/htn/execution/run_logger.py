"""RunLogger: one JSON file per planning iteration plus summary.json.

(Named run_logger.py, not logging.py, so it cannot be confused with the
standard library's logging module.)"""

import json
import re
from datetime import datetime
from pathlib import Path


class RunLogger:
    """Writes results/htn_runs/<timestamp>_<task-slug>/iter_<n>.json, summary.json
    and connectivity.json."""

    def __init__(self, root, task_text):
        slug = re.sub(r"[^a-z0-9]+", "_", task_text.lower()).strip("_")[:40] or "task"
        self.dir = Path(root) / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def log_iteration(self, n, record):
        (self.dir / f"iter_{n:02d}.json").write_text(json.dumps(record, indent=2) + "\n")

    def write_summary(self, summary):
        self.write_json("summary.json", summary)

    def write_json(self, name, data):
        (self.dir / name).write_text(json.dumps(data, indent=2) + "\n")


class NullLogger:
    """Writes nothing (tests)."""
    dir = None

    def log_iteration(self, n, record):
        pass

    def write_summary(self, summary):
        pass

    def write_json(self, name, data):
        pass
