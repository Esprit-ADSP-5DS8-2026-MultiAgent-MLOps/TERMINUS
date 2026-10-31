"""Journal des tentatives en JSONL : futur dataset de fine-tuning."""
from __future__ import annotations

import json
import time
from pathlib import Path


def log_step(workspace: Path, **event) -> None:
    event["ts"] = time.time()
    with (workspace / "traces.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
