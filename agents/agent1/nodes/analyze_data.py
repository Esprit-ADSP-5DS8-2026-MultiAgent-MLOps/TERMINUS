"""Profile the input CSV without changing it."""

from typing import Any

import pandas as pd

from ..state import Agent1State


def analyze_data(state: Agent1State) -> dict[str, Any]:
    """Collect dtypes, missing counts, and unique counts for each CSV column."""
    try:
        frame = pd.read_csv(state["csv_path"])
        columns: dict[str, dict[str, Any]] = {}
        for name in frame.columns:
            series = frame[name]
            columns[str(name)] = {
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "boolean": bool(pd.api.types.is_bool_dtype(series)),
                "value_counts": {
                    str(value): int(count)
                    for value, count in series.value_counts(dropna=False).items()
                },
            }
        return {
            "profile": {
                "rows": int(len(frame)),
                "columns": columns,
            }
        }
    except Exception as exc:
        return {
            "profile": {},
            "errors": [{"node": "analyze_data", "message": str(exc)}],
        }
