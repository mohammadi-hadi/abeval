"""Reading eval results and pairing two runs on a shared item id."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Blank lines are skipped."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def extract(records: list[dict], metric: str, id_key: str = "id") -> dict:
    """Map item id -> metric value. Duplicate ids and missing fields are errors."""
    out: dict = {}
    for rec in records:
        if id_key not in rec:
            raise ValueError(f"record missing id key {id_key!r}: {rec}")
        if metric not in rec:
            raise ValueError(f"record {rec[id_key]!r} missing metric {metric!r}")
        item_id = rec[id_key]
        if item_id in out:
            raise ValueError(f"duplicate id {item_id!r}")
        value = rec[metric]
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, (int, float)):
            raise ValueError(f"metric {metric!r} for id {item_id!r} is not numeric: {value!r}")
        out[item_id] = float(value)
    return out


def pair(
    a: dict, b: dict
) -> tuple[list[float], list[float], int]:
    """Inner-join two id->value maps.

    Returns (a_values, b_values, n_dropped) where n_dropped counts ids present
    in only one of the two runs. Order follows a's insertion order.
    """
    shared = [item_id for item_id in a if item_id in b]
    dropped = (len(a) - len(shared)) + (len(b) - len(shared))
    return [a[i] for i in shared], [b[i] for i in shared], dropped
