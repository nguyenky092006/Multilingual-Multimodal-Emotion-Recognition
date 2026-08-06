#!/usr/bin/env python
"""Aggregate completed JSON metrics without inventing missing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    records = []
    for path in args.paths:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        for name in ("uar", "macro_f1", "accuracy"):
            if name not in record:
                raise ValueError(f"{path} has no {name}")
        records.append(record)
    result = {"runs": len(records), "synthetic": all(item.get("synthetic") is True for item in records)}
    for name in ("uar", "macro_f1", "accuracy"):
        values = np.asarray([item[name] for item in records], dtype=float)
        result[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
