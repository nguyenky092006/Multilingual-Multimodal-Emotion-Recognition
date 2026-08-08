#!/usr/bin/env python
"""Train/evaluate a config across fixed seeds and aggregate completed records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from mmer.config import load_yaml
from mmer.experiments import aggregate_seed_metrics, seed_config


def _inside_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes project root: {value}")
    return candidate


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 41])
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/seed_sweeps"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = _inside_root(root, args.config)
    base = load_yaml(config_path)
    sweep_dir = _inside_root(root, args.output_dir / config_path.stem)
    config_dir = sweep_dir / "resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    sources: list[str] = []
    for seed in args.seeds:
        resolved = seed_config(base, seed)
        generated_path = config_dir / f"seed{seed}.yaml"
        generated_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        metrics_path = _inside_root(root, Path(str(resolved["output_dir"])) / "evaluation_metrics.json")
        if not (args.skip_completed and metrics_path.is_file()):
            common = ["--config", str(generated_path), "--project-root", str(root)]
            _run(
                [sys.executable, str(root / "scripts" / "train.py"), *common],
                sweep_dir / f"seed{seed}_train.log",
            )
            _run(
                [sys.executable, str(root / "scripts" / "evaluate.py"), *common],
                sweep_dir / f"seed{seed}_evaluate.log",
            )
        if not metrics_path.is_file():
            raise FileNotFoundError(f"evaluation did not create {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        records.append(record)
        sources.append(metrics_path.relative_to(root).as_posix())

    summary = aggregate_seed_metrics(records, args.seeds)
    summary["sources"] = sources
    summary["base_config"] = config_path.relative_to(root).as_posix()
    summary_path = sweep_dir / "seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "summary_path": summary_path.relative_to(root).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
