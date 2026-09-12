#!/usr/bin/env python3
"""Rename experiment logs from log_*.json to {benchmark}_{method}_*.json.

Replaces the leading \"log\" prefix with \"{benchmark}_{uncertainty_method}\",
keeping the remainder of the filename unchanged. Metadata is read from each
JSON (dataset_name / uncertainty_method), with a fallback to parsing the
serialized args string.

Example:
  log_2026_09_07_23_53_59.json
    -> MedVIGIL_vse_masked_2026_09_07_23_53_59.json

Usage:
  python plotting/rename_logs.py exp_03
  python plotting/rename_logs.py exp_03 --dry_run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

LOG_PREFIX_RE = re.compile(r"^log(?=_)")
BENCHMARK_IN_ARGS_RE = re.compile(r"benchmark=['\"]([^'\"]+)['\"]")
UNCERTAINTY_IN_ARGS_RE = re.compile(r"uncertainty=['\"]([^'\"]+)['\"]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename log_*.json files in an experiment directory to "
            "{benchmark}_{uncertainty_method}_*.json."
        )
    )
    parser.add_argument(
        "exp_dir",
        type=str,
        help="Experiment directory containing log_*.json files (e.g. exp, exp_00).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned renames without modifying files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also rename log_*.json files in subdirectories.",
    )
    return parser.parse_args()


def load_log(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _from_args(log: Dict[str, Any], pattern: re.Pattern[str]) -> Optional[str]:
    args_str = str(log.get("args", ""))
    match = pattern.search(args_str)
    return match.group(1) if match else None


def extract_benchmark_and_method(log: Dict[str, Any]) -> Tuple[str, str]:
    benchmark = log.get("dataset_name") or _from_args(log, BENCHMARK_IN_ARGS_RE)
    method = log.get("uncertainty_method") or _from_args(log, UNCERTAINTY_IN_ARGS_RE)
    if not benchmark or not method:
        missing = []
        if not benchmark:
            missing.append("benchmark/dataset_name")
        if not method:
            missing.append("uncertainty_method")
        raise ValueError(f"missing {', '.join(missing)}")
    return str(benchmark), str(method)


def sanitize_name_part(value: str) -> str:
    """Make a string safe for use in a filename fragment."""
    cleaned = re.sub(r"[^\w.\-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(f"empty name part after sanitizing {value!r}")
    return cleaned


def build_new_name(old_name: str, benchmark: str, method: str) -> str:
    if not LOG_PREFIX_RE.match(old_name):
        raise ValueError(f"filename does not start with 'log_': {old_name}")
    prefix = f"{sanitize_name_part(benchmark)}_{sanitize_name_part(method)}"
    return LOG_PREFIX_RE.sub(prefix, old_name, count=1)


def iter_log_files(exp_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/" if recursive else ""
    return sorted(p for p in exp_dir.glob(f"{pattern}log_*.json") if p.is_file())


def rename_logs(exp_dir: Path, dry_run: bool = False, recursive: bool = False) -> int:
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"experiment directory not found: {exp_dir}")

    log_files = iter_log_files(exp_dir, recursive=recursive)
    if not log_files:
        print(f"No log_*.json files found in {exp_dir}")
        return 0

    renamed = 0
    skipped = 0
    failed = 0

    for path in log_files:
        try:
            log = load_log(path)
            benchmark, method = extract_benchmark_and_method(log)
            new_name = build_new_name(path.name, benchmark, method)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            failed += 1
            continue

        dest = path.with_name(new_name)
        if dest == path:
            print(f"[keep] {path.name}")
            skipped += 1
            continue
        if dest.exists():
            print(f"[skip] {path.name} -> {new_name} (target exists)")
            skipped += 1
            continue

        action = "would rename" if dry_run else "rename"
        print(f"[{action}] {path.name} -> {new_name}")
        if not dry_run:
            path.rename(dest)
        renamed += 1

    print(
        f"Done. renamed={renamed}, skipped={skipped}, failed={failed} "
        f"(dry_run={dry_run})"
    )
    return failed


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    failed = rename_logs(exp_dir, dry_run=args.dry_run, recursive=args.recursive)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
