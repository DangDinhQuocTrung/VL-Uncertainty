"""Compare incorrect hallucination-detection samples between two logs.

Only samples where both logs agree on answer correctness are considered.
Then report overlap / IoU of samples where detection was wrong
(``flag_detection_correct`` is False).

Example:
  PYTHONPATH=. ~/venv/health/bin/python plotting/compare_detection_errors.py \\
      --log_a exp_02/more_plots_visual_certainty/ViLP_semantic_entropy_2026_09_12_20_26_35_revised.json \\
      --log_b exp_02/ViLP_vauq_2026_08_25_19_44_39_revised.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from utils.constants import LOG_META_KEYS


def load_log(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r") as f:
        return json.load(f)


def _is_sample_entry(key: str, value: Any) -> bool:
    if key in LOG_META_KEYS or not isinstance(value, dict):
        return False
    if "Hallucination detection" in key or "Perturbation detection" in key:
        return False
    return "flag_answer_correct" in value and "flag_detection_correct" in value


def collect_comparable_error_sets(
    log_a: Dict[str, Any], log_b: Dict[str, Any]
) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    """Return (error_ids_a, error_ids_b, stats) on answer-correctness-aligned samples."""
    keys_a = {k for k, v in log_a.items() if _is_sample_entry(k, v)}
    keys_b = {k for k, v in log_b.items() if _is_sample_entry(k, v)}
    shared = keys_a & keys_b

    comparable: Set[str] = set()
    errors_a: Set[str] = set()
    errors_b: Set[str] = set()
    n_answer_disagree = 0
    n_missing_valid = 0

    for idx in shared:
        sa, sb = log_a[idx], log_b[idx]
        if not sa.get("flag_sample_valid", True) or not sb.get("flag_sample_valid", True):
            n_missing_valid += 1
            continue
        if bool(sa["flag_answer_correct"]) != bool(sb["flag_answer_correct"]):
            n_answer_disagree += 1
            continue

        comparable.add(idx)
        if not bool(sa["flag_detection_correct"]):
            errors_a.add(idx)
        if not bool(sb["flag_detection_correct"]):
            errors_b.add(idx)

    stats = {
        "n_shared_keys": len(shared),
        "n_only_a": len(keys_a - keys_b),
        "n_only_b": len(keys_b - keys_a),
        "n_skipped_invalid": n_missing_valid,
        "n_answer_disagree": n_answer_disagree,
        "n_comparable": len(comparable),
        "n_errors_a": len(errors_a),
        "n_errors_b": len(errors_b),
    }
    return errors_a, errors_b, stats


def set_iou(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 1.0 if not a and not b else 0.0
    return len(a & b) / len(union)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlap / IoU of wrong hallucination detections between two logs."
    )
    parser.add_argument("--log_a", type=str, required=True)
    parser.add_argument("--log_b", type=str, required=True)
    parser.add_argument(
        "--label_a",
        type=str,
        default=None,
        help="Short name for log A (default: parent/stem).",
    )
    parser.add_argument(
        "--label_b",
        type=str,
        default=None,
        help="Short name for log B (default: parent/stem).",
    )
    parser.add_argument(
        "--print_ids",
        action="store_true",
        help="Print sample ids in intersection / A-only / B-only.",
    )
    args = parser.parse_args()

    path_a = Path(args.log_a)
    path_b = Path(args.log_b)
    label_a = args.label_a or f"{path_a.parent.name}/{path_a.stem}"
    label_b = args.label_b or f"{path_b.parent.name}/{path_b.stem}"

    log_a = load_log(str(path_a))
    log_b = load_log(str(path_b))
    errors_a, errors_b, stats = collect_comparable_error_sets(log_a, log_b)

    inter = errors_a & errors_b
    only_a = errors_a - errors_b
    only_b = errors_b - errors_a
    iou = set_iou(errors_a, errors_b)
    # Overlap as Jaccard is IoU; also report recall-style overlaps.
    overlap_vs_a = (len(inter) / len(errors_a)) if errors_a else float("nan")
    overlap_vs_b = (len(inter) / len(errors_b)) if errors_b else float("nan")

    print(f"Log A: {path_a}")
    print(f"Log B: {path_b}")
    print(f"Label A: {label_a}")
    print(f"Label B: {label_b}")
    print()
    print("--- Filtering ---")
    print(f"Shared sample keys:           {stats['n_shared_keys']}")
    print(f"Only in A / only in B:        {stats['n_only_a']} / {stats['n_only_b']}")
    print(f"Skipped (invalid sample):     {stats['n_skipped_invalid']}")
    print(f"Skipped (answer disagree):    {stats['n_answer_disagree']}")
    print(f"Comparable (answer agree):    {stats['n_comparable']}")
    print()
    print("--- Wrong hallucination detections (on comparable set) ---")
    print(f"|errors A| ({label_a}): {len(errors_a)}")
    print(f"|errors B| ({label_b}): {len(errors_b)}")
    print(f"|A ∩ B|:                    {len(inter)}")
    print(f"|A − B|:                    {len(only_a)}")
    print(f"|B − A|:                    {len(only_b)}")
    print(f"|A ∪ B|:                    {len(errors_a | errors_b)}")
    print()
    print("--- Overlap / IoU ---")
    print(f"IoU  |A ∩ B| / |A ∪ B|:     {iou:.4f}")
    print(f"Overlap vs A |A ∩ B| / |A|: {overlap_vs_a:.4f}")
    print(f"Overlap vs B |A ∩ B| / |B|: {overlap_vs_b:.4f}")

    if args.print_ids:
        def _sorted_ids(ids: Set[str]):
            return sorted(ids, key=lambda x: int(x) if str(x).isdigit() else str(x))

        print()
        print("--- Sample ids ---")
        print("Intersection:", _sorted_ids(inter))
        print("A only:", _sorted_ids(only_a))
        print("B only:", _sorted_ids(only_b))


if __name__ == "__main__":
    main()
