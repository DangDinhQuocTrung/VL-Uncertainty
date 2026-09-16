#!/usr/bin/env python3
"""Plot uncertainty metrics by gender and age group from a HAM10000 log.

Modes:
  uncertainty_score   – mean uncertainty per demographic group (default)
  misclassify_auroc   – misclassification AUROC per group
                        (same definition as main.py: AUROC of uncertainty
                         vs. 1 - flag_answer_correct)
  base_acc            – VQA accuracy (mean of flag_answer_correct) per group
  base_acc_selective  – VQA accuracy after dropping the 20% highest-uncertainty
                        samples; scope controlled by --selective_scope:
                          per_group (default): drop within each demographic group
                          total: drop globally, then score per group

Example:
  python plotting/plot_uncertainty_by_demographic.py \\
      --log exp/log_2026_09_09_00_24_51.json \\
      --mode uncertainty_score

  python plotting/plot_uncertainty_by_demographic.py \\
      --log exp/log_2026_09_09_00_24_51.json \\
      --mode misclassify_auroc

  python plotting/plot_uncertainty_by_demographic.py \\
      --log exp/log_2026_09_09_00_24_51.json \\
      --mode base_acc

  python plotting/plot_uncertainty_by_demographic.py \\
      --log exp/log_2026_09_09_00_24_51.json \\
      --mode base_acc_selective \\
      --selective_scope per_group

  python plotting/plot_uncertainty_by_demographic.py \\
      --log exp/log_2026_09_09_00_24_51.json \\
      --mode base_acc_selective \\
      --selective_scope total
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


GENDER_ORDER = ("female", "male")
AGE_BINS: List[Tuple[str, float, float]] = [
    ("0-30", 0, 30),
    ("30-50", 30, 50),
    ("50-70", 50, 70),
    ("70-90", 70, 90),
]
SELECTIVE_REJECT_FRACTION = 0.2

MetricFn = Callable[[List[Dict[str, Any]]], float]
PreprocessFn = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a HAM10000 experiment log and plot a demographic metric "
            "by gender and age group."
        )
    )
    parser.add_argument(
        "--log",
        type=str,
        default="exp/log_2026_09_09_00_24_51.json",
        help="Path to a log_*.json file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="uncertainty_score",
        choices=[
            "uncertainty_score",
            "misclassify_auroc",
            "base_acc",
            "base_acc_selective",
        ],
        help=(
            "uncertainty_score: mean uncertainty per group; "
            "misclassify_auroc: misclassification AUROC per group; "
            "base_acc: VQA accuracy per group; "
            "base_acc_selective: VQA accuracy after dropping the 20% "
            "highest-uncertainty samples (see --selective_scope)."
        ),
    )
    parser.add_argument(
        "--selective_scope",
        type=str,
        default="total",
        choices=["per_group", "total"],
        help=(
            "Only used with --mode base_acc_selective. "
            "per_group: drop the top 20% uncertainty within each group; "
            "total: drop the top 20% uncertainty over all samples, then "
            "compute accuracy per group."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output image path. Defaults to "
            "<log_dir>/<log_stem>_<mode>_by_demographic.png"
        ),
    )
    parser.add_argument(
        "--show",
        type=lambda x: x.lower() == "true",
        default="False",
        help="If true, also open an interactive plot window.",
    )
    return parser.parse_args()


def load_samples(log_path: str) -> List[Dict[str, Any]]:
    with open(log_path, "r") as f:
        log_dict = json.load(f)

    samples = []
    for key, value in log_dict.items():
        if not str(key).isdigit() or not isinstance(value, dict):
            continue
        if value.get("flag_sample_valid") is False:
            continue
        age = value.get("age")
        gender = value.get("gender")
        uncertainty = value.get("uncertainty")
        flag_answer_correct = value.get("flag_answer_correct")
        if age is None or gender is None or uncertainty is None:
            continue
        if flag_answer_correct is None:
            continue
        samples.append(
            {
                "age": float(age),
                "gender": str(gender).lower(),
                "uncertainty": float(uncertainty),
                "flag_answer_correct": bool(flag_answer_correct),
            }
        )
    if not samples:
        raise ValueError(
            f"No valid samples with age/gender/uncertainty/"
            f"flag_answer_correct in {log_path}"
        )
    return samples


def age_group_label(age: float) -> Optional[str]:
    """Map age to a bin label. Intervals are [lo, hi), except the last which is [lo, hi]."""
    for i, (label, lo, hi) in enumerate(AGE_BINS):
        if i < len(AGE_BINS) - 1:
            if lo <= age < hi:
                return label
        else:
            if lo <= age <= hi:
                return label
    return None


def gender_key(sample: Dict[str, Any]) -> Optional[str]:
    gender = sample["gender"]
    return gender if gender in GENDER_ORDER else None


def age_key(sample: Dict[str, Any]) -> Optional[str]:
    return age_group_label(sample["age"])


def mean_uncertainty(group_samples: List[Dict[str, Any]]) -> float:
    if not group_samples:
        return float("nan")
    return float(np.mean([s["uncertainty"] for s in group_samples]))


def base_accuracy(group_samples: List[Dict[str, Any]]) -> float:
    """VQA accuracy: fraction of samples with flag_answer_correct."""
    if not group_samples:
        return float("nan")
    return float(np.mean([s["flag_answer_correct"] for s in group_samples]))


def drop_highest_uncertainty(
    group_samples: List[Dict[str, Any]],
    reject_fraction: float = SELECTIVE_REJECT_FRACTION,
) -> List[Dict[str, Any]]:
    """Keep the lowest-uncertainty samples; drop the top reject_fraction."""
    n = len(group_samples)
    if n == 0:
        return []
    n_drop = int(n * reject_fraction)
    n_keep = n - n_drop
    if n_keep <= 0:
        return []
    ordered = sorted(group_samples, key=lambda s: s["uncertainty"])
    return ordered[:n_keep]


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUROC (higher score → positive), matching torchmetrics/main.py."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank scores ascending; average ranks for ties (Mann–Whitney / ROC).
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Ranks are 1-based.
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[labels == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def misclassify_auroc(group_samples: List[Dict[str, Any]]) -> float:
    """AUROC of uncertainty vs. incorrectness, matching main.py."""
    if len(group_samples) < 2:
        return float("nan")
    uncertainty_scores = np.array(
        [s["uncertainty"] for s in group_samples], dtype=np.float64
    )
    # main.py: detection_gt = 1 - correctness_gt
    detection_gt = np.array(
        [0 if s["flag_answer_correct"] else 1 for s in group_samples],
        dtype=np.int64,
    )
    return binary_auroc(uncertainty_scores, detection_gt)


def metric_by_group(
    samples: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], Optional[str]],
    ordered_labels: List[str],
    metric_fn: MetricFn,
    preprocess_fn: Optional[PreprocessFn] = None,
) -> Tuple[List[str], List[float], List[int]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        label = key_fn(sample)
        if label is None:
            continue
        buckets[label].append(sample)

    values = []
    counts = []
    for label in ordered_labels:
        group = buckets.get(label, [])
        if preprocess_fn is not None:
            group = preprocess_fn(group)
        values.append(metric_fn(group))
        counts.append(len(group))
    return list(ordered_labels), values, counts


def group_fairness_stats(values: List[float]) -> Tuple[float, float]:
    """Return (max-min gap, variance) over finite group metric values."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan")
    gap = float(finite.max() - finite.min())
    var = float(np.var(finite))  # population variance over groups
    return gap, var


def plot_bars(
    ax: plt.Axes,
    labels: List[str],
    values: List[float],
    counts: List[int],
    title: str,
    ylabel: str,
    color: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, edgecolor="black", linewidth=0.6, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymax = np.nanmax(values) if len(values) else 1.0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        ax.set_ylim(0, ymax * 1.2)

    for bar, value, count in zip(bars, values, counts):
        if not np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                f"n/a\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    samples = load_samples(str(log_path))

    mode_config = {
        "uncertainty_score": {
            "metric_fn": mean_uncertainty,
            "preprocess_fn": None,
            "ylabel": "Mean uncertainty",
            "title_prefix": "Mean uncertainty",
            "fig_title": "Uncertainty by demographic",
            "ylim": None,
        },
        "misclassify_auroc": {
            "metric_fn": misclassify_auroc,
            "preprocess_fn": None,
            "ylabel": "Misclassification AUROC",
            "title_prefix": "Misclassification AUROC",
            "fig_title": "Misclassification AUROC by demographic",
            "ylim": (0.0, 1.05),
        },
        "base_acc": {
            "metric_fn": base_accuracy,
            "preprocess_fn": None,
            "ylabel": "VQA accuracy",
            "title_prefix": "VQA accuracy",
            "fig_title": "VQA accuracy by demographic",
            "ylim": (0.0, 1.05),
        },
        "base_acc_selective": {
            "metric_fn": base_accuracy,
            "ylabel": "Selective VQA accuracy",
            "fig_title": "Selective VQA accuracy by demographic",
            "ylim": (0.0, 1.05),
        },
    }
    cfg = mode_config[args.mode]
    metric_fn = cfg["metric_fn"]
    ylabel = cfg["ylabel"]
    fig_title = cfg["fig_title"]
    ylim = cfg["ylim"]

    # Selective rejection: either within each group, or once over all samples.
    samples_for_metric = samples
    preprocess_fn: Optional[PreprocessFn] = None
    if args.mode == "base_acc_selective":
        if args.selective_scope == "per_group":
            preprocess_fn = drop_highest_uncertainty
            title_prefix = "Selective VQA accuracy (20% per group)"
        else:
            samples_for_metric = drop_highest_uncertainty(samples)
            title_prefix = "Selective VQA accuracy (20% total)"
            print(
                f"Selective total: kept {len(samples_for_metric)}/"
                f"{len(samples)} samples after global drop"
            )
    else:
        preprocess_fn = cfg.get("preprocess_fn")
        title_prefix = cfg["title_prefix"]

    age_labels_order = [bin_label for bin_label, _, _ in AGE_BINS]
    gender_labels, gender_values, gender_counts = metric_by_group(
        samples_for_metric,
        gender_key,
        list(GENDER_ORDER),
        metric_fn,
        preprocess_fn,
    )
    age_labels, age_values, age_counts = metric_by_group(
        samples_for_metric,
        age_key,
        age_labels_order,
        metric_fn,
        preprocess_fn,
    )

    mode_tag = args.mode
    if args.mode == "base_acc_selective":
        mode_tag = f"{args.mode}_{args.selective_scope}"

    skipped_age = sum(1 for s in samples if age_key(s) is None)
    print(
        f"Loaded {len(samples)} samples from {log_path} "
        f"(mode={mode_tag})"
    )
    if skipped_age:
        print(
            f"Skipped {skipped_age} samples outside age bins "
            f"{[b[0] for b in AGE_BINS]}"
        )
    print("Gender:", dict(zip(gender_labels, zip(gender_values, gender_counts))))
    print("Age:", dict(zip(age_labels, zip(age_values, age_counts))))
    if args.mode in ("base_acc", "base_acc_selective"):
        gender_gap, gender_var = group_fairness_stats(gender_values)
        age_gap, age_var = group_fairness_stats(age_values)
        print(
            f"Gender fairness: max-min gap={gender_gap:.4f}, "
            f"variance={gender_var:.6f}"
        )
        print(
            f"Age fairness: max-min gap={age_gap:.4f}, "
            f"variance={age_var:.6f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    plot_bars(
        axes[0],
        gender_labels,
        gender_values,
        gender_counts,
        title=f"{title_prefix} by gender",
        ylabel=ylabel,
        color="#4C78A8",
        ylim=ylim,
    )
    plot_bars(
        axes[1],
        age_labels,
        age_values,
        age_counts,
        title=f"{title_prefix} by age group",
        ylabel=ylabel,
        color="#F58518",
        ylim=ylim,
    )
    fig.suptitle(f"{fig_title} ({log_path.name})", fontsize=11)

    output = (
        Path(args.output)
        if args.output
        else log_path.with_name(f"{log_path.stem}_{mode_tag}_by_demographic.png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
