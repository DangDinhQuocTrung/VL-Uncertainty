"""Plot uncertainty metrics by visual certainty (H_vis).

Reads experiment logs produced with --compute_visual_statistics true.

Semantic entropy (VSE Sec. 3.2 / Fig. 3-Left):
  - Split samples by H_vis (bottom/top percentile)
  - Plot semantic entropy for visually confident vs uncertain groups

EUQ:
  - Split by visual-token mean head conflict / ignorance
  - Plot those EUQ scores, or semantic entropy if --reference_log is an SE log

VAUQ:
  - Logs both H_vis and EUQ visual scores (on the VAUQ-masked input)
  - Emits three figures: se, euq_conflict, euq_ignorance

Also reports detection AUROC on the full valid test set using the same visual
scores used for grouping (plots themselves may use incorrect answers only):
  - SE:  visual_entropy
  - EUQ: visual_mean_head_conflict_value / visual_mean_head_ignorance_value
  - VAUQ: all three of the above

Example:
  python plotting/plot_visual_certainty_se.py \\
      --log exp/log_euq.json \\
      --reference_log exp/log_semantic_entropy.json \\
      --percentile 0.2 \\
      --incorrect_only true
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional import auroc

from utils.constants import (
    DETECTION_AUROC_KEYS,
    LOG_META_KEYS,
    PERTURBATION_DETECTION_DATASETS,
)


EUQ_AUROC_SPECS = (
    ("visual_mean_head_conflict_value", "Visual Mean Head Conflict"),
    ("visual_mean_head_ignorance_value", "Visual Mean Head Ignorance"),
)
SE_AUROC_SPECS = (("visual_entropy", r"Visual Entropy ($H_{vis}$)"),)
VAUQ_AUROC_SPECS = SE_AUROC_SPECS + EUQ_AUROC_SPECS


def load_log(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r") as f:
        return json.load(f)


def print_run_metrics(log: Dict[str, Any]) -> None:
    """Print base task accuracy and detection AUROC stored in the log."""
    base_acc = log.get("Base task Accuracy")
    if base_acc is not None:
        print(f"Base task Accuracy: {float(base_acc):.4f}")
    else:
        print("Base task Accuracy: (not found in log)")

    printed_auroc = False
    for key in DETECTION_AUROC_KEYS:
        if key in log:
            print(f"{key}: {float(log[key]):.4f}")
            printed_auroc = True
    if not printed_auroc:
        print("Detection AUROC: (not found in log)")


def detect_method(log: Dict[str, Any]) -> str:
    method = str(log.get("uncertainty_method", "")).lower()
    if "semantic_entropy" in method:
        return "semantic_entropy"
    if method == "euq":
        return "euq"
    if method == "vauq":
        return "vauq"
    raise ValueError(
        f"Unsupported uncertainty_method '{log.get('uncertainty_method')}'. "
        "Expected semantic_entropy, euq, or vauq."
    )


def _is_sample_entry(key: str, value: Any, required_keys: Tuple[str, ...]) -> bool:
    if key in LOG_META_KEYS or not isinstance(value, dict):
        return False
    if "Hallucination detection" in key or "Perturbation detection" in key:
        return False
    return all(k in value for k in required_keys)


def load_samples(log: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
    if method == "semantic_entropy":
        required = ("visual_entropy", "uncertainty")
    elif method == "euq":
        required = (
            "visual_mean_head_conflict_value",
            "visual_mean_head_ignorance_value",
        )
    elif method == "vauq":
        required = (
            "visual_entropy",
            "visual_mean_head_conflict_value",
            "visual_mean_head_ignorance_value",
            "uncertainty",
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    samples = []
    for key, value in log.items():
        if not _is_sample_entry(key, value, required):
            continue
        if not value.get("flag_sample_valid", True):
            continue
        sample = {
            "idx": int(key) if str(key).isdigit() else key,
            "flag_answer_correct": bool(value.get("flag_answer_correct", True)),
            "flag_perturbed_inputs": value.get("flag_perturbed_inputs"),
        }
        if method in ("semantic_entropy", "vauq"):
            sample["visual_entropy"] = float(value["visual_entropy"])
            sample["uncertainty"] = float(value["uncertainty"])
        if method in ("euq", "vauq"):
            sample["visual_mean_head_conflict_value"] = float(
                value["visual_mean_head_conflict_value"]
            )
            sample["visual_mean_head_ignorance_value"] = float(
                value["visual_mean_head_ignorance_value"]
            )
        samples.append(sample)

    if not samples:
        raise ValueError(
            f"No samples with required keys {required} found in log. "
            "Re-run with --compute_visual_statistics true."
        )
    return samples


def load_uncertainty_by_idx(log: Dict[str, Any]) -> Dict[Any, float]:
    """Map sample idx -> uncertainty from a (typically semantic entropy) log."""
    uncertainties = {}
    for key, value in log.items():
        if not _is_sample_entry(key, value, ("uncertainty",)):
            continue
        if not value.get("flag_sample_valid", True):
            continue
        idx = int(key) if str(key).isdigit() else key
        uncertainties[idx] = float(value["uncertainty"])
    if not uncertainties:
        raise ValueError(
            "No samples with 'uncertainty' found in --reference_log."
        )
    return uncertainties


def attach_reference_uncertainty(
    samples: List[Dict[str, Any]],
    reference_uncertainties: Dict[Any, float],
) -> List[Dict[str, Any]]:
    """Overwrite plotted uncertainty with values from the reference log."""
    merged = []
    missing = 0
    for sample in samples:
        if sample["idx"] not in reference_uncertainties:
            missing += 1
            continue
        sample = dict(sample)
        sample["uncertainty"] = reference_uncertainties[sample["idx"]]
        merged.append(sample)
    if missing:
        print(
            f"Dropped {missing} samples with no matching idx in --reference_log."
        )
    if not merged:
        raise ValueError(
            "No overlapping sample idx between --log and --reference_log."
        )
    print(
        f"Using reference uncertainty for {len(merged)} overlapping samples."
    )
    return merged


def infer_detection_mode(log: Dict[str, Any]) -> Tuple[str, bool]:
    """Return (detection_label, is_perturbation_detection)."""
    dataset = log.get("dataset_name")
    if dataset is None:
        args_str = str(log.get("args", ""))
        for name in PERTURBATION_DETECTION_DATASETS:
            if f"benchmark='{name}'" in args_str or f'benchmark="{name}"' in args_str:
                dataset = name
                break
    is_perturbation = dataset in PERTURBATION_DETECTION_DATASETS
    label = "Perturbation detection" if is_perturbation else "Hallucination detection"
    return label, is_perturbation


def auroc_specs_for_method(method: str) -> Tuple[Tuple[str, str], ...]:
    if method == "euq":
        return EUQ_AUROC_SPECS
    if method == "semantic_entropy":
        return SE_AUROC_SPECS
    if method == "vauq":
        return VAUQ_AUROC_SPECS
    raise ValueError(f"Unsupported method: {method}")


def compute_visual_statistics_auroc(
    samples: List[Dict[str, Any]],
    log: Dict[str, Any],
    method: str,
) -> None:
    """Print detection AUROC for visual statistics on the full valid sample set."""
    detection_label, is_perturbation = infer_detection_mode(log)
    auroc_specs = auroc_specs_for_method(method)

    if is_perturbation:
        missing = [s["idx"] for s in samples if s.get("flag_perturbed_inputs") is None]
        if missing:
            raise ValueError(
                "Perturbation detection requires flag_perturbed_inputs, "
                f"missing on samples: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        detection_gt = torch.tensor(
            [bool(s["flag_perturbed_inputs"]) for s in samples], dtype=torch.int
        )
    else:
        # Higher visual score should rank incorrect answers higher.
        detection_gt = torch.tensor(
            [not s["flag_answer_correct"] for s in samples], dtype=torch.int
        )

    n_pos = int(detection_gt.sum().item())
    n_neg = int(len(samples) - n_pos)
    print(
        f"{detection_label} AUROC from visual statistics "
        f"(full valid set, n={len(samples)}, pos={n_pos}, neg={n_neg}):"
    )
    logged = log.get(f"{detection_label} AUROC")
    if logged is not None:
        print(f"  logged uncertainty AUROC: {float(logged):.4f}")

    if detection_gt.unique().numel() < 2:
        print("  (undefined; need both positive and negative labels)")
        return

    for field, label in auroc_specs:
        if any(field not in s for s in samples):
            print(f"  {label} ({field}): (missing in log)")
            continue
        scores = torch.tensor([s[field] for s in samples], dtype=torch.float)
        score = float(auroc(scores, detection_gt, task="binary").item())
        print(f"  {label} ({field}): {score:.4f}")


def split_by_visual_certainty(
    samples: List[Dict[str, Any]],
    value_key: str,
    split_key: str,
    percentile: float = 0.2,
    incorrect_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if incorrect_only:
        samples = [s for s in samples if not s["flag_answer_correct"]]
    if len(samples) < 2:
        raise ValueError(
            f"Need at least 2 samples after filtering (got {len(samples)}). "
            "Try --incorrect_only false or a larger log."
        )

    split_values = np.asarray([s[split_key] for s in samples], dtype=np.float64)
    values = np.asarray([s[value_key] for s in samples], dtype=np.float64)

    lo = float(np.quantile(split_values, percentile))
    hi = float(np.quantile(split_values, 1.0 - percentile))

    confident_mask = split_values <= lo
    uncertain_mask = split_values >= hi

    if confident_mask.sum() == 0 or uncertain_mask.sum() == 0:
        order = np.argsort(split_values)
        k = max(1, int(round(len(samples) * percentile)))
        confident_idx = order[:k]
        uncertain_idx = order[-k:]
        values_conf = values[confident_idx]
        values_unc = values[uncertain_idx]
        lo = float(split_values[confident_idx].max())
        hi = float(split_values[uncertain_idx].min())
    else:
        values_conf = values[confident_mask]
        values_unc = values[uncertain_mask]

    stats = {
        "n_total": float(len(samples)),
        "n_confident": float(len(values_conf)),
        "n_uncertain": float(len(values_unc)),
        "split_low_threshold": lo,
        "split_high_threshold": hi,
        "confident_mean": float(values_conf.mean()) if len(values_conf) else float("nan"),
        "uncertain_mean": float(values_unc.mean()) if len(values_unc) else float("nan"),
    }
    return values_conf, values_unc, stats


def _plot_one_hist(ax, values: np.ndarray, title: str, color: str, xlabel: str):
    if len(values) == 0:
        ax.set_title(f"{title}\n(n=0)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        return
    ax.hist(values, bins="auto", color=color, edgecolor="black", alpha=0.85)
    ax.axvline(
        values.mean(),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={values.mean():.3f}",
    )
    ax.set_title(f"{title}\n(n={len(values)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(frameon=False)


def plot_distributions(
    values_confident: np.ndarray,
    values_uncertain: np.ndarray,
    stats: Dict[str, float],
    out_path: str,
    xlabel: str,
    group_labels: Tuple[str, str],
    title: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _plot_one_hist(
        axes[0],
        values_confident,
        group_labels[0],
        color="#4C78A8",
        xlabel=xlabel,
    )
    _plot_one_hist(
        axes[1],
        values_uncertain,
        group_labels[1],
        color="#F58518",
        xlabel=xlabel,
    )

    subtitle = (
        f"thresholds: split≤{stats['split_low_threshold']:.3f} / "
        f"split≥{stats['split_high_threshold']:.3f}  |  "
        f"N={int(stats['n_total'])}"
    )
    fig.suptitle(title or f"{xlabel} by Visual Certainty", fontsize=13)
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def _default_out_path(method: str, metric: str, out: Optional[str]) -> str:
    if out is not None:
        if method == "semantic_entropy":
            return out
        stem, ext = os.path.splitext(out)
        ext = ext or ".png"
        return f"{stem}_{metric}{ext}"
    if method == "semantic_entropy":
        return "exp/fig_visual_certainty_se.png"
    if method == "vauq":
        return f"exp/fig_visual_certainty_vauq_{metric}.png"
    return f"exp/fig_visual_certainty_euq_{metric}.png"


def _plot_se_style(
    samples: List[Dict[str, Any]],
    args,
    subset: str,
    method: str,
    value_xlabel: Optional[str],
    title_override: Optional[str] = None,
):
    values_conf, values_unc, stats = split_by_visual_certainty(
        samples,
        value_key="uncertainty",
        split_key="visual_entropy",
        percentile=args.percentile,
        incorrect_only=args.incorrect_only,
    )
    print("Split stats (visual_entropy):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    title = title_override or (
        args.title
        or f"{'VAUQ Uncertainty' if method == 'vauq' else 'Semantic Entropy'} "
        f"by Visual Certainty ({subset})"
    )
    out_path = _default_out_path(method, "se", args.out)
    plot_distributions(
        values_conf,
        values_unc,
        stats,
        out_path,
        xlabel=value_xlabel or (
            "VAUQ Uncertainty" if method == "vauq" else "Semantic Entropy"
        ),
        group_labels=(
            "Visually Confident (low $H_{vis}$)",
            "Visually Uncertain (high $H_{vis}$)",
        ),
        title=title,
    )


def _plot_euq_style(
    samples: List[Dict[str, Any]],
    args,
    subset: str,
    method: str,
    plot_uncertainty: bool,
    suffix_prefix: str = "",
):
    for metric_key, metric_label in (
        ("visual_mean_head_conflict_value", "Visual Mean Head Conflict"),
        ("visual_mean_head_ignorance_value", "Visual Mean Head Ignorance"),
    ):
        value_key = "uncertainty" if plot_uncertainty else metric_key
        values_conf, values_unc, stats = split_by_visual_certainty(
            samples,
            value_key=value_key,
            split_key=metric_key,
            percentile=args.percentile,
            incorrect_only=args.incorrect_only,
        )
        print(f"Split stats ({metric_key}):")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        base = "conflict" if "conflict" in metric_key else "ignorance"
        suffix = f"{suffix_prefix}{base}" if suffix_prefix else base
        out_path = _default_out_path(method, suffix, args.out)
        if plot_uncertainty:
            xlabel = "Semantic Entropy"
            title = f"Semantic Entropy by {metric_label} ({subset})"
        else:
            xlabel = metric_label
            title = f"{metric_label} by Visual Certainty ({subset})"
        plot_distributions(
            values_conf,
            values_unc,
            stats,
            out_path,
            xlabel=xlabel,
            group_labels=(
                f"Low {metric_label}",
                f"High {metric_label}",
            ),
            title=title,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot uncertainty metrics by visual certainty (H_vis)."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to exp/log_*.json produced with --compute_visual_statistics true.",
    )
    parser.add_argument(
        "--reference_log",
        type=str,
        default=None,
        help=(
            "Log whose 'uncertainty' values are plotted. Defaults to --log. "
            "Pass a semantic_entropy log to compare EUQ visual scores against SE."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.2,
        help="Fraction used for bottom/top H_vis split (paper default: 0.2).",
    )
    parser.add_argument(
        "--incorrect_only",
        type=lambda x: x.lower() == "true",
        default="True",
        help="If true, keep only incorrect answers (paper Sec. 3.2). Default: True.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output figure path. For euq/vauq, metric plots use stem suffixes "
            "(se / conflict|euq_conflict / ignorance|euq_ignorance)."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title override (semantic_entropy / vauq se plot only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log = load_log(args.log)
    print("Primary log:")
    print_run_metrics(log)

    reference_path = args.reference_log or args.log
    use_reference = os.path.abspath(reference_path) != os.path.abspath(args.log)
    if use_reference:
        reference_log = load_log(reference_path)
        print(f"Reference log ({reference_path}):")
        print_run_metrics(reference_log)
    else:
        reference_log = log

    method = detect_method(log)
    print(f"Detected method: {method}")
    samples = load_samples(log, method)
    compute_visual_statistics_auroc(samples, log, method)
    if use_reference:
        samples = attach_reference_uncertainty(
            samples, load_uncertainty_by_idx(reference_log)
        )

    subset = "incorrect answers" if args.incorrect_only else "all answers"
    plot_uncertainty = use_reference
    value_xlabel = "Semantic Entropy" if plot_uncertainty else None

    if method == "semantic_entropy":
        _plot_se_style(samples, args, subset, method, value_xlabel)
        return

    if method == "vauq":
        # Three figures: se, euq_conflict, euq_ignorance.
        _plot_se_style(samples, args, subset, method, value_xlabel)
        if plot_uncertainty:
            _plot_euq_style(
                samples,
                args,
                subset,
                method,
                plot_uncertainty=True,
                suffix_prefix="euq_",
            )
        else:
            for metric_key, metric_label in (
                ("visual_mean_head_conflict_value", "Visual Mean Head Conflict"),
                ("visual_mean_head_ignorance_value", "Visual Mean Head Ignorance"),
            ):
                values_conf, values_unc, stats = split_by_visual_certainty(
                    samples,
                    value_key="uncertainty",
                    split_key=metric_key,
                    percentile=args.percentile,
                    incorrect_only=args.incorrect_only,
                )
                print(f"Split stats ({metric_key}):")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                suffix = (
                    "euq_conflict" if "conflict" in metric_key else "euq_ignorance"
                )
                out_path = _default_out_path(method, suffix, args.out)
                plot_distributions(
                    values_conf,
                    values_unc,
                    stats,
                    out_path,
                    xlabel="VAUQ Uncertainty",
                    group_labels=(
                        f"Low {metric_label}",
                        f"High {metric_label}",
                    ),
                    title=f"VAUQ Uncertainty by {metric_label} ({subset})",
                )
        return

    _plot_euq_style(samples, args, subset, method, plot_uncertainty)


if __name__ == "__main__":
    main()
