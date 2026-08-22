"""Plot semantic entropy distributions by visual certainty.

Reads an experiment log produced with:
  --uncertainty semantic_entropy_nli --compute_visual_entropy true

Reproduces the VSE paper (arXiv:2606.31407) Sec. 3.2 / Fig. 3-Left style analysis:
  - H_vis = average LogitLens entropy over visual tokens
  - Visually Confident  = bottom `percentile` of H_vis
  - Visually Uncertain  = top `percentile` of H_vis
  - Plot semantic entropy distributions for the two groups
    (optionally restricted to incorrect answers, as in the paper).

Example:
  python interpretability/plot_visual_certainty_se.py \\
      --log exp/log_YYYY_MM_DD_HH_MM_SS.json \\
      --percentile 0.2 \\
      --incorrect_only \\
      --out interpretability/fig_visual_certainty_se.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


META_KEYS = {
    "args",
    "begin_time_str",
    "end_time_str",
    "dataset_name",
    "uncertainty_method",
    "Total samples",
    "Base task Accuracy",
}

DETECTION_AUROC_KEYS = (
    "Hallucination detection AUROC",
    "Perturbation detection AUROC",
)


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


def _is_sample_entry(key: str, value: Any) -> bool:
    if key in META_KEYS or not isinstance(value, dict):
        return False
    if "Hallucination detection" in key or "Perturbation detection" in key:
        return False
    return "uncertainty" in value and "visual_entropy" in value


def load_samples(log: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = []
    for key, value in log.items():
        if not _is_sample_entry(key, value):
            continue
        if not value.get("flag_sample_valid", True):
            continue
        samples.append(
            {
                "idx": int(key) if str(key).isdigit() else key,
                "visual_entropy": float(value["visual_entropy"]),
                "uncertainty": float(value["uncertainty"]),
                "flag_answer_correct": bool(value.get("flag_answer_correct", True)),
            }
        )
    if not samples:
        raise ValueError(
            "No samples with both 'visual_entropy' and 'uncertainty' found in log. "
            "Re-run with --compute_visual_entropy true."
        )
    return samples


def split_by_visual_certainty(
    samples: List[Dict[str, Any]],
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

    h_vis = np.asarray([s["visual_entropy"] for s in samples], dtype=np.float64)
    se = np.asarray([s["uncertainty"] for s in samples], dtype=np.float64)

    lo = float(np.quantile(h_vis, percentile))
    hi = float(np.quantile(h_vis, 1.0 - percentile))

    # Confident = low visual entropy; uncertain = high visual entropy.
    confident_mask = h_vis <= lo
    uncertain_mask = h_vis >= hi

    # If ties at the quantile push one side empty, fall back to strict argsort ranks.
    if confident_mask.sum() == 0 or uncertain_mask.sum() == 0:
        order = np.argsort(h_vis)
        k = max(1, int(round(len(samples) * percentile)))
        confident_idx = order[:k]
        uncertain_idx = order[-k:]
        se_conf = se[confident_idx]
        se_unc = se[uncertain_idx]
        lo = float(h_vis[confident_idx].max())
        hi = float(h_vis[uncertain_idx].min())
    else:
        se_conf = se[confident_mask]
        se_unc = se[uncertain_mask]

    stats = {
        "n_total": float(len(samples)),
        "n_confident": float(len(se_conf)),
        "n_uncertain": float(len(se_unc)),
        "h_vis_low_threshold": lo,
        "h_vis_high_threshold": hi,
        "se_confident_mean": float(se_conf.mean()) if len(se_conf) else float("nan"),
        "se_uncertain_mean": float(se_unc.mean()) if len(se_unc) else float("nan"),
    }
    return se_conf, se_unc, stats


def _plot_one_hist(ax, values: np.ndarray, title: str, color: str):
    if len(values) == 0:
        ax.set_title(f"{title}\n(n=0)")
        ax.set_xlabel("Semantic Entropy")
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
    ax.set_xlabel("Semantic Entropy")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)


def plot_distributions(
    se_confident: np.ndarray,
    se_uncertain: np.ndarray,
    stats: Dict[str, float],
    out_path: str,
    title: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _plot_one_hist(
        axes[0],
        se_confident,
        "Visually Confident (low $H_{vis}$)",
        color="#4C78A8",
    )
    _plot_one_hist(
        axes[1],
        se_uncertain,
        "Visually Uncertain (high $H_{vis}$)",
        color="#F58518",
    )

    subtitle = (
        f"thresholds: H_vis≤{stats['h_vis_low_threshold']:.3f} / "
        f"H_vis≥{stats['h_vis_high_threshold']:.3f}  |  "
        f"N={int(stats['n_total'])}"
    )
    fig.suptitle(title or "Semantic Entropy by Visual Certainty", fontsize=13)
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SE distributions for visually certain vs uncertain samples."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to exp/log_*.json produced with --compute_visual_entropy true.",
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
        default="exp/fig_visual_certainty_se.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log = load_log(args.log)
    print_run_metrics(log)
    samples = load_samples(log)
    se_conf, se_unc, stats = split_by_visual_certainty(
        samples,
        percentile=args.percentile,
        incorrect_only=args.incorrect_only,
    )
    print("Split stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    title = args.title
    if title is None:
        subset = "incorrect answers" if args.incorrect_only else "all answers"
        title = f"Semantic Entropy by Visual Certainty ({subset})"

    plot_distributions(se_conf, se_unc, stats, args.out, title=title)


if __name__ == "__main__":
    main()
