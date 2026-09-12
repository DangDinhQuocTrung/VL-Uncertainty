#!/usr/bin/env python3
"""Plot uncertainty quantification AUROC results from exp log JSON files.

Example:
  python scripts/plot_uq_results.py --dataset ViLP --mode revised
  python scripts/plot_uq_results.py --dataset MedVIGIL --mode raw --exp_dir ./exp
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import DETECTION_AUROC_KEYS

LOG_NAME_RE = re.compile(
    r"^(?:log|.+)_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}(?:_revised)?\.json$"
)
BENCHMARK_IN_ARGS_RE = re.compile(r"benchmark=['\"]([^'\"]+)['\"]")
UNCERTAINTY_IN_ARGS_RE = re.compile(r"uncertainty=['\"]([^'\"]+)['\"]")

METHOD_DISPLAY = {
    "vl_uncertainty": "VL-Uncertainty",
    "semantic_entropy": "Semantic Entropy",
    "vauq": "VAUQ",
    "svar": "SVAR",
    "euq": "EUQ",
    "nll": "NLL",
    "pro": "PRO",
    "rds": "RDS",
    "vse": "VSE",
    "vse_masked": "VSE-Masked",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter exp logs by dataset and plot hallucination/perturbation "
            "detection AUROC for each uncertainty method."
        )
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="./exp",
        help=(
            "Directory containing experiment logs "
            "(log_*.json or {benchmark}_{method}_*.json)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Keep only logs whose dataset_name matches this value.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="raw",
        choices=["raw", "revised"],
        help=(
            "raw: use original experiment logs (exclude *_revised.json); "
            "revised: use only *_revised.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <exp_dir>/auroc_<dataset>_<mode>.png",
    )
    parser.add_argument(
        "--show",
        type=lambda x: x.lower() == "true",
        default="False",
        help="If true, also open an interactive plot window.",
    )
    return parser.parse_args()


def get_dataset_name(log_dict):
    dataset_name = log_dict.get("dataset_name")
    if dataset_name:
        return str(dataset_name)
    args_str = str(log_dict.get("args", ""))
    match = BENCHMARK_IN_ARGS_RE.search(args_str)
    return match.group(1) if match else None


def get_method_name(log_dict):
    method = log_dict.get("uncertainty_method")
    if method:
        return str(method)
    args_str = str(log_dict.get("args", ""))
    match = UNCERTAINTY_IN_ARGS_RE.search(args_str)
    return match.group(1) if match else "unknown"


def metadata_items(log_dict):
    """Yield top-level metadata only (skip per-sample numeric keys)."""
    for key, value in log_dict.items():
        if str(key).isdigit():
            continue
        yield str(key), value


def extract_detection_auroc(log_dict):
    """Pick the hallucination/perturbation detection AUROC from metadata.

    Logs can contain several metric fields; only the detection AUROC keys are
    used. If both hallucination and perturbation AUROCs are present, prefer the
    one that matches ``dataset_name`` / ``PERTURBATION`` naming in the key set.
    """
    auroc_candidates = {}
    for key, value in metadata_items(log_dict):
        if key in DETECTION_AUROC_KEYS:
            auroc_candidates[key] = float(value)
        elif key.endswith(" AUROC") or key.endswith("_AUROC") or key == "AUROC":
            # Catch unexpected AUROC field names without picking Accuracy/etc.
            try:
                auroc_candidates[key] = float(value)
            except (TypeError, ValueError):
                continue

    if not auroc_candidates:
        return None, None

    for preferred in DETECTION_AUROC_KEYS:
        if preferred in auroc_candidates:
            return preferred, auroc_candidates[preferred]

    # Fallback: any AUROC-like metadata key
    key = sorted(auroc_candidates.keys())[0]
    return key, auroc_candidates[key]


def list_log_files(exp_dir: Path, mode: str):
    files = []
    for path in sorted(exp_dir.glob("*.json")):
        name = path.name
        if name.endswith("_answer_collection.json"):
            continue
        if not LOG_NAME_RE.match(name):
            continue
        is_revised = name.endswith("_revised.json")
        if mode == "revised":
            if is_revised:
                files.append(path)
        else:  # raw
            if not is_revised:
                files.append(path)
    return files


def load_results(exp_dir: Path, dataset: str, mode: str):
    """Return list of result dicts for the requested dataset and mode."""
    results = []
    for path in list_log_files(exp_dir, mode):
        with open(path, "r") as f:
            log_dict = json.load(f)

        log_dataset = get_dataset_name(log_dict)
        if log_dataset != dataset:
            continue

        auroc_key, auroc = extract_detection_auroc(log_dict)
        if auroc is None:
            print(f"- Skip (no detection AUROC): {path.name}")
            continue

        base_acc = log_dict.get("Base task Accuracy")
        if base_acc is None:
            print(f"- Warning: missing Base task Accuracy in {path.name}")
            base_acc = float("nan")
        else:
            base_acc = float(base_acc)

        results.append(
            {
                "path": path,
                "dataset": log_dataset,
                "method": get_method_name(log_dict),
                "auroc_key": auroc_key,
                "auroc": auroc,
                "base_acc": base_acc,
            }
        )
    return results


def dedupe_by_method(results):
    """Keep the latest log per method when multiple runs exist."""
    by_method = {}
    for row in results:
        method = row["method"]
        prev = by_method.get(method)
        if prev is None or row["path"].name > prev["path"].name:
            if prev is not None:
                print(
                    f"- Multiple logs for method={method}; "
                    f"keeping newer {row['path'].name} over {prev['path'].name}"
                )
            by_method[method] = row
    # Stable order: prefer known METHODS order, then alphabetical
    preferred = list(METHOD_DISPLAY.keys())
    methods = sorted(
        by_method.keys(),
        key=lambda m: (preferred.index(m) if m in preferred else len(preferred), m),
    )
    return [by_method[m] for m in methods]


def display_method(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def plot_auroc_bar(results, dataset: str, mode: str, output_path: Path, show: bool):
    methods = [display_method(r["method"]) for r in results]
    aurocs = [r["auroc"] for r in results]
    base_accs = [r["base_acc"] for r in results if not np.isnan(r["base_acc"])]
    median_base = statistics.median(base_accs) if base_accs else float("nan")

    ylabel = results[0]["auroc_key"] if results else "Detection AUROC"
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(methods))]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(methods) + 2), 7.5))
    x = np.arange(len(methods))
    bars = ax.bar(x, aurocs, width=0.65, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(1.0, max(aurocs) * 1.15 if aurocs else 1.0))
    ax.set_title(f"Dataset: {dataset}  |  mode: {mode}")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    for bar, value in zip(bars, aurocs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    median_text = (
        f"Median base task accuracy: {median_base:.3f}"
        if not np.isnan(median_base)
        else "Median base task accuracy: n/a"
    )
    ax.text(
        0.02,
        0.98,
        median_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], ec="black", label=methods[i])
        for i in range(len(methods))
    ]
    handles.append(
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
    )
    ax.legend(handles=handles, title="Method", loc="upper right", framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"- Saved plot to {output_path}")
    print(f"- {median_text}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    results = load_results(exp_dir, args.dataset, args.mode)
    if not results:
        raise FileNotFoundError(
            f"No matching {args.mode} logs for dataset={args.dataset} in {exp_dir}"
        )

    results = dedupe_by_method(results)
    print(f"- Found {len(results)} method(s) for dataset={args.dataset} mode={args.mode}:")
    for row in results:
        print(
            f"  {row['method']:20s}  AUROC={row['auroc']:.4f}  "
            f"base_acc={row['base_acc']:.4f}  [{row['path'].name}]"
        )

    output = (
        Path(args.output)
        if args.output
        else exp_dir / f"auroc_{args.dataset}_{args.mode}.png"
    )
    plot_auroc_bar(results, args.dataset, args.mode, output, show=args.show)


if __name__ == "__main__":
    main()
