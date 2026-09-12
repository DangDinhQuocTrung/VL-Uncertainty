"""Recompute detection AUROC for logit-based UE fields from an NLL log.

Supports fields logged by methods/nll.py (one generation pass):
  - avg_nll / max_nll          (ANLL / Max-NLL; higher = more uncertain)
  - avg_ent / max_ent          (AvgEnt / MaxEnt; higher = more uncertain)
  - avg_prob / max_prob        (raw confidence; AUROC uses 1 - prob)
  - min_prob                   (AUROC uses 1 - min_prob; = Max-NLL ranking)
  - uncertainty                (whatever --nll_mode selected at run time)

Example:
  PYTHONPATH=. python plotting/compute_logit_field_auroc.py \\
      --log exp_02/log_YYYY_MM_DD_HH_MM_SS.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchmetrics.functional import auroc

from utils.constants import LOG_META_KEYS, PERTURBATION_DETECTION_DATASETS


# Fields used directly as uncertainty (higher => more uncertain).
DIRECT_UNCERTAIN_FIELDS = (
    "uncertainty",
    "avg_nll",
    "max_nll",
    "avg_ent",
    "max_ent",
)

# Confidence fields: evaluate AUROC on (1 - value).
CONFIDENCE_FIELDS = (
    "avg_prob",
    "max_prob",
    "min_prob",
)

SCORE_FIELDS = DIRECT_UNCERTAIN_FIELDS + CONFIDENCE_FIELDS


def load_log(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r") as f:
        return json.load(f)


def _is_sample_entry(key: str, value: Any) -> bool:
    if key in LOG_META_KEYS or not isinstance(value, dict):
        return False
    if "Hallucination detection" in key or "Perturbation detection" in key:
        return False
    return bool(value.get("flag_sample_valid", True)) and "uncertainty" in value


def infer_detection_mode(log: Dict[str, Any]) -> Tuple[str, bool]:
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


def load_samples(log: Dict[str, Any], score_fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    samples = []
    for key, value in log.items():
        if not _is_sample_entry(key, value):
            continue
        sample = {
            "idx": int(key) if str(key).isdigit() else key,
            "flag_answer_correct": bool(value.get("flag_answer_correct", True)),
            "flag_perturbed_inputs": value.get("flag_perturbed_inputs"),
        }
        for field in score_fields:
            if field in value:
                sample[field] = float(value[field])
        samples.append(sample)
    if not samples:
        raise ValueError(
            "No valid samples with 'uncertainty' found in log. "
            "Pass an nll / logit run log from main.py."
        )
    return samples


def detection_targets(
    samples: List[Dict[str, Any]], is_perturbation_detection: bool
) -> torch.Tensor:
    if is_perturbation_detection:
        missing = [s["idx"] for s in samples if s.get("flag_perturbed_inputs") is None]
        if missing:
            raise ValueError(
                "Perturbation detection requires flag_perturbed_inputs, "
                f"missing on samples: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        targets = [bool(s["flag_perturbed_inputs"]) for s in samples]
    else:
        targets = [not s["flag_answer_correct"] for s in samples]
    return torch.tensor(targets, dtype=torch.int)


def field_scores(samples: List[Dict[str, Any]], field: str) -> Optional[torch.Tensor]:
    if any(field not in s for s in samples):
        return None
    raw = torch.tensor([s[field] for s in samples], dtype=torch.float)
    if field in CONFIDENCE_FIELDS:
        return 1.0 - raw
    return raw


def compute_field_auroc(
    samples: List[Dict[str, Any]],
    field: str,
    detection_gt: torch.Tensor,
) -> Optional[float]:
    scores = field_scores(samples, field)
    if scores is None:
        return None
    if detection_gt.unique().numel() < 2:
        return float("nan")
    return float(auroc(scores, detection_gt, task="binary").item())


def print_summary(log: Dict[str, Any], detection_label: str, n_samples: int) -> None:
    method = log.get("uncertainty_method", "(unknown)")
    dataset = log.get("dataset_name", "(unknown)")
    print(f"dataset: {dataset}")
    print(f"uncertainty_method: {method}")
    print(f"n_samples: {n_samples}")
    print(f"detection task: {detection_label}")
    base_acc = log.get("Base task Accuracy")
    if base_acc is not None:
        print(f"Base task Accuracy (logged): {float(base_acc):.4f}")
    logged_auroc = log.get(f"{detection_label} AUROC")
    if logged_auroc is not None:
        print(f"{detection_label} AUROC (logged, uncertainty): {float(logged_auroc):.4f}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute detection AUROC for logit UE fields from an nll log."
    )
    parser.add_argument("--log", type=str, required=True, help="Path to nll exp/log_*.json.")
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=list(SCORE_FIELDS),
        help=f"Score fields to evaluate. Default: {', '.join(SCORE_FIELDS)}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log = load_log(args.log)
    score_fields = tuple(args.fields)

    detection_label, is_perturbation = infer_detection_mode(log)
    samples = load_samples(log, score_fields)
    detection_gt = detection_targets(samples, is_perturbation)

    print_summary(log, detection_label, len(samples))
    print(f"{detection_label} AUROC by field (prob fields use 1 - prob):")
    for field in score_fields:
        score = compute_field_auroc(samples, field, detection_gt)
        if score is None:
            print(f"  {field}: (missing in log — re-run with updated methods/nll.py)")
        elif score != score:
            print(f"  {field}: (undefined; need both positive and negative labels)")
        else:
            print(f"  {field}: {score:.4f}")


if __name__ == "__main__":
    main()
