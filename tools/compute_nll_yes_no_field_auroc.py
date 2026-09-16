"""Recompute detection AUROC for NLL Yes/No score fields from an existing log.

Matches main.py:
  - Hallucination detection: detection_gt = 1 - flag_answer_correct
  - Perturbation detection (MedVIGIL): detection_gt = flag_perturbed_inputs

Raw log fields:
  - uncertainty
  - positive_answer_yes_logit / positive_answer_no_logit
  - negative_answer_yes_logit / negative_answer_no_logit

Derived combined scores (higher => more uncertain):
  - yn_pos_entropy:          H(softmax([y+, n+]))
  - yn_neg_entropy:          H(softmax([y-, n-]))
  - yn_avg_entropy:          0.5 * (H(p+) + H(p-))          # ignorance
  - yn_disagreement:         |p+_yes - p-_no|                # conflict vs flipped neg
  - yn_js_divergence:        JS(p+, flip(p-))
  - yn_margin_abs_diff:      |(y+ - n+) - (n- - y-)|         # inconsistent margins
  - yn_neg_combined_margin:  -|0.5 * ((y+-n+) + (n--y-))|    # low joint confidence
  - yn_inconsistency:        1 if pos/neg disagree after flip, else 0
  - yn_total:                yn_avg_entropy + yn_disagreement
  - yn_pos_ent_plus_disagree: yn_pos_entropy + yn_disagreement
  - yn_pos_ent_plus_js:      yn_pos_entropy + yn_js_divergence
  - yn_pos_ent_x_disagree:   yn_pos_entropy * yn_disagreement
  - yn_pos_ent_x_js:         yn_pos_entropy * yn_js_divergence
  - yn_pos_ent_x1p_disagree: yn_pos_entropy * (1 + yn_disagreement)
  - yn_pos_ent_x1p_js:       yn_pos_entropy * (1 + yn_js_divergence)

Example:
  python plotting/compute_nll_yes_no_field_auroc.py \\
      --log exp_04/log_YYYY_MM_DD_HH_MM_SS.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchmetrics.functional import auroc

from utils.constants import LOG_META_KEYS, PERTURBATION_DETECTION_DATASETS


RAW_SCORE_FIELDS = (
    "max_nll", "avg_nll",
    "uncertainty",
    "positive_answer_yes_logit",
    "positive_answer_no_logit",
    "negative_answer_yes_logit",
    "negative_answer_no_logit",
)

DERIVED_SCORE_FIELDS = (
    "yn_pos_entropy",
    "yn_neg_entropy",
    "yn_avg_entropy",
    "yn_disagreement",
    "yn_js_divergence",
    "yn_margin_abs_diff",
    "yn_neg_combined_margin",
    "yn_inconsistency",
    "yn_total",
    "yn_pos_ent_plus_disagree",
    "yn_pos_ent_plus_js",
    "yn_pos_ent_x_disagree",
    "yn_pos_ent_x_js",
    "yn_pos_ent_x1p_disagree",
    "yn_pos_ent_x1p_js",
)

LOGIT_FIELDS = (
    "positive_answer_yes_logit",
    "positive_answer_no_logit",
    "negative_answer_yes_logit",
    "negative_answer_no_logit",
)


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


def _binary_entropy(p_yes: float) -> float:
    p = min(max(p_yes, 1e-12), 1.0 - 1e-12)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def _js_divergence_binary(p_yes: float, q_yes: float) -> float:
    """Jensen–Shannon divergence between two Bernoulli distributions (nats)."""
    m = 0.5 * (p_yes + q_yes)
    # KL(Bern(p) || Bern(m)) + KL(Bern(q) || Bern(m))
    def kl(a: float, b: float) -> float:
        a = min(max(a, 1e-12), 1.0 - 1e-12)
        b = min(max(b, 1e-12), 1.0 - 1e-12)
        return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))

    return 0.5 * kl(p_yes, m) + 0.5 * kl(q_yes, m)


def derive_yes_no_scores(sample: Dict[str, Any]) -> Dict[str, float]:
    """Combine pos/neg yes-no logits into uncertainty scores.

    Positive and negative questions are semantic flips, so a consistent model
    should answer opposite labels. We map the negative distribution into the
    positive semantic frame by swapping yes/no:
        p-_yes_equiv = P_neg(no) = 1 - P_neg(yes)
    """
    y_pos = float(sample["positive_answer_yes_logit"])
    n_pos = float(sample["positive_answer_no_logit"])
    y_neg = float(sample["negative_answer_yes_logit"])
    n_neg = float(sample["negative_answer_no_logit"])

    pos_probs = torch.softmax(torch.tensor([y_pos, n_pos], dtype=torch.float), dim=0)
    neg_probs = torch.softmax(torch.tensor([y_neg, n_neg], dtype=torch.float), dim=0)
    p_pos_yes = float(pos_probs[0])
    p_neg_yes = float(neg_probs[0])
    # Flip negative into positive wording: "yes" on pos ≡ "no" on neg.
    p_neg_yes_equiv = 1.0 - p_neg_yes

    margin_pos = y_pos - n_pos
    # Negative margin in positive semantic frame: prefer "no" on the flipped Q.
    margin_neg_equiv = n_neg - y_neg
    combined_margin = 0.5 * (margin_pos + margin_neg_equiv)

    pos_pred_yes = margin_pos > 0
    neg_pred_yes_equiv = margin_neg_equiv > 0
    inconsistent = float(pos_pred_yes != neg_pred_yes_equiv)

    pos_entropy = _binary_entropy(p_pos_yes)
    neg_entropy = _binary_entropy(p_neg_yes)
    avg_entropy = 0.5 * (pos_entropy + neg_entropy)
    disagreement = abs(p_pos_yes - p_neg_yes_equiv)
    js = _js_divergence_binary(p_pos_yes, p_neg_yes_equiv)

    return {
        "yn_pos_entropy": pos_entropy,
        "yn_neg_entropy": neg_entropy,
        "yn_avg_entropy": avg_entropy,
        "yn_disagreement": disagreement,
        "yn_js_divergence": js,
        "yn_margin_abs_diff": abs(margin_pos - margin_neg_equiv),
        # Negate so larger score = less confident / more uncertain for AUROC.
        "yn_neg_combined_margin": -abs(combined_margin),
        "yn_inconsistency": inconsistent,
        "yn_total": avg_entropy + disagreement,
        # Positive-entropy + conflict combinations.
        "yn_pos_ent_plus_disagree": pos_entropy + disagreement,
        "yn_pos_ent_plus_js": pos_entropy + js,
        "yn_pos_ent_x_disagree": pos_entropy * disagreement,
        "yn_pos_ent_x_js": pos_entropy * js,
        "yn_pos_ent_x1p_disagree": pos_entropy * (1.0 + disagreement),
        "yn_pos_ent_x1p_js": pos_entropy * (1.0 + js),
    }


def load_samples(log: Dict[str, Any], score_fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    samples = []
    need_logits = any(f in DERIVED_SCORE_FIELDS or f in LOGIT_FIELDS for f in score_fields)
    for key, value in log.items():
        if not _is_sample_entry(key, value):
            continue
        sample = {
            "idx": int(key) if str(key).isdigit() else key,
            "flag_answer_correct": bool(value.get("flag_answer_correct", True)),
            "flag_perturbed_inputs": value.get("flag_perturbed_inputs"),
        }
        for field in RAW_SCORE_FIELDS:
            if field in value:
                sample[field] = float(value[field])

        if need_logits and all(f in sample for f in LOGIT_FIELDS):
            sample.update(derive_yes_no_scores(sample))

        samples.append(sample)
    if not samples:
        raise ValueError(
            "No valid samples with 'uncertainty' found in log. "
            "Pass an nll_yes_no run log from main.py."
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
        # Higher score should rank incorrect answers higher.
        targets = [not s["flag_answer_correct"] for s in samples]
    return torch.tensor(targets, dtype=torch.int)


def compute_field_auroc(
    samples: List[Dict[str, Any]],
    field: str,
    detection_gt: torch.Tensor,
) -> Optional[float]:
    if any(field not in s for s in samples):
        return None
    scores = torch.tensor([s[field] for s in samples], dtype=torch.float)
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
        description="Compute detection AUROC for NLL Yes/No score fields from a log."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to an nll_yes_no exp/log_*.json file.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=list(RAW_SCORE_FIELDS) + list(DERIVED_SCORE_FIELDS),
        help=(
            "Score fields to evaluate. Default: raw logit fields + derived "
            f"combinations ({', '.join(DERIVED_SCORE_FIELDS)})."
        ),
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
    print(f"{detection_label} AUROC by field:")
    for field in score_fields:
        score = compute_field_auroc(samples, field, detection_gt)
        if score is None:
            print(f"  {field}: (missing in log)")
        elif score != score:  # NaN
            print(f"  {field}: (undefined; need both positive and negative labels)")
        else:
            print(f"  {field}: {score:.4f}")
    return


if __name__ == "__main__":
    main()
