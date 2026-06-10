import argparse
import json
from pathlib import Path

import torch
from torchmetrics.functional import auroc


RESULTS_JSON_PATH = Path(__file__).with_name("discussion_results_0000-0100.json")
CALIBRATION_THRESHOLD = 0.5


def load_question_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = []
    for key, value in data.items():
        if not key.isdigit() or not isinstance(value, dict):
            continue
        if value.get("confidence") is None or "correct" not in value:
            continue
        records.append(value)
    return records


def calibrate_confidence(confidence: float) -> float:
    if confidence <= CALIBRATION_THRESHOLD:
        return 0.0
    if confidence >= 1.0:
        return 1.0
    return (confidence - CALIBRATION_THRESHOLD) / (1.0 - CALIBRATION_THRESHOLD)


def compute_brier_score(confidences: list[float], outcomes: list[float]) -> float | None:
    if not confidences:
        return None
    return sum((p - y) ** 2 for p, y in zip(confidences, outcomes)) / len(confidences)


def compute_auroc(confidences: list[float], outcomes: list[float]) -> float | None:
    if len(confidences) < 2:
        return None

    num_positive = sum(outcomes)
    num_negative = len(outcomes) - num_positive
    if num_positive == 0 or num_negative == 0:
        return None

    preds = torch.tensor(confidences, dtype=torch.float32)
    targets = torch.tensor(outcomes, dtype=torch.long)
    return auroc(preds, targets, task="binary").item()


def evaluate(confidences: list[float], outcomes: list[float]) -> dict[str, float | None]:
    return {
        "brier_score": compute_brier_score(confidences, outcomes),
        "auroc": compute_auroc(confidences, outcomes),
    }


def run_calibration(results_path: Path = RESULTS_JSON_PATH) -> dict:
    records = load_question_records(results_path)
    if not records:
        raise ValueError(f"No records with confidence and correctness in {results_path}")

    outcomes = [record["f1"] if record["f1"] is not None else 0.0 for record in records]
    raw_confidences = [float(record.get("confidence", 0.5)) for record in records]
    calibrated_confidences = [calibrate_confidence(c) for c in raw_confidences]

    raw_metrics = evaluate(raw_confidences, outcomes)
    calibrated_metrics = evaluate(calibrated_confidences, outcomes)

    print(f"Loaded {len(records)} records from {results_path}")
    print(
        f"Calibration: confidence <= {CALIBRATION_THRESHOLD} -> 0.0; "
        f"({CALIBRATION_THRESHOLD}, 1.0] linearly mapped to (0.0, 1.0]"
    )
    print()
    print("Original confidence:")
    print(f"  Brier score: {raw_metrics['brier_score']:.4f}" if raw_metrics["brier_score"] is not None else "  Brier score: unavailable")
    print(f"  AUROC:       {raw_metrics['auroc']:.4f}" if raw_metrics["auroc"] is not None else "  AUROC:       unavailable")
    print()
    print("Calibrated confidence:")
    print(f"  Brier score: {calibrated_metrics['brier_score']:.4f}" if calibrated_metrics["brier_score"] is not None else "  Brier score: unavailable")
    print(f"  AUROC:       {calibrated_metrics['auroc']:.4f}" if calibrated_metrics["auroc"] is not None else "  AUROC:       unavailable")

    return {
        "num_records": len(records),
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate confidence scores and compute Brier score and AUROC."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_JSON_PATH,
        help="Discussion results JSON (default: hotpot/discussion_results.json)",
    )
    args = parser.parse_args()
    run_calibration(args.input)


if __name__ == "__main__":
    main()
