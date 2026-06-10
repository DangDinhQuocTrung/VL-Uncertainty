import asyncio
import json
from pathlib import Path

from hotpot.check_agent import create_dtu_model_client
from hotpot.check_discussion_single import (
    build_result_record,
    discuss_hotpot_question_quiet,
    evaluate_hotpot_answer,
    load_hotpot_dataset,
)

QUESTION_START_INDEX = 0
QUESTION_END_INDEX = 100
RESULTS_JSON_PATH = Path(__file__).with_name(f"discussion_results_{QUESTION_START_INDEX:04d}-{QUESTION_END_INDEX:04d}.json")
def load_existing_records(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}

    data = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        key: value
        for key, value in data.items()
        if key.isdigit() and isinstance(value, dict)
    }


def collect_confidence_outcomes(
    records: dict[str, dict],
    *,
    outcome_key: str = "correct",
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for record in records.values():
        confidence = record.get("confidence", 0.0)
        confidence = 0.0 if confidence is None else confidence
        outcome = 1.0 if record[outcome_key] else 0.0
        pairs.append((float(confidence), outcome))
    return pairs


def compute_brier_score(records: dict[str, dict]) -> float | None:
    pairs = collect_confidence_outcomes(records, outcome_key="correct")
    if not pairs:
        return None
    return sum((confidence - outcome) ** 2 for confidence, outcome in pairs) / len(pairs)


def compute_pearson_correlation(
    records: dict[str, dict],
    *,
    outcome_key: str = "correct",
) -> float | None:
    pairs = collect_confidence_outcomes(records, outcome_key=outcome_key)
    if len(pairs) < 2:
        return None

    confidences = [confidence for confidence, _ in pairs]
    outcomes = [outcome for _, outcome in pairs]
    n = len(pairs)

    mean_confidence = sum(confidences) / n
    mean_outcome = sum(outcomes) / n

    covariance = sum(
        (confidence - mean_confidence) * (outcome - mean_outcome)
        for confidence, outcome in pairs
    )
    confidence_std = sum((confidence - mean_confidence) ** 2 for confidence in confidences) ** 0.5
    outcome_std = sum((outcome - mean_outcome) ** 2 for outcome in outcomes) ** 0.5

    if confidence_std == 0.0 or outcome_std == 0.0:
        return None

    return covariance / (confidence_std * outcome_std)


def compute_hotpot_metrics(records: dict[str, dict]) -> dict[str, float | None]:
    if not records:
        return {"em": None, "f1": None}

    em_scores = [float(record["em"]) for record in records.values() if "em" in record]
    f1_scores = [float(record["f1"]) for record in records.values() if "f1" in record]
    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else None,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
    }


def build_results(records: dict[str, dict]) -> dict:
    num_questions = len(records)
    num_correct = sum(1 for record in records.values() if record["correct"])
    llm_accuracy = num_correct / num_questions if num_questions else 0.0
    hotpot_metrics = compute_hotpot_metrics(records)

    return {
        **records,
        "accuracy": llm_accuracy,
        "em": hotpot_metrics["em"],
        "f1": hotpot_metrics["f1"],
        "brier_score": compute_brier_score(records),
        "pearson_correlation": compute_pearson_correlation(records, outcome_key="correct"),
        "pearson_correlation_em": compute_pearson_correlation(records, outcome_key="em"),
    }


def write_results(output_path: Path, results: dict) -> None:
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def run_hotpot_batch(
    start_index: int = QUESTION_START_INDEX,
    end_index: int = QUESTION_END_INDEX,
    output_path: Path = RESULTS_JSON_PATH,
) -> dict:
    dataset = load_hotpot_dataset()
    records = load_existing_records(output_path)
    if records:
        print(f"Loaded {len(records)} existing records from {output_path}", flush=True)

    model_client = create_dtu_model_client()
    total_questions = end_index - start_index

    try:
        for index in range(start_index, end_index):
            if str(index) in records:
                print(
                    f"\nSkipping question {index} "
                    f"({index - start_index + 1}/{total_questions}, already saved)",
                    flush=True,
                )
                continue

            print(f"\nRunning question {index} ({index - start_index + 1}/{total_questions})...", flush=True)
            output = await discuss_hotpot_question_quiet(
                index,
                dataset=dataset,
                model_client=model_client,
            )
            evaluation = await evaluate_hotpot_answer(output, model_client=model_client)
            record = build_result_record(output, evaluation)
            records[str(index)] = record
            write_results(output_path, records)
            print(
                f"  model_answer={record['model_answer']!r} "
                f"gt_answer={record['gt_answer']!r} "
                f"confidence={record['confidence']} "
                f"correct={record['correct']} "
                f"em={record['em']} "
                f"f1={record['f1']:.3f}",
                flush=True,
            )
            print(f"  Saved progress to {output_path}", flush=True)
    finally:
        await model_client.close()

    results = build_results(records)
    write_results(output_path, results)
    num_questions = len(records)
    num_correct = sum(1 for record in records.values() if record["correct"])
    llm_accuracy = results["accuracy"]
    hotpot_metrics = {"em": results["em"], "f1": results["f1"]}
    brier_score = results["brier_score"]
    pearson_correlation = results["pearson_correlation"]
    pearson_correlation_em = results["pearson_correlation_em"]

    print(f"\nFinished. Results in {output_path}", flush=True)
    print(f"LLM accuracy: {num_correct}/{num_questions} = {llm_accuracy:.2%}", flush=True)
    if hotpot_metrics["em"] is None:
        print("HotpotQA EM: unavailable", flush=True)
    else:
        print(f"HotpotQA EM: {hotpot_metrics['em']:.2%}", flush=True)
    if hotpot_metrics["f1"] is None:
        print("HotpotQA F1: unavailable", flush=True)
    else:
        print(f"HotpotQA F1: {hotpot_metrics['f1']:.4f}", flush=True)
    if brier_score is None:
        print("Brier score (vs LLM correctness): unavailable", flush=True)
    else:
        print(f"Brier score (vs LLM correctness): {brier_score:.4f}", flush=True)
    if pearson_correlation is None:
        print(
            "Pearson correlation (confidence vs LLM correctness): unavailable",
            flush=True,
        )
    else:
        print(
            f"Pearson correlation (confidence vs LLM correctness): {pearson_correlation:.4f}",
            flush=True,
        )
    if pearson_correlation_em is None:
        print(
            "Pearson correlation (confidence vs HotpotQA EM): unavailable",
            flush=True,
        )
    else:
        print(
            f"Pearson correlation (confidence vs HotpotQA EM): {pearson_correlation_em:.4f}",
            flush=True,
        )

    return results


def check_discussion() -> None:
    asyncio.run(
        run_hotpot_batch(
            start_index=QUESTION_START_INDEX,
            end_index=QUESTION_END_INDEX,
            output_path=RESULTS_JSON_PATH,
        )
    )


if __name__ == "__main__":
    check_discussion()
