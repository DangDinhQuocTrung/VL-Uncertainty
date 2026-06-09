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
QUESTION_END_INDEX = 5
RESULTS_JSON_PATH = Path(__file__).with_name("discussion_results.json")


async def run_hotpot_batch(
    start_index: int = QUESTION_START_INDEX,
    end_index: int = QUESTION_END_INDEX,
    output_path: Path = RESULTS_JSON_PATH,
) -> dict[str, dict[str, str | bool]]:
    dataset = load_hotpot_dataset()
    results: dict[str, dict[str, str | bool]] = {}
    model_client = create_dtu_model_client()

    try:
        for index in range(start_index, end_index):
            print(f"\nRunning question {index} ({index - start_index + 1}/{end_index - start_index})...", flush=True)
            output = await discuss_hotpot_question_quiet(
                index,
                dataset=dataset,
                model_client=model_client,
            )
            evaluation = await evaluate_hotpot_answer(output, model_client=model_client)
            record = build_result_record(output, evaluation)
            results[str(index)] = record
            print(
                f"  model_answer={record['model_answer']!r} "
                f"gt_answer={record['gt_answer']!r} "
                f"correct={record['correct']}",
                flush=True,
            )
    finally:
        await model_client.close()

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote results to {output_path}", flush=True)

    num_questions = len(results)
    num_correct = sum(1 for record in results.values() if record["correct"])
    accuracy = num_correct / num_questions if num_questions else 0.0
    print(f"Accuracy: {num_correct}/{num_questions} = {accuracy:.2%}", flush=True)

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
