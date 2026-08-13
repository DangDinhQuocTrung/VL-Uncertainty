import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import torch
from torchmetrics.functional import accuracy, auroc, precision, recall
from tqdm import tqdm

from llm.Claude import Claude
from llm.Gemma import Gemma
from llm.Qwen import Qwen
from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm
from utils.constants import LLM_MAP, PERTURBATION_DETECTION_DATASETS
from utils.metrics import compute_f1_score

LOG_NAME_RE = re.compile(r"^log_.*\.json$")
BENCHMARK_IN_ARGS_RE = re.compile(r"benchmark=['\"]([^'\"]+)['\"]")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process exp log JSON files: collect unique answers per question ID, "
            "re-judge with a larger LLM, then write revised logs with updated "
            "llm_answer_check and detection metrics."
        )
    )
    parser.add_argument("--output_dir", type=str, default="./exp")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Match the log dataset_name field (fallback: benchmark= in args).",
    )
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help="Judge LLM name, e.g. gemma-3-27b-it, Qwen2.5-72B-Instruct, or claude-sonnet-5.",
    )
    parser.add_argument(
        "--resume",
        type=lambda x: x.lower() == "true",
        default="True",
        help="Reuse existing collection checks when dataset and judge LLM match.",
    )
    return parser.parse_args()


def obtain_judge_llm(llm_name):
    llm_class = LLM_MAP.get(llm_name)
    if llm_class is not None:
        return llm_class(llm_name)
    lowered = llm_name.lower()
    if lowered.startswith("claude"):
        return Claude(llm_name)
    if "qwen" in lowered:
        return Qwen(llm_name)
    if "gemma" in lowered:
        return Gemma(llm_name)
    raise ValueError(
        f"Unsupported LLM: {llm_name}. Known names: {list(LLM_MAP.keys())}"
    )


def get_dataset_name(log_dict):
    dataset_name = log_dict.get("dataset_name")
    if dataset_name:
        return str(dataset_name)
    args_str = str(log_dict.get("args", ""))
    match = BENCHMARK_IN_ARGS_RE.search(args_str)
    return match.group(1) if match else None


def list_log_files(output_dir):
    files = []
    for path in sorted(Path(output_dir).glob("*.json")):
        name = path.name
        if name.endswith("_revised.json") or name.endswith("_answer_collection.json"):
            continue
        if not LOG_NAME_RE.match(name):
            continue
        files.append(path)
    return files


def iter_sample_ids(log_dict):
    """Return string question IDs using Total samples, plus any extra numeric keys.

    Log keys are question IDs, but dict iteration also hits metadata such as
    weight_dir. Sample IDs are 0 .. Total samples-1 when every sample is valid.
    Extra numeric keys are included so invalid-sample holes are not dropped.
    """
    total = log_dict.get("Total samples")
    digit_ids = [int(k) for k in log_dict.keys() if str(k).isdigit()]
    upper = int(total) if total is not None else 0
    if digit_ids:
        upper = max(upper, max(digit_ids) + 1)
    ids = []
    for idx in range(upper):
        key = str(idx)
        if key in log_dict:
            ids.append(key)
    return ids


def get_sample(log_dict, sample_id):
    sample = log_dict.get(str(sample_id))
    if sample is None:
        sample = log_dict.get(int(sample_id)) if str(sample_id).isdigit() else None
    return sample if isinstance(sample, dict) else None


def parse_answer_correctness(llm_answer_check, model_answer, gt_answer):
    check = (llm_answer_check or "").strip().lower()
    return (
        "yes" in check
        or "y" in check
        or (str(model_answer).strip().lower() == str(gt_answer).strip().lower())
    )


def dump_json(path, payload, pretty=True):
    with open(path, "w") as f:
        json.dump(payload, f, indent=4 if pretty else None)
        if pretty:
            f.write("\n")


def save_collection(path, dataset, llm_name, matched_files, collection):
    dump_json(
        path,
        {
            "dataset": dataset,
            "judge_llm": llm_name,
            "source_files": [p.name if isinstance(p, Path) else str(p) for p in matched_files],
            "collection": list(collection.values()),
        },
        pretty=True,
    )


def load_existing_collection(path, dataset, llm_name):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        payload = json.load(f)
    if payload.get("dataset") != dataset or payload.get("judge_llm") != llm_name:
        print(
            f"- Ignoring existing collection at {path} "
            f"(dataset={payload.get('dataset')}, judge_llm={payload.get('judge_llm')})."
        )
        return {}
    cached = {}
    for item in payload.get("collection", []):
        sample_id = str(item.get("ID"))
        answers = item.get("unique_answers", [])
        checks = item.get("llm_answer_check", [])
        for answer, check in zip(answers, checks):
            if check:
                cached[(sample_id, answer)] = check
    return cached


def build_collection(log_files, dataset):
    collection = OrderedDict()
    matched_files = []
    for path in log_files:
        with open(path, "r") as f:
            log_dict = json.load(f)
        if get_dataset_name(log_dict) != dataset:
            continue
        matched_files.append(path)
        fname = path.name
        for sample_id in iter_sample_ids(log_dict):
            sample = get_sample(log_dict, sample_id)
            if not sample or not sample.get("flag_sample_valid", True):
                continue
            if "answer" not in sample or "question" not in sample or "gt_answer" not in sample:
                continue
            answer = sample["answer"]
            entry = collection.get(sample_id)
            if entry is None:
                entry = {
                    "ID": sample_id,
                    "question": sample["question"],
                    "gt_answer": sample["gt_answer"],
                    "unique_answers": [],
                    "llm_answer_check": [],
                    "file_names": [],
                }
                collection[sample_id] = entry
            elif entry["question"] != sample["question"] or str(entry["gt_answer"]) != str(
                sample["gt_answer"]
            ):
                print(
                    f"- Warning: ID {sample_id} in {fname} has a different question/gt_answer "
                    f"than the first seen entry; keeping the first."
                )
            if answer not in entry["unique_answers"]:
                entry["unique_answers"].append(answer)
                entry["file_names"].append([fname])
            else:
                answer_idx = entry["unique_answers"].index(answer)
                if fname not in entry["file_names"][answer_idx]:
                    entry["file_names"][answer_idx].append(fname)
    return matched_files, collection


def is_exact_answer_match(gt_answer, answer):
    return str(answer).strip().lower() == str(gt_answer).strip().lower()


def all_checks_cached(collection, cached_checks):
    for item in collection.values():
        for answer in item["unique_answers"]:
            if is_exact_answer_match(item["gt_answer"], answer):
                continue
            if (str(item["ID"]), answer) not in cached_checks:
                return False
    return True


def fill_checks_from_cache(collection, cached_checks):
    for item in collection.values():
        checks = []
        for answer in item["unique_answers"]:
            if is_exact_answer_match(item["gt_answer"], answer):
                checks.append("yes")
            else:
                checks.append(cached_checks[(str(item["ID"]), answer)])
        item["llm_answer_check"] = checks


def judge_collection(
    collection,
    llm,
    cached_checks=None,
    save_path=None,
    dataset=None,
    llm_name=None,
    matched_files=None,
):
    cached_checks = cached_checks or {}
    judge_cache = dict(cached_checks)
    items = list(collection.values())
    total_unique = sum(len(item["unique_answers"]) for item in items)
    print(f"- Judging {total_unique} unique (ID, answer) pairs with {llm.version}.")
    with tqdm(total=total_unique, desc="LLM judge") as pbar:
        for item in items:
            checks = []
            sample = {"question": item["question"], "gt_answer": item["gt_answer"]}
            for answer in item["unique_answers"]:
                cache_key = (str(item["ID"]), answer)
                semantic_key = (item["question"], str(item["gt_answer"]), answer)
                if is_exact_answer_match(item["gt_answer"], answer):
                    check = "yes"
                elif cache_key in judge_cache:
                    check = judge_cache[cache_key]
                elif semantic_key in judge_cache:
                    check = judge_cache[semantic_key]
                else:
                    _, check = evaluate_answer_correctness_by_llm(llm, sample, answer)
                    judge_cache[cache_key] = check
                    judge_cache[semantic_key] = check
                checks.append(check)
                pbar.update(1)
            item["llm_answer_check"] = checks
            if save_path is not None:
                save_collection(save_path, dataset, llm_name, matched_files or [], collection)
    return collection


def lookup_check(entry, answer):
    try:
        idx = entry["unique_answers"].index(answer)
    except ValueError:
        return None
    checks = entry.get("llm_answer_check") or []
    if idx >= len(checks):
        return None
    return checks[idx]


def recompute_metrics(log_dict, dataset_name):
    is_perturbation_detection = dataset_name in PERTURBATION_DETECTION_DATASETS
    detection_label = (
        "Perturbation detection" if is_perturbation_detection else "Hallucination detection"
    )
    cnt_correct_base = 0
    cnt_correct_detection = 0
    total = 0
    uncertainty_scores = []
    correctness_gt = []

    for sample_id in iter_sample_ids(log_dict):
        sample = get_sample(log_dict, sample_id)
        if not sample or not sample.get("flag_sample_valid", True):
            continue
        if "flag_answer_correct" not in sample or "uncertainty" not in sample:
            continue

        if is_perturbation_detection and "flag_perturbed_inputs" in sample:
            sample["flag_detection_correct"] = bool(sample["flag_perturbed_inputs"]) == bool(
                sample.get("flag_predict_hallucination", False)
            )
        elif "flag_predict_hallucination" in sample:
            flag_answer_correct = bool(sample["flag_answer_correct"])
            flag_predict_hallucination = bool(sample["flag_predict_hallucination"])
            sample["flag_detection_correct"] = (
                flag_answer_correct and not flag_predict_hallucination
            ) or (not flag_answer_correct and flag_predict_hallucination)

        if sample["flag_answer_correct"]:
            cnt_correct_base += 1
        if sample.get("flag_detection_correct"):
            cnt_correct_detection += 1
        total += 1
        uncertainty_scores.append(sample["uncertainty"])
        if is_perturbation_detection:
            correctness_gt.append(bool(sample.get("flag_perturbed_inputs", False)))
        else:
            correctness_gt.append(bool(sample["flag_answer_correct"]))

    if total == 0:
        log_dict["Total samples"] = 0
        return log_dict

    uncertainty_scores = torch.tensor(uncertainty_scores)
    correctness_gt = torch.tensor(correctness_gt, dtype=torch.int)
    detection_gt = correctness_gt if is_perturbation_detection else (1 - correctness_gt)
    auroc_score = auroc(uncertainty_scores, detection_gt, task="binary").item()
    f1_score_result, best_threshold = compute_f1_score(
        uncertainty_scores, detection_gt, seeking=True
    )
    threshold = best_threshold if best_threshold is not None else 1.0
    thresholded_uncertainty_scores = (uncertainty_scores >= threshold).float()
    precision_score = precision(
        thresholded_uncertainty_scores, detection_gt, task="binary"
    ).item()
    recall_score = recall(
        thresholded_uncertainty_scores, detection_gt, task="binary"
    ).item()
    accuracy_score = accuracy(
        thresholded_uncertainty_scores, detection_gt, task="binary"
    ).item()

    log_dict["Base task Accuracy"] = cnt_correct_base / total
    log_dict[f"{detection_label} Accuracy"] = cnt_correct_detection / total
    log_dict[f"{detection_label} AUROC"] = auroc_score
    log_dict[f"{detection_label} F1_Score"] = f1_score_result
    log_dict[f"{detection_label} Found Threshold"] = threshold
    log_dict[f"{detection_label} Precision"] = precision_score
    log_dict[f"{detection_label} Recall"] = recall_score
    log_dict[f"{detection_label} Found Accuracy"] = accuracy_score
    log_dict["Total samples"] = total
    return log_dict


def apply_collection_to_logs(matched_files, collection, output_dir, dataset, llm_name):
    revised_paths = []
    for path in matched_files:
        with open(path, "r") as f:
            raw = f.read()
        pretty = raw.startswith("{\n")
        log_dict = json.loads(raw)
        updated = 0
        for sample_id in iter_sample_ids(log_dict):
            sample = get_sample(log_dict, sample_id)
            if not sample or not sample.get("flag_sample_valid", True):
                continue
            entry = collection.get(str(sample_id))
            if entry is None or "answer" not in sample:
                continue
            check = lookup_check(entry, sample["answer"])
            if check is None:
                continue
            sample["llm_answer_check"] = check
            sample["flag_answer_correct"] = parse_answer_correctness(
                check, sample["answer"], sample.get("gt_answer", entry.get("gt_answer", ""))
            )
            updated += 1
        log_dict["revised_judge_llm"] = llm_name
        log_dict["dataset_name"] = log_dict.get("dataset_name") or dataset
        recompute_metrics(log_dict, dataset)
        revised_path = Path(output_dir) / f"{path.stem}_revised.json"
        dump_json(revised_path, log_dict, pretty=pretty)
        revised_paths.append(revised_path)
        print(f"- Updated {updated} samples: {path.name} -> {revised_path.name}")
    return revised_paths


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset
    llm_name = args.llm

    log_files = list_log_files(output_dir)
    if not log_files:
        raise FileNotFoundError(f"No log_*.json files found in {output_dir}")

    matched_files, collection = build_collection(log_files, dataset)
    if not matched_files:
        raise FileNotFoundError(
            f"No log JSON files with dataset_name={dataset} found in {output_dir}"
        )
    if not collection:
        raise RuntimeError(f"No valid question/answer samples found for dataset={dataset}")

    collection_path = output_dir / f"{dataset}_{llm_name}_answer_collection.json"
    cached_checks = load_existing_collection(collection_path, dataset, llm_name) if args.resume else {}
    if cached_checks and all_checks_cached(collection, cached_checks):
        print("- All unique answers already judged; skipping LLM load.")
        fill_checks_from_cache(collection, cached_checks)
    else:
        llm = obtain_judge_llm(llm_name)
        judge_collection(
            collection,
            llm,
            cached_checks=cached_checks,
            save_path=collection_path,
            dataset=dataset,
            llm_name=llm_name,
            matched_files=matched_files,
        )

    save_collection(collection_path, dataset, llm_name, matched_files, collection)
    print(f"- Answer collection saved at {collection_path}")

    apply_collection_to_logs(matched_files, collection, output_dir, dataset, llm_name)
    print("- Done.")


if __name__ == "__main__":
    main()
