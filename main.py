import argparse
import collections
import json
import math
import os
import random
import re
import warnings

import numpy as np
import torch
import gc
from tqdm import tqdm
from PIL import Image
from torchmetrics.functional import auroc, precision, recall, accuracy

from utils.constants import *
from methods.vl_uncertainty import *
from lvlm.model_manager import LLaVAModelManager
from methods.svar.svar import estimate_uncertainty_by_svar
from methods.euq.euq import estimate_uncertainty_by_euq
from methods.vauq import estimate_uncertainty_by_vauq
from utils.metrics import compute_f1_score

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick_benchmark", type=lambda x: x.lower() == "true", default="True")
    parser.add_argument("--use_fastest", type=lambda x: x.lower() == "true", default="False")
    parser.add_argument("--lvlm", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--use_model_manager", type=lambda x: x.lower() == "true", default="False")
    parser.add_argument("--benchmark", type=str, default="ViLP")
    parser.add_argument("--llm", type=str, default="Qwen2.5-3B-Instruct")
    parser.add_argument("--uncertainty", type=str, default="vauq")
    parser.add_argument("--uncertainty_threshold", type=float, default=1.0)

    # Perturbation-specific arguments
    parser.add_argument("--visual_perturbation", type=str, default="blurring")
    parser.add_argument(
        "--blur_radius_list", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4]
    )
    parser.add_argument("--textual_perturbation", type=str, default="llm_rephrasing")
    parser.add_argument(
        "--textual_perturbation_temp_list",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    parser.add_argument(
        "--textual_perturbation_instruction_template",
        type=str,
        default="Given the input question: '{question}', generate a semantically equivalent variation by changing the wording, structure, grammar, or narrative. Ensure the perturbed question maintains the same meaning as the original. Provide only the rephrased question as the output.",
    )
    parser.add_argument("--pair_order", type=str, default="progressively")

    # Sampling-specific arguments
    parser.add_argument("--inference_temp", type=float, default=0.0)
    parser.add_argument("--sampling_temp", type=float, default=1.0)
    parser.add_argument("--sampling_time", type=int, default=5)
    args = parser.parse_args()
    print(vars(args))
    return args


def obtain_lvlm(args):
    if args.use_model_manager:
        return LLaVAModelManager(args.lvlm, args.use_fastest)
    else:
        lvlm_class = LVLM_MAP.get(args.lvlm)
    if not lvlm_class:
        raise ValueError(f"Unsupported LVLM: {args.lvlm}")
    use_flash_attention = False if args.uncertainty in ["vauq"] else True
    return lvlm_class(args.lvlm, use_fastest=args.use_fastest, use_flash_attention=use_flash_attention)


def obtain_benchmark(args):
    benchmark_class = BENCHMARK_MAP.get(args.benchmark)
    if not benchmark_class:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")
    return benchmark_class()


def obtain_llm(args):
    llm_class = LLM_MAP.get(args.llm)
    if not llm_class:
        raise ValueError(f"Unsupported LLM: {args.llm}")
    return llm_class(args.llm)


def obtain_single_sample(args, benchmark, idx, log_dict):
    sample = benchmark.retrieve(idx)
    log_dict[idx]["question"] = sample["question"]
    log_dict[idx]["gt_answer"] = sample["gt_answer"]
    return sample


def handle_single(args, idx, lvlm, benchmark, llm, log_dict):
    sample = obtain_single_sample(args, benchmark, idx, log_dict)
    if (
        sample is None
        or sample["img"] is None
        or sample["question"] is None
        or sample["gt_answer"] is None
    ):
        log_dict[idx]["flag_sample_valid"] = False
        return
    log_dict[idx]["flag_sample_valid"] = True

    # Log images
    # image = np.array(sample["img"])
    # print("Image:", image.shape, image.dtype, image.min(), image.max())
    # output_dir = "/work3/dida/outputs_LVLM/VL"
    # Image.fromarray(image).save(os.path.join(output_dir, f"{args.benchmark}_{idx}.png"))
    # with open(os.path.join(output_dir, f"{args.benchmark}_{idx}.json"), "w") as f:
    #     no_image_sample = sample.copy()
    #     no_image_sample.pop("img")
    #     json.dump(no_image_sample, f, indent=4)

    # Inference
    if args.uncertainty in BLACK_BOX_METHODS:
        estimate_uncertainty_by_vl_or_semantic_entropy(args, lvlm, sample, llm, log_dict)
    elif args.uncertainty == "svar":
        estimate_uncertainty_by_svar(args, lvlm, sample, llm, log_dict)
    elif args.uncertainty == "euq":
        estimate_uncertainty_by_euq(args, lvlm, sample, llm, log_dict)
    elif args.uncertainty == "vauq":
        estimate_uncertainty_by_vauq(args, lvlm, sample, llm, log_dict)
    else:
        raise ValueError(f"Unsupported method: {args.uncertainty}")
    return


def handle_batch(args, lvlm, benchmark, llm):
    log_dict = {}
    log_dict["args"] = str(args)
    begin_time_str = get_cur_time()
    log_dict["begin_time_str"] = begin_time_str

    total = 0
    cnt_correct_base = 0
    cnt_correct_detection = 0
    uncertainty_scores = []
    correctness_gt = []
    benchmark_size = benchmark.obtain_size()
    print(f"Benchmark size: {benchmark_size}")
    if args.quick_benchmark:
        benchmark_size = min(benchmark_size, 5)

    # Run the benchmark
    split_inference_quantification = 1 if args.uncertainty in ["euq"] else 0
    args.split_inference_quantification = split_inference_quantification
    for idx in tqdm(range(benchmark_size)):
        log_dict[idx] = {}
        handle_single(args, idx, lvlm, benchmark, llm, log_dict)
        if not log_dict[idx]["flag_sample_valid"]:
            continue
        if log_dict[idx]["flag_answer_correct"]:
            cnt_correct_base += 1
        if split_inference_quantification:
            continue
        if log_dict[idx]["flag_detection_correct"]:
            cnt_correct_detection += 1
        total += 1
        uncertainty_scores.append(log_dict[idx]["uncertainty"])
        correctness_gt.append(log_dict[idx]["flag_answer_correct"])
    # Run again after removing the LVLM
    args.split_inference_quantification = 2
    if split_inference_quantification:
        if not args.use_fastest:
            lvlm.model.to("cpu")
        del lvlm
        gc.collect()
        torch.cuda.empty_cache()
        lvlm = None
        for idx in tqdm(range(benchmark_size)):
            handle_single(args, idx, lvlm, benchmark, llm, log_dict)
            if not log_dict[idx]["flag_sample_valid"]:
                continue
            if log_dict[idx]["flag_detection_correct"]:
                cnt_correct_detection += 1
            total += 1
            uncertainty_scores.append(log_dict[idx]["uncertainty"])
            correctness_gt.append(log_dict[idx]["flag_answer_correct"])

    # Compute metrics
    uncertainty_scores = torch.tensor(uncertainty_scores)
    correctness_gt = torch.tensor(correctness_gt, dtype=torch.int)
    incorrectness_gt = 1 - correctness_gt
    auroc_score = auroc(uncertainty_scores, incorrectness_gt, task="binary").item()
    f1_score_result, best_threshold = compute_f1_score(uncertainty_scores, incorrectness_gt, seeking=True)
    threshold = best_threshold if best_threshold is not None else args.uncertainty_threshold
    thresholded_uncertainty_scores = (uncertainty_scores >= threshold).float()
    precision_score = precision(thresholded_uncertainty_scores, incorrectness_gt, task="binary").item()
    recall_score = recall(thresholded_uncertainty_scores, incorrectness_gt, task="binary").item()
    accuracy_score = accuracy(thresholded_uncertainty_scores, incorrectness_gt, task="binary").item()

    # Log metrics
    log_dict["Base task Accuracy"] = (cnt_correct_base / total)
    log_dict["Hallucination detection Accuracy"] = (cnt_correct_detection / total)
    log_dict["Hallucination detection AUROC"] = auroc_score
    log_dict["Hallucination detection F1_Score"] = f1_score_result
    log_dict["Hallucination detection Found Threshold"] = threshold
    log_dict["Hallucination detection Precision"] = precision_score
    log_dict["Hallucination detection Recall"] = recall_score
    log_dict["Hallucination detection Found Accuracy"] = accuracy_score
    log_dict["Total samples"] = total
    end_time_str = get_cur_time()
    log_dict["end_time_str"] = end_time_str
    if not os.path.exists("exp"):
        os.makedirs("exp")
    with open(f"exp/log_{begin_time_str}.json", "w") as f:
        json.dump(log_dict, f, indent=4 if args.quick_benchmark else None)
    print(f"- Full log is saved at exp/log_dict_{begin_time_str}.json.")


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    fix_seed(args.seed)
    lvlm = obtain_lvlm(args)
    benchmark = obtain_benchmark(args)
    llm = obtain_llm(args)
    handle_batch(args, lvlm, benchmark, llm)


if __name__ == "__main__":
    main()
