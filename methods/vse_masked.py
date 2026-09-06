"""VSE-Masked: Visual Semantic Entropy with progressive visual-token black-out.

Same aggregation as VSE (``methods/vse.py``), but image views are built by
blacking out an increasing fraction of top-attended visual tokens instead of
adding Gaussian noise.
"""

import torch

from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm
from methods.vauq import build_masked_image
from methods.vauq_utils import compute_attention_over_visual_tokens
from methods.vse import infer_single_sample, prototype_semantic_aggregation, hallucination_detection
from utils.constants import is_choice_question


def perturbation_of_visual_prompt_vse_masked(args, lvlm, sample, inputs, outputs):
    """Progressively black out top-attended visual tokens (e.g. 10/20/30/40%)."""
    device = getattr(lvlm, "device", None)
    (
        _image_token_id,
        visual_token_positions,
        _start,
        _end,
        sum_attention_over_visual_tokens,
        _,
    ) = compute_attention_over_visual_tokens(
        lvlm.model, lvlm.processor, inputs, outputs, args.lvlm, device
    )

    percents = list(getattr(args, "vse_mask_percents", [10, 20, 30, 40]))
    max_percent = max(percents)
    n_tokens = int(visual_token_positions.shape[0])
    if n_tokens == 0:
        raise RuntimeError("No visual tokens found for vse_masked perturbation.")

    ranked = torch.argsort(sum_attention_over_visual_tokens, descending=True)
    perturbed_img_list = []
    mask_counts = []
    for percent in percents:
        k = max(1, int(round(float(percent) / 100.0 * n_tokens)))
        k = min(k, n_tokens)
        max_k = min(k, int(round(float(max_percent) / 100.0 * n_tokens)))
        start_k = max_k - k
        # positions_to_zero = visual_token_positions[ranked[:k]]
        positions_to_zero = visual_token_positions[ranked[start_k:max_k]]
        masked_image, _grid_h, _grid_w, masked_indices = build_masked_image(
            lvlm,
            sample["img"],
            inputs,
            visual_token_positions,
            positions_to_zero,
        )
        perturbed_img_list.append(masked_image)
        mask_counts.append(len(masked_indices))

    return perturbed_img_list, percents, mask_counts, n_tokens


def _log_clean_answer(args, sample, llm, log_dict, answer):
    log_dict[sample["idx"]]["answer"] = answer
    flag_answer_correct = True
    if is_choice_question(args, sample):
        flag_answer_correct = str(sample["gt_answer"]) in answer
    else:
        flag_answer_correct, llm_answer_check = evaluate_answer_correctness_by_llm(
            llm, sample, answer
        )
        log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
    log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct


def visual_semantic_entropy(args, lvlm, sample, llm, log_dict, perturbed_img_list):
    log_dict[sample["idx"]]["vse_num_views"] = len(perturbed_img_list)

    log_dict[sample["idx"]]["answer_sampling_list"] = []
    for perturbed_img in perturbed_img_list:
        perturbed_sample = sample.copy()
        perturbed_sample["img"] = perturbed_img
        infer_single_sample(args, lvlm, perturbed_sample, True, llm, log_dict)

    device = getattr(lvlm, "device", None)
    prototype_semantic_aggregation(args, sample, log_dict, device=device)
    hallucination_detection(args, sample, log_dict)


def estimate_uncertainty_by_vse_masked(args, lvlm, sample, llm, log_dict):
    """VSE with progressive top-token black-out instead of Gaussian noise."""
    answer, inputs, outputs, _answers = lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp,
        return_more=True,
        return_mode=2,
    )
    _log_clean_answer(args, sample, llm, log_dict, answer)

    perturbed_img_list, percents, mask_counts, n_tokens = (
        perturbation_of_visual_prompt_vse_masked(args, lvlm, sample, inputs, outputs)
    )
    log_dict[sample["idx"]]["vse_mask_percents"] = [float(p) for p in percents]
    log_dict[sample["idx"]]["vse_mask_token_counts"] = mask_counts
    log_dict[sample["idx"]]["n_visual_tokens"] = int(n_tokens)

    visual_semantic_entropy(
        args, lvlm, sample, llm, log_dict, perturbed_img_list=perturbed_img_list,
    )
    return log_dict
