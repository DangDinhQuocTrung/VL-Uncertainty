import math

import numpy as np

from methods.beam_utils import generate_beam_candidates, generate_greedy_answer
from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm
from utils.constants import is_choice_question


def compute_pro_score(probs, alpha=0.4):
    """Approximate predictive entropy from top sequence probabilities (PRO).

    PRO(x) = -log(p_K) - sum_i p_i log(p_i / p_K),
    where p_i are sequence probabilities with p_i >= alpha, sorted descending.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[np.isfinite(probs) & (probs > 0.0)]
    if probs.size == 0:
        return 0.0, 0

    top_probs = np.sort(probs)[::-1]
    selected = top_probs[top_probs >= alpha]
    if selected.size < 1:
        selected = top_probs[:1]

    pk = selected[-1]
    # Final term is zero when i == K, matching the official PRO implementation.
    score = -math.log(pk) - float(np.sum(selected[:-1] * np.log(selected[:-1] / pk)))
    return score, int(selected.size)


def estimate_uncertainty_by_pro(args, lvlm, sample, llm, log_dict):
    pro_alpha = getattr(args, "pro_alpha", 0.4)

    # Main answer: greedy / inference-temp (aligned with VAUQ).
    answer = generate_greedy_answer(args, lvlm, sample)
    # Uncertainty: beam candidates only.
    beam = generate_beam_candidates(args, lvlm, sample)

    log_dict[sample["idx"]]["answer"] = answer
    log_dict[sample["idx"]]["answer_sampling_list"] = beam["answers"]
    log_dict[sample["idx"]]["beam_nlls"] = beam["nlls"]
    log_dict[sample["idx"]]["beam_probs"] = beam["probs"]

    flag_answer_correct = True
    if is_choice_question(args, sample):
        flag_answer_correct = str(sample["gt_answer"]) in answer
    else:
        flag_answer_correct, llm_answer_check = evaluate_answer_correctness_by_llm(
            llm, sample, answer
        )
        log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
    log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct

    uncertainty, selected_k = compute_pro_score(beam["probs"], alpha=pro_alpha)
    log_dict[sample["idx"]]["pro_alpha"] = pro_alpha
    log_dict[sample["idx"]]["pro_selected_k"] = selected_k
    log_dict[sample["idx"]]["top1_nll"] = beam["nlls"][beam["best_idx"]]
    log_dict[sample["idx"]]["uncertainty"] = uncertainty
    log_dict[sample["idx"]]["uncertainty_threshold"] = args.uncertainty_threshold

    flag_predict_hallucination = uncertainty >= args.uncertainty_threshold
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        flag_answer_correct and not flag_predict_hallucination
    ) or (not flag_answer_correct and flag_predict_hallucination)
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct
    return log_dict
