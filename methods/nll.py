import torch
import torch.nn.functional as F
from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm, evaluate_multiple_choice_answer_correctness
from utils.constants import is_choice_question


def compute_token_nlls(inputs, outputs):
    """Compute per-token negative log-likelihoods of the generated answer."""
    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs["sequences"][0, prompt_len:]
    num_generated = min(len(outputs["scores"]), generated_ids.shape[0])

    token_nlls = []
    for step_idx in range(num_generated):
        step_logits = outputs["scores"][step_idx]
        log_probs = F.log_softmax(step_logits, dim=-1)
        token_id = generated_ids[step_idx]
        token_nlls.append(-log_probs[0, token_id])

    if not token_nlls:
        return torch.zeros(0)

    return torch.stack(token_nlls)


def compute_nll(inputs, outputs, mode="avg"):
    token_nlls = compute_token_nlls(inputs, outputs)
    if token_nlls.numel() == 0:
        return 0.0, 0.0, 0.0

    avg_nll = token_nlls.mean().item()
    max_nll = token_nlls.max().item()

    mode = mode.lower()
    if mode in ("avg", "average", "anll"):
        uncertainty = avg_nll
    elif mode in ("max", "mnll"):
        uncertainty = max_nll
    else:
        raise ValueError(
            f"Unsupported NLL mode: {mode}. Expected one of: avg, average, max."
        )
    return uncertainty, avg_nll, max_nll


def estimate_uncertainty_by_nll(args, lvlm, sample, llm, log_dict):
    # Generate answer
    answer, inputs, outputs, _answers = lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp,
        return_more=True,
        return_mode=1,
    )
    log_dict[sample["idx"]]["answer"] = answer
    flag_answer_correct = True
    if is_choice_question(args, sample):
        flag_answer_correct, llm_answer_check = evaluate_multiple_choice_answer_correctness(llm, sample, answer)
    else:
        flag_answer_correct, llm_answer_check = evaluate_answer_correctness_by_llm(llm, sample, answer)
    log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
    log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct
    log_dict[sample["idx"]]["answer_sampling_list"] = [answer]

    # Compute negative log-likelihood of the answer
    nll_mode = getattr(args, "nll_mode", "avg")
    uncertainty, avg_nll, max_nll = compute_nll(inputs, outputs, mode=nll_mode)

    log_dict[sample["idx"]]["avg_nll"] = avg_nll
    log_dict[sample["idx"]]["max_nll"] = max_nll
    log_dict[sample["idx"]]["nll_mode"] = nll_mode
    log_dict[sample["idx"]]["uncertainty"] = uncertainty
    log_dict[sample["idx"]]["uncertainty_threshold"] = args.uncertainty_threshold
    flag_predict_hallucination = log_dict[sample["idx"]]["uncertainty"] >= args.uncertainty_threshold
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        log_dict[sample["idx"]]["flag_answer_correct"] and not flag_predict_hallucination
    ) or (
        not log_dict[sample["idx"]]["flag_answer_correct"] and flag_predict_hallucination
    )
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct
    return log_dict
