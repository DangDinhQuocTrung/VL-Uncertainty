import torch
import torch.nn.functional as F
from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm, evaluate_multiple_choice_answer_correctness
from utils.constants import is_choice_question


# Modes that use chosen-token NLL (Manakul et al. / clinical VQA Max-NLL paper).
NLL_MODES = ("avg", "average", "anll", "max", "mnll")
# VSE paper logit baselines (Kostumov et al. EMNLP'24-style token entropy / probability).
ENT_MODES = ("avg_ent", "avgent", "max_ent", "maxent")
PROB_MODES = ("avg_prob", "avgprob", "max_prob", "maxprob")
ALL_LOGIT_MODES = NLL_MODES + ENT_MODES + PROB_MODES


def compute_token_logit_stats(inputs, outputs):
    """Per-generated-token NLLs, full-vocab entropies, and chosen-token probs."""
    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs["sequences"][0, prompt_len:]
    num_generated = min(len(outputs["scores"]), generated_ids.shape[0])

    token_nlls = []
    token_ents = []
    token_probs = []
    for step_idx in range(num_generated):
        step_logits = outputs["scores"][step_idx]
        log_probs = F.log_softmax(step_logits, dim=-1)
        probs = log_probs.exp()
        token_id = generated_ids[step_idx]
        token_nlls.append(-log_probs[0, token_id])
        token_probs.append(probs[0, token_id])
        # Shannon entropy of the full next-token distribution (nats).
        token_ents.append(-(probs * log_probs).nan_to_num(0.0).sum(dim=-1)[0])

    if not token_nlls:
        empty = torch.zeros(0)
        return empty, empty, empty

    return torch.stack(token_nlls), torch.stack(token_ents), torch.stack(token_probs)


def compute_token_nlls(inputs, outputs):
    """Compute per-token negative log-likelihoods of the generated answer."""
    token_nlls, _, _ = compute_token_logit_stats(inputs, outputs)
    return token_nlls


def aggregate_logit_uncertainty(token_nlls, token_ents, token_probs, mode="avg"):
    """Aggregate token stats into a scalar uncertainty (higher => more uncertain)."""
    if token_nlls.numel() == 0:
        zeros = {
            "avg_nll": 0.0,
            "max_nll": 0.0,
            "avg_ent": 0.0,
            "max_ent": 0.0,
            "avg_prob": 0.0,
            "max_prob": 0.0,
            "min_prob": 0.0,
        }
        return 0.0, zeros

    avg_nll = token_nlls.mean().item()
    max_nll = token_nlls.max().item()
    avg_ent = token_ents.mean().item()
    max_ent = token_ents.max().item()
    avg_prob = token_probs.mean().item()
    max_prob = token_probs.max().item()
    min_prob = token_probs.min().item()

    stats = {
        "avg_nll": avg_nll,
        "max_nll": max_nll,
        "avg_ent": avg_ent,
        "max_ent": max_ent,
        "avg_prob": avg_prob,
        "max_prob": max_prob,
        "min_prob": min_prob,
    }

    mode = mode.lower()
    if mode in ("avg", "average", "anll"):
        uncertainty = avg_nll
    elif mode in ("max", "mnll"):
        uncertainty = max_nll
    elif mode in ("avg_ent", "avgent"):
        uncertainty = avg_ent
    elif mode in ("max_ent", "maxent"):
        uncertainty = max_ent
    elif mode in ("avg_prob", "avgprob"):
        # Prob is confidence; invert so higher score = more uncertain (AUROC convention).
        uncertainty = 1.0 - avg_prob
    elif mode in ("max_prob", "maxprob"):
        uncertainty = 1.0 - max_prob
    else:
        raise ValueError(
            f"Unsupported logit mode: {mode}. Expected one of: {', '.join(ALL_LOGIT_MODES)}."
        )
    return uncertainty, stats


def compute_nll(inputs, outputs, mode="avg"):
    """Backward-compatible NLL aggregation; also returns avg/max NLL."""
    token_nlls, token_ents, token_probs = compute_token_logit_stats(inputs, outputs)
    uncertainty, stats = aggregate_logit_uncertainty(token_nlls, token_ents, token_probs, mode=mode)
    return uncertainty, stats["avg_nll"], stats["max_nll"]


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

    # Logit-based uncertainty: NLL (clinical VQA paper) + Ent/Prob (VSE baselines).
    nll_mode = getattr(args, "nll_mode", "avg")
    token_nlls, token_ents, token_probs = compute_token_logit_stats(inputs, outputs)
    uncertainty, stats = aggregate_logit_uncertainty(
        token_nlls, token_ents, token_probs, mode=nll_mode
    )

    for key, value in stats.items():
        log_dict[sample["idx"]][key] = value
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
