"""Radial Dispersion Score (RDS) uncertainty estimation.

Paper: Distance Is All You Need (arXiv:2512.04351)
Repo: https://github.com/manhitv/RDS

Modes:
  - eigenembed: EigenEmbed / EigenScore on external embeddings (repo SVD variant)
  - base: RDS = sum_i ||u_i - mean||_1
  - weighted: RDSw = sum_i p_i ||u_i - mean_w||_1, p_i ∝ exp(-ANLL_i)
"""

import numpy as np
import torch
import torch.nn.functional as F

from methods.beam_utils import generate_greedy_answer, generate_temperature_samples
from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm, evaluate_multiple_choice_answer_correctness
from utils.constants import is_choice_question

_EMBED_MODEL = None


def _get_embed_model(model_name, device):
    global _EMBED_MODEL
    if _EMBED_MODEL is not None and getattr(_EMBED_MODEL, "_rds_name", None) == model_name:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "RDS requires the sentence-transformers package. "
            "Install with: pip install sentence-transformers"
        ) from exc
    model = SentenceTransformer(model_name, device=str(device))
    model._rds_name = model_name
    _EMBED_MODEL = model
    return model


def embed_answers(answers, model_name="all-MiniLM-L6-v2", device=None):
    """Embed answers and L2-normalize onto the unit hypersphere."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = _get_embed_model(model_name, device)
    texts = [a if (a is not None and str(a).strip()) else " " for a in answers]
    embeddings = embed_model.encode(texts, convert_to_tensor=True, device=device)
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    return embeddings


def compute_eigen_embed(embeddings, alpha=1e-3):
    """EigenEmbed as in the RDS repo (mean log10 singular values of sample covariance)."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected (N, D) embeddings, got {embeddings.shape}")
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    cov = np.cov(embeddings)  # (N, N)
    cov = cov + alpha * np.eye(n)
    _, singular_values, _ = np.linalg.svd(cov)
    singular_values = np.sort(singular_values)[::-1]
    singular_values = np.clip(singular_values, 1e-12, None)
    return float(np.mean(np.log10(singular_values)))


def compute_rds_base(embeddings):
    """RDS = sum_i ||u_i - mean||_1."""
    mean_embedding = embeddings.mean(dim=0)
    diffs = embeddings - mean_embedding
    return torch.norm(diffs, p=1, dim=1).sum().item()


def compute_rds_weighted(embeddings, avg_nlls):
    """RDSw with p_i ∝ exp(-ANLL_i)."""
    probs = np.exp(-np.asarray(avg_nlls, dtype=np.float64))
    probs = probs / max(probs.sum(), 1e-12)
    probs_t = torch.tensor(probs, dtype=embeddings.dtype, device=embeddings.device)
    weighted_mean = (probs_t.unsqueeze(1) * embeddings).sum(dim=0)
    diffs = embeddings - weighted_mean.unsqueeze(0)
    distances = torch.norm(diffs, p=1, dim=1)
    return float((probs_t * distances).sum().item()), probs.tolist()


def compute_rds_score(embeddings, avg_nlls, mode="base"):
    mode = mode.lower()
    if mode in ("eigenembed", "ee", "eigen"):
        return compute_eigen_embed(embeddings), None
    if mode in ("base", "rds"):
        return compute_rds_base(embeddings), None
    if mode in ("weighted", "rdsw", "rds_w"):
        score, probs = compute_rds_weighted(embeddings, avg_nlls)
        return score, probs
    raise ValueError(
        f"Unsupported RDS mode: {mode}. Expected one of: eigenembed, base, weighted."
    )


def estimate_uncertainty_by_rds(args, lvlm, sample, llm, log_dict):
    rds_mode = getattr(args, "rds_mode", "base")
    embed_model_name = getattr(args, "rds_embed_model", "all-MiniLM-L6-v2")

    # Main answer: greedy / inference-temp (aligned with VAUQ).
    answer = generate_greedy_answer(args, lvlm, sample)
    # Uncertainty: temperature samples only (RDS paper).
    samples = generate_temperature_samples(args, lvlm, sample)
    answers = samples["answers"]
    avg_nlls = samples["avg_nlls"]

    log_dict[sample["idx"]]["answer"] = answer
    log_dict[sample["idx"]]["answer_sampling_list"] = answers
    log_dict[sample["idx"]]["sample_nlls"] = samples["nlls"]
    log_dict[sample["idx"]]["sample_avg_nlls"] = avg_nlls
    log_dict[sample["idx"]]["sample_probs"] = samples["probs"]
    log_dict[sample["idx"]]["rds_n_samples"] = samples["n_samples"]
    log_dict[sample["idx"]]["rds_sampling_temp"] = samples["sampling_temp"]

    flag_answer_correct = True
    if is_choice_question(args, sample):
        flag_answer_correct, llm_answer_check = evaluate_multiple_choice_answer_correctness(llm, sample, answer)
    else:
        flag_answer_correct, llm_answer_check = evaluate_answer_correctness_by_llm(llm, sample, answer)
    log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
    log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct

    device = getattr(lvlm, "device", None)
    embeddings = embed_answers(answers, model_name=embed_model_name, device=device)

    # Log all three variants for analysis; selected mode drives uncertainty.
    eigenembed = compute_eigen_embed(embeddings)
    rds_base = compute_rds_base(embeddings)
    rds_weighted, weight_probs = compute_rds_weighted(embeddings, avg_nlls)

    uncertainty, _ = compute_rds_score(embeddings, avg_nlls, mode=rds_mode)

    log_dict[sample["idx"]]["rds_mode"] = rds_mode
    log_dict[sample["idx"]]["rds_embed_model"] = embed_model_name
    log_dict[sample["idx"]]["eigenembed"] = eigenembed
    log_dict[sample["idx"]]["rds_base"] = rds_base
    log_dict[sample["idx"]]["rds_weighted"] = rds_weighted
    log_dict[sample["idx"]]["rds_weight_probs"] = weight_probs
    log_dict[sample["idx"]]["uncertainty"] = uncertainty
    log_dict[sample["idx"]]["uncertainty_threshold"] = args.uncertainty_threshold

    flag_predict_hallucination = uncertainty >= args.uncertainty_threshold
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        flag_answer_correct and not flag_predict_hallucination
    ) or (not flag_answer_correct and flag_predict_hallucination)
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct
    return log_dict
