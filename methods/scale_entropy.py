"""Scale Entropy uncertainty estimation.

Computes semantic entropy on the original image and on a half-resolution
(width and height halved) copy, then reports their mean as uncertainty.
"""

from PIL import Image

from methods.vl_uncertainty import (
    hallucination_detection,
    infer_single_sample,
    uncertainty_estimation,
)

_SE_SNAPSHOT_KEYS = (
    "answer_sampling_list",
    "answer_cluster_idx",
    "cluster_dis",
    "entailment",
    "uncertainty",
    "se_clustering",
    "nli_model",
)


def resize_image_half(image):
    """Halve image width and height (at least 1px per side)."""
    w, h = image.size
    new_w = max(1, w // 2)
    new_h = max(1, h // 2)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return image.resize((new_w, new_h), resample)


def _snapshot_se_fields(entry):
    return {key: entry[key] for key in _SE_SNAPSHOT_KEYS if key in entry}


def _sample_and_estimate_entropy(args, lvlm, sample, llm, log_dict):
    log_dict[sample["idx"]]["answer_sampling_list"] = []
    for _ in range(args.sampling_time):
        infer_single_sample(args, lvlm, sample, True, llm, log_dict)
    uncertainty_estimation(args, sample, llm, log_dict)
    return float(log_dict[sample["idx"]]["uncertainty"])


def scale_entropy(args, lvlm, sample, llm, log_dict):
    # Semantic entropy at original resolution.
    se_original = _sample_and_estimate_entropy(args, lvlm, sample, llm, log_dict)
    snap_original = _snapshot_se_fields(log_dict[sample["idx"]])

    # Semantic entropy at half resolution.
    scaled_sample = sample.copy()
    scaled_img = resize_image_half(sample["img"])
    scaled_sample["img"] = scaled_img
    log_dict[sample["idx"]]["scale_entropy_original_size"] = list(sample["img"].size)
    log_dict[sample["idx"]]["scale_entropy_half_size"] = list(scaled_img.size)

    se_scaled = _sample_and_estimate_entropy(args, lvlm, scaled_sample, llm, log_dict)
    snap_scaled = _snapshot_se_fields(log_dict[sample["idx"]])

    # Keep original SE fields as the primary sampling log; store scaled separately.
    for key, value in snap_original.items():
        log_dict[sample["idx"]][key] = value
    log_dict[sample["idx"]]["answer_sampling_list_scaled"] = snap_scaled.get(
        "answer_sampling_list", []
    )
    log_dict[sample["idx"]]["answer_cluster_idx_scaled"] = snap_scaled.get(
        "answer_cluster_idx", []
    )
    log_dict[sample["idx"]]["cluster_dis_scaled"] = snap_scaled.get("cluster_dis", {})
    if "entailment" in snap_scaled:
        log_dict[sample["idx"]]["entailment_scaled"] = snap_scaled["entailment"]

    uncertainty = 0.5 * (se_original + se_scaled)
    log_dict[sample["idx"]]["uncertainty_original"] = se_original
    log_dict[sample["idx"]]["uncertainty_scaled"] = se_scaled
    log_dict[sample["idx"]]["uncertainty"] = float(uncertainty)
    hallucination_detection(args, sample, log_dict)


def estimate_uncertainty_by_scale_entropy(args, lvlm, sample, llm, log_dict):
    infer_single_sample(args, lvlm, sample, False, llm, log_dict)
    scale_entropy(args, lvlm, sample, llm, log_dict)
    return log_dict
