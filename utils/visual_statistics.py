"""Visual statistics for interpretability analysis.

- Semantic entropy path: LogitLens visual entropy H_vis (VSE Sec. 3.2).
- EUQ path: mean head conflict / ignorance over visual tokens.
- VAUQ path: both of the above, computed on the VAUQ-masked visual input.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from methods.euq.evidence import EvidenceModel
from utils.model_utils import get_lm_head, resolve_image_token_id


def _token_entropy_from_logits(logits):
    """Shannon entropy (nats) of each row in logits (N, V)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _normalize_model_type(model_type):
    name = str(model_type).lower()
    if "llava" in name:
        return "llava"
    if "qwen2.5" in name:
        return "qwen2.5"
    if "qwen" in name:
        return "qwen"
    if "gemma" in name:
        return "gemma"
    raise ValueError(f"Unsupported model: {model_type}")


def _get_layer0(lvlm, model_type):
    """Resolve layer 0 on the LVLM wrapper (same paths as methods/vauq.py)."""
    model_type = _normalize_model_type(model_type)
    if model_type == "llava":
        return lvlm.model.language_model.model.layers[0]
    if model_type == "qwen2.5":
        return lvlm.model.model.layers[0]
    if model_type == "qwen":
        return lvlm.model.model.language_model.layers[0]
    if model_type == "gemma":
        return lvlm.model.language_model.model.layers[0]
    raise ValueError(f"Unsupported model: {model_type}")


def _register_visual_mask_hook(lvlm, model_type, positions_to_zero):
    """Zero selected visual-token positions at the input to layer 0 (VAUQ masking)."""
    if positions_to_zero is None:
        return None
    positions_to_zero = positions_to_zero.to(dtype=torch.long)
    if positions_to_zero.numel() == 0:
        return None

    def pre_hook(module, input_):
        hidden_states = input_[0]
        pos = positions_to_zero.to(hidden_states.device)
        if hidden_states.shape[1] > int(pos.max().item()):
            hidden_states[:, pos, :] = 0.0
        return (hidden_states,) + input_[1:]

    layer_0 = _get_layer0(lvlm, model_type)
    return layer_0.register_forward_pre_hook(pre_hook)


@torch.no_grad()
def _forward_visual_hidden(
    lvlm,
    image,
    question,
    positions_to_zero=None,
    model_type=None,
):
    """Final-layer hidden states at visual token positions (optionally VAUQ-masked)."""
    if not hasattr(lvlm, "prepare_inputs"):
        raise ValueError(
            "visual statistics require lvlm.prepare_inputs(image, question). "
            f"Got {type(lvlm).__name__} ({getattr(lvlm, 'version', None)})."
        )

    inputs = lvlm.prepare_inputs(image, question)
    image_token_id = resolve_image_token_id(lvlm)
    visual_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    if visual_positions.numel() == 0:
        raise RuntimeError(
            f"No visual tokens found for image_token_id={image_token_id}."
        )

    handle = None
    if positions_to_zero is not None:
        if model_type is None:
            raise ValueError(
                "model_type is required when applying VAUQ positions_to_zero masking."
            )
        handle = _register_visual_mask_hook(lvlm, model_type, positions_to_zero)
    try:
        outputs = lvlm.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    finally:
        if handle is not None:
            handle.remove()

    # HF model outputs typically expose the final hidden state after the model's
    # last normalization step, so applying the final norm again would distort
    # the LogitLens ranking.
    hidden = outputs.hidden_states[-1][0, visual_positions]
    return hidden, visual_positions


def _compute_entropy_from_hidden(hidden, lm_head, chunk_size=256):
    entropies = []
    for start in range(0, hidden.shape[0], chunk_size):
        chunk = hidden[start : start + chunk_size]
        ent = _token_entropy_from_logits(lm_head(chunk))
        entropies.append(ent.detach().float().cpu())
    token_entropies = torch.cat(entropies, dim=0)
    return {
        "visual_entropy": float(token_entropies.mean().item()),
        "n_visual_tokens": int(hidden.shape[0]),
        "visual_token_entropies": [float(x) for x in token_entropies.tolist()],
    }


def _compute_euq_from_hidden(hidden, weight_dir, lvlm_version, device):
    weight_dir = Path(weight_dir)
    head_state_dict = torch.load(
        weight_dir / f"{lvlm_version}_head_weights.pth",
        map_location=device,
    )
    head_evidence_model = EvidenceModel(head_state_dict)

    conflict_sum = 0.0
    ig_sum = 0.0
    n_tokens = hidden.shape[0]
    conflict_values = []
    ig_values = []
    for i in range(n_tokens):
        feature = hidden[i : i + 1]
        head_evidence_model.get_evidence_weights(feature.T)
        conflict = head_evidence_model.get_evidence_conflict().item()
        ig = head_evidence_model.get_evidence_ignorance().item()
        conflict_sum += conflict
        ig_sum += ig
        conflict_values.append(float(conflict))
        ig_values.append(float(ig))

    return {
        "visual_mean_head_conflict_value": conflict_sum / n_tokens,
        "visual_mean_head_ignorance_value": ig_sum / n_tokens,
        "n_visual_tokens": int(n_tokens),
        "visual_token_head_conflict_values": conflict_values,
        "visual_token_head_ignorance_values": ig_values,
    }


@torch.no_grad()
def compute_visual_entropy(
    lvlm,
    image,
    question,
    chunk_size=256,
    positions_to_zero=None,
    model_type=None,
):
    """Average LogitLens entropy over final-layer visual tokens."""
    hidden, visual_positions = _forward_visual_hidden(
        lvlm,
        image,
        question,
        positions_to_zero=positions_to_zero,
        model_type=model_type,
    )
    lm_head = get_lm_head(lvlm.model)
    result = _compute_entropy_from_hidden(hidden, lm_head, chunk_size=chunk_size)
    result["n_visual_tokens"] = int(visual_positions.numel())
    return result


@torch.no_grad()
def compute_visual_euq_head_statistics(
    lvlm,
    image,
    question,
    weight_dir,
    lvlm_version,
    device=None,
    positions_to_zero=None,
    model_type=None,
):
    """Mean EUQ head conflict / ignorance over visual-token lm_head inputs."""
    if device is None:
        device = getattr(lvlm, "device", torch.device("cpu"))

    hidden, visual_positions = _forward_visual_hidden(
        lvlm,
        image,
        question,
        positions_to_zero=positions_to_zero,
        model_type=model_type,
    )
    result = _compute_euq_from_hidden(hidden, weight_dir, lvlm_version, device)
    result["n_visual_tokens"] = int(visual_positions.numel())
    return result


@torch.no_grad()
def compute_visual_entropy_and_euq(
    lvlm,
    image,
    question,
    weight_dir,
    lvlm_version,
    device=None,
    chunk_size=256,
    positions_to_zero=None,
    model_type=None,
):
    """Both H_vis and EUQ visual stats from a single (optionally masked) forward."""
    if device is None:
        device = getattr(lvlm, "device", torch.device("cpu"))

    hidden, visual_positions = _forward_visual_hidden(
        lvlm,
        image,
        question,
        positions_to_zero=positions_to_zero,
        model_type=model_type,
    )
    lm_head = get_lm_head(lvlm.model)
    entropy_stats = _compute_entropy_from_hidden(hidden, lm_head, chunk_size=chunk_size)
    euq_stats = _compute_euq_from_hidden(hidden, weight_dir, lvlm_version, device)
    n_tokens = int(visual_positions.numel())
    return {
        **entropy_stats,
        **euq_stats,
        "n_visual_tokens": n_tokens,
    }


def _log_entropy_stats(log_dict, idx, result, log_per_image=False):
    log_dict[idx]["visual_entropy"] = result["visual_entropy"]
    log_dict[idx]["n_visual_tokens"] = result["n_visual_tokens"]
    if log_per_image:
        log_dict[idx]["visual_token_entropies"] = result["visual_token_entropies"]


def _log_euq_stats(log_dict, idx, euq_stats, log_per_image=False):
    log_dict[idx]["visual_mean_head_conflict_value"] = euq_stats[
        "visual_mean_head_conflict_value"
    ]
    log_dict[idx]["visual_mean_head_ignorance_value"] = euq_stats[
        "visual_mean_head_ignorance_value"
    ]
    log_dict[idx]["n_visual_tokens"] = euq_stats["n_visual_tokens"]
    if log_per_image:
        log_dict[idx]["visual_token_head_conflict_values"] = euq_stats[
            "visual_token_head_conflict_values"
        ]
        log_dict[idx]["visual_token_head_ignorance_values"] = euq_stats[
            "visual_token_head_ignorance_values"
        ]


def maybe_log_visual_statistics(
    args,
    lvlm,
    sample,
    log_dict,
    positions_to_zero=None,
):
    """Optionally log visual statistics for interpretability plots.

    For VAUQ, pass ``positions_to_zero`` so statistics are computed on the
    masked visual input (same ablation as the VAUQ forward), and both visual
    entropy and EUQ head statistics are logged.
    """
    if not getattr(args, "compute_visual_statistics", False):
        return None

    log_per_image = False
    idx = sample["idx"]
    uncertainty = getattr(args, "uncertainty", "").lower()
    model_type = getattr(args, "lvlm", None)

    if uncertainty == "vauq" and lvlm is not None:
        combined = compute_visual_entropy_and_euq(
            lvlm,
            sample["img"],
            sample["question"],
            weight_dir=lvlm.weight_dir,
            lvlm_version=lvlm.version,
            device=getattr(lvlm, "device", None),
            positions_to_zero=positions_to_zero,
            model_type=model_type,
        )
        _log_entropy_stats(log_dict, idx, combined, log_per_image=log_per_image)
        _log_euq_stats(log_dict, idx, combined, log_per_image=log_per_image)
        return combined

    if uncertainty == "euq" and lvlm is not None:
        euq_stats = compute_visual_euq_head_statistics(
            lvlm,
            sample["img"],
            sample["question"],
            weight_dir=lvlm.weight_dir,
            lvlm_version=lvlm.version,
            device=getattr(lvlm, "device", None),
            positions_to_zero=positions_to_zero,
            model_type=model_type,
        )
        _log_euq_stats(log_dict, idx, euq_stats, log_per_image=log_per_image)
        return euq_stats

    result = compute_visual_entropy(
        lvlm,
        sample["img"],
        sample["question"],
        positions_to_zero=positions_to_zero,
        model_type=model_type,
    )
    _log_entropy_stats(log_dict, idx, result, log_per_image=log_per_image)
    return result
