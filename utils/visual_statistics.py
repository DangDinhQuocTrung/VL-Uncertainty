"""Visual statistics for interpretability analysis.

- Semantic entropy path: LogitLens visual entropy H_vis (VSE Sec. 3.2).
- EUQ path: mean head conflict / ignorance over visual tokens.
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


@torch.no_grad()
def compute_visual_entropy(lvlm, image, question, chunk_size=256):
    """Average LogitLens entropy over final-layer visual tokens."""
    if not hasattr(lvlm, "prepare_inputs"):
        raise ValueError(
            "compute_visual_entropy requires lvlm.prepare_inputs(image, question). "
            f"Got {type(lvlm).__name__} ({getattr(lvlm, 'version', None)})."
        )

    inputs = lvlm.prepare_inputs(image, question)
    image_token_id = resolve_image_token_id(lvlm)
    visual_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    if visual_positions.numel() == 0:
        raise RuntimeError(
            f"No visual tokens found for image_token_id={image_token_id}."
        )

    outputs = lvlm.model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    # HF model outputs typically expose the final hidden state after the model's
    # last normalization step, so applying the final norm again would distort
    # the LogitLens ranking.
    hidden = outputs.hidden_states[-1][0, visual_positions]

    lm_head = get_lm_head(lvlm.model)
    entropies = []
    for start in range(0, hidden.shape[0], chunk_size):
        chunk = hidden[start : start + chunk_size]
        ent = _token_entropy_from_logits(lm_head(chunk))
        entropies.append(ent.detach().float().cpu())
    h_vis = float(torch.cat(entropies, dim=0).mean().item())

    return {
        "visual_entropy": h_vis,
        "n_visual_tokens": int(visual_positions.numel()),
    }


@torch.no_grad()
def compute_visual_euq_head_statistics(lvlm, image, question, weight_dir, lvlm_version, device=None):
    """Mean EUQ head conflict / ignorance over visual-token lm_head inputs."""
    if not hasattr(lvlm, "prepare_inputs"):
        raise ValueError(
            "compute_visual_euq_head_statistics requires lvlm.prepare_inputs. "
            f"Got {type(lvlm).__name__}."
        )
    if device is None:
        device = getattr(lvlm, "device", torch.device("cpu"))

    inputs = lvlm.prepare_inputs(image, question)
    image_token_id = resolve_image_token_id(lvlm)
    visual_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    if visual_positions.numel() == 0:
        raise RuntimeError(
            f"No visual tokens found for image_token_id={image_token_id}."
        )

    outputs = lvlm.model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    # Use the returned final hidden state directly; see note above.
    hidden = outputs.hidden_states[-1][0, visual_positions]

    weight_dir = Path(weight_dir)
    head_state_dict = torch.load(
        weight_dir / f"{lvlm_version}_head_weights.pth",
        map_location=device,
    )
    head_evidence_model = EvidenceModel(head_state_dict)

    conflict_sum = 0.0
    ig_sum = 0.0
    n_tokens = hidden.shape[0]
    for i in range(n_tokens):
        feature = hidden[i : i + 1]
        head_evidence_model.get_evidence_weights(feature.T)
        conflict_sum += head_evidence_model.get_evidence_conflict().item()
        ig_sum += head_evidence_model.get_evidence_ignorance().item()

    return {
        "visual_mean_head_conflict_value": conflict_sum / n_tokens,
        "visual_mean_head_ignorance_value": ig_sum / n_tokens,
        "n_visual_tokens": int(n_tokens),
    }


def maybe_log_visual_statistics(args, lvlm, sample, log_dict):
    """Optionally log visual statistics for interpretability plots."""
    if not getattr(args, "compute_visual_statistics", False):
        return None

    idx = sample["idx"]
    result = compute_visual_entropy(lvlm, sample["img"], sample["question"])
    log_dict[idx]["visual_entropy"] = result["visual_entropy"]
    log_dict[idx]["n_visual_tokens"] = result["n_visual_tokens"]

    if args.uncertainty == "euq" and lvlm is not None:
        euq_stats = compute_visual_euq_head_statistics(
            lvlm,
            sample["img"],
            sample["question"],
            weight_dir=lvlm.weight_dir,
            lvlm_version=lvlm.version,
            device=getattr(lvlm, "device", None),
        )
        log_dict[idx]["visual_mean_head_conflict_value"] = euq_stats[
            "visual_mean_head_conflict_value"
        ]
        log_dict[idx]["visual_mean_head_ignorance_value"] = euq_stats[
            "visual_mean_head_ignorance_value"
        ]

    return result
