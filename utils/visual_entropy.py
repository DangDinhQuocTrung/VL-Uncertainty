"""Visual entropy via LogitLens on final-layer visual tokens.

Matches Visual Semantic Entropy (arXiv:2606.31407), Sec. 3.2:

    z_i = W v_i,   z̃_i = softmax(z_i)
    H_vis = avg_i [ -sum_k z̃_i(k) log z̃_i(k) ]

where v_i are final-layer visual token embeddings and W is the LM head.
Lower H_vis => more visually confident representation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.model_utils import get_final_norm, get_lm_head, resolve_image_token_id


def _token_entropy_from_logits(logits):
    """Shannon entropy (nats) of each row in logits (N, V)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


@torch.no_grad()
def compute_visual_entropy(lvlm, image, question, chunk_size=256):
    """Compute average LogitLens entropy over visual tokens.

    Uses lvlm.prepare_inputs(...) so preprocessing matches generate().
    Then runs a single forward pass (no decoding) to read final-layer
    visual-token hidden states.

    Returns:
        dict with keys:
          - visual_entropy (float): H_vis
          - n_visual_tokens (int)
          - visual_token_entropies (list[float]): per-token entropies
    """
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
    # Final transformer layer residual stream.
    hidden = outputs.hidden_states[-1][0, visual_positions]  # (N_vis, D)

    final_norm = get_final_norm(lvlm.model)
    if final_norm is not None:
        hidden = final_norm(hidden)

    lm_head = get_lm_head(lvlm.model)
    # Chunk over visual tokens to limit peak memory on large vocabularies.
    entropies = []
    for start in range(0, hidden.shape[0], chunk_size):
        chunk = hidden[start : start + chunk_size]
        logits = lm_head(chunk)
        ent = _token_entropy_from_logits(logits)
        entropies.append(ent.detach().float().cpu())
    token_entropies = torch.cat(entropies, dim=0)
    h_vis = float(token_entropies.mean().item())

    return {
        "visual_entropy": h_vis,
        "n_visual_tokens": int(visual_positions.numel()),
        "visual_token_entropies": [float(x) for x in token_entropies.tolist()],
    }


def maybe_log_visual_entropy(args, lvlm, sample, log_dict):
    """Optionally compute and store H_vis for the original (image, question)."""
    if not getattr(args, "compute_visual_entropy", False):
        return None
    result = compute_visual_entropy(lvlm, sample["img"], sample["question"])
    idx = sample["idx"]
    log_dict[idx]["visual_entropy"] = result["visual_entropy"]
    log_dict[idx]["n_visual_tokens"] = result["n_visual_tokens"]
    log_dict[idx]["visual_token_entropies"] = result["visual_token_entropies"]
    return result
