"""TunedLens early-exit uncertainty.

Run the LVLM once on a sample (no temperature / perturbation sampling). Apply
pretrained TunedLens affine maps so each intermediate layer can be read out in
final-layer space, decode a full answer per layer (teacher-forced along the
greedy trajectory), and score uncertainty by how late the final answer first
stabilizes across layers (earlier => more certain).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from interpretability.scopes_lens_utils import get_num_layers
from methods.evaluate_by_llm import (
    evaluate_answer_correctness_by_llm,
    evaluate_multiple_choice_answer_correctness,
)
from methods.nli import get_nli_classifier
from utils.constants import is_choice_question
from utils.misc import parse_original_question
from utils.model_utils import get_lm_head


DEFAULT_TUNED_LENS_DIR = (
    "/work3/dida/outputs_LVLM/patchscopes_full_pile/Qwen/"
    "Qwen2.5-7B-Instruct_mappings_pile"
)

_MAPPING_CACHE: dict[str, dict[int, np.ndarray]] = {}


def _normalize_answer(text):
    return " ".join(str(text).strip().split())


def _resolve_tokenizer(lvlm):
    processor = getattr(lvlm, "processor", None)
    if processor is None:
        raise ValueError("tuned_lens_uncertainty requires lvlm.processor.")
    return getattr(processor, "tokenizer", processor)


def _mapping_path(weight_dir, layer_idx, last_layer):
    return Path(weight_dir) / f"mapping_{layer_idx:02d}-{last_layer:02d}.npy"


def load_tuned_lens_mappings(weight_dir, num_layers, layer_indices=None):
    """Load affine TunedLens maps ``mapping_{L:02d}-{last:02d}.npy`` (cached)."""
    weight_dir = str(weight_dir)
    last_layer = num_layers - 1
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(i) for i in layer_indices]

    cache_key = f"{weight_dir}::{last_layer}"
    if cache_key not in _MAPPING_CACHE:
        _MAPPING_CACHE[cache_key] = {}
    cache = _MAPPING_CACHE[cache_key]

    mappings = {}
    for layer_idx in layer_indices:
        if layer_idx in cache:
            mappings[layer_idx] = cache[layer_idx]
            continue
        path = _mapping_path(weight_dir, layer_idx, last_layer)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing TunedLens mapping for layer {layer_idx}: {path}"
            )
        mapping = np.load(path)
        cache[layer_idx] = mapping
        mappings[layer_idx] = mapping
    return mappings


def apply_tuned_lens(hidden, mapping, out_dtype=None):
    """Apply affine map stored in homogeneous coordinates: ``[h,1] @ M``, drop pad.

    ``mapping`` has shape ``(D+1, D+1)`` as produced by the Pile TunedLens
    training used in ``interpretability/check_lens_text.py``.
    """
    if hidden.ndim == 1:
        hidden = hidden.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    device = hidden.device
    mapping_t = torch.as_tensor(mapping, device=device, dtype=torch.float32)
    x = hidden.float()
    ones = torch.ones(x.shape[0], 1, device=device, dtype=torch.float32)
    mapped = torch.cat([x, ones], dim=-1) @ mapping_t
    mapped = mapped[:, :-1]
    if out_dtype is not None:
        mapped = mapped.to(dtype=out_dtype)
    if squeeze:
        mapped = mapped.squeeze(0)
    return mapped


def select_layer_indices(num_layers, min_layer=0, layer_stride=1, layer_indices=None):
    if layer_indices is not None:
        indices = sorted({int(i) for i in layer_indices})
        for i in indices:
            if i < 0 or i >= num_layers:
                raise ValueError(
                    f"tuned_lens layer index {i} out of range [0, {num_layers - 1}]."
                )
        return indices
    min_layer = max(0, int(min_layer))
    layer_stride = max(1, int(layer_stride))
    indices = list(range(min_layer, num_layers, layer_stride))
    if (num_layers - 1) not in indices:
        indices.append(num_layers - 1)
    return indices


def _extend_inputs_with_answer(inputs, answer_ids):
    """Copy multimodal inputs and append greedy answer token ids."""
    if answer_ids.ndim == 1:
        answer_ids = answer_ids.unsqueeze(0)
    n_answer = int(answer_ids.shape[-1])
    fwd = {k: v for k, v in inputs.items()}
    fwd["input_ids"] = torch.cat([inputs["input_ids"], answer_ids], dim=-1)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        extra = torch.ones(
            inputs["attention_mask"].shape[0],
            n_answer,
            device=inputs["attention_mask"].device,
            dtype=inputs["attention_mask"].dtype,
        )
        fwd["attention_mask"] = torch.cat([inputs["attention_mask"], extra], dim=-1)
    return fwd, n_answer


@torch.no_grad()
def decode_layer_answers(
    lvlm,
    inputs,
    answer_ids,
    mappings,
    layer_indices,
):
    """Teacher-forced TunedLens answers along the greedy token trajectory.

    For each selected layer ``L`` and each answer position ``j``, map the hidden
    state at ``prompt_len - 1 + j`` through the TunedLens affine and read out the
    next-token prediction via ``lm_head``.
    """
    model = lvlm.model
    processor = lvlm.processor
    tokenizer = _resolve_tokenizer(lvlm)
    lm_head = get_lm_head(model)
    prompt_len = int(inputs["input_ids"].shape[-1])

    if answer_ids.ndim == 2:
        answer_ids = answer_ids[0]
    answer_ids = answer_ids.to(inputs["input_ids"].device)

    if answer_ids.numel() == 0:
        return {int(i): "" for i in layer_indices}

    fwd_inputs, n_answer = _extend_inputs_with_answer(inputs, answer_ids)
    outputs = model(
        **fwd_inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    # hidden_states[0] = embeddings; hidden_states[L+1] = after transformer layer L.
    pos_start = prompt_len - 1
    pos_end = pos_start + n_answer

    eos_id = getattr(tokenizer, "eos_token_id", None)
    layer_answers = {}
    for layer_idx in layer_indices:
        h = hidden_states[layer_idx + 1][0, pos_start:pos_end, :]
        mapped = apply_tuned_lens(h, mappings[layer_idx], out_dtype=h.dtype)
        logits = lm_head(mapped)
        pred_ids = logits.argmax(dim=-1).tolist()
        if eos_id is not None:
            trimmed = []
            for tid in pred_ids:
                if int(tid) == int(eos_id):
                    break
                trimmed.append(int(tid))
            pred_ids = trimmed
        layer_answers[int(layer_idx)] = processor.batch_decode(
            [pred_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    return layer_answers


def compute_earliness_uncertainty(
    layer_indices,
    layer_answers,
    reference_answer,
    nli=None,
    question=None,
):
    """Uncertainty in ``[0, 1]`` from how late the final answer crystallizes.

    Find the earliest layer index ``i`` such that answers from ``i`` through the
    last selected layer are all NLI-equivalent to ``reference_answer``. Then

        uncertainty = rank(i) / (n_layers - 1)

    so matching from the first layer => 0, only at the final layer => 1.
    """
    n = len(layer_indices)
    if n == 0:
        return 1.0, None, []

    match_flags = []
    for layer_idx in layer_indices:
        layer_answer = layer_answers[layer_idx]
        if nli is None:
            match_flags.append(
                _normalize_answer(layer_answer) == _normalize_answer(reference_answer)
            )
        else:
            equivalent, _, _ = nli.bidirectional_entailment(
                layer_answer.strip(), reference_answer.strip(), question=question,
            )
            match_flags.append(bool(equivalent))

    stable_rank = n - 1
    for rank in range(n):
        if all(match_flags[rank:]):
            stable_rank = rank
            break

    denom = max(n - 1, 1)
    uncertainty = float(stable_rank / denom)
    stable_layer = int(layer_indices[stable_rank])
    return uncertainty, stable_layer, match_flags


def estimate_uncertainty_by_tuned_lens(args, lvlm, sample, llm, log_dict):
    weight_dir = getattr(args, "tuned_lens_dir", None) or DEFAULT_TUNED_LENS_DIR
    min_layer = int(getattr(args, "tuned_lens_min_layer", 0))
    layer_stride = int(getattr(args, "tuned_lens_layer_stride", 1))
    layer_indices_arg = getattr(args, "tuned_lens_layers", None)
    nli_model = get_nli_classifier()

    # Single greedy decode (no multi-sample uncertainty).
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

    num_layers = get_num_layers(lvlm.model)
    layer_indices = select_layer_indices(
        num_layers,
        min_layer=min_layer,
        layer_stride=layer_stride,
        layer_indices=layer_indices_arg,
    )
    mappings = load_tuned_lens_mappings(weight_dir, num_layers, layer_indices)

    prompt_len = int(inputs["input_ids"].shape[-1])
    answer_ids = outputs["sequences"][0, prompt_len:]
    layer_answer_map = decode_layer_answers(lvlm, inputs, answer_ids, mappings, layer_indices)
    # Early -> late for answer_sampling_list (reuse SE-style container).
    answer_sampling_list = [answer]
    # Prefer the last-layer TunedLens string as the crystallization reference so
    # comparison stays within the same readout family; fall back to greedy text.
    reference_answer = answer_sampling_list[-1] if answer_sampling_list else answer
    uncertainty, stable_layer, match_flags = compute_earliness_uncertainty(
        layer_indices, layer_answer_map, reference_answer,
        nli=nli_model,
    )

    log_dict[sample["idx"]]["answer_sampling_list"] = answer_sampling_list
    log_dict[sample["idx"]]["tuned_lens_layers"] = layer_indices
    log_dict[sample["idx"]]["tuned_lens_layer_answers"] = {
        str(i): layer_answer_map[i] for i in layer_indices
    }
    log_dict[sample["idx"]]["tuned_lens_match_flags"] = match_flags
    log_dict[sample["idx"]]["tuned_lens_stable_layer"] = stable_layer
    log_dict[sample["idx"]]["tuned_lens_dir"] = str(weight_dir)
    log_dict[sample["idx"]]["tuned_lens_reference_answer"] = reference_answer
    log_dict[sample["idx"]]["uncertainty"] = uncertainty
    log_dict[sample["idx"]]["uncertainty_threshold"] = args.uncertainty_threshold

    flag_predict_hallucination = uncertainty >= args.uncertainty_threshold
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        flag_answer_correct and not flag_predict_hallucination
    ) or (not flag_answer_correct and flag_predict_hallucination)
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct
    return log_dict
