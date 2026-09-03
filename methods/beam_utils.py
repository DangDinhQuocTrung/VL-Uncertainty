"""Shared multi-candidate generation helpers for PRO (beam) and RDS (sampling)."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def generate_greedy_answer(args, lvlm, sample):
    """Greedy / inference-temp decode used as the shared main answer (like VAUQ)."""
    return lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp,
    )


def generate_beam_candidates(args, lvlm, sample):
    """Generate num_beams answers with sequence NLLs/probs via beam search.

    Uses standard beam search by default. Diverse beams only if diversity_penalty > 0
    (can crash on bf16 + output_scores in recent Transformers).
    """
    num_beams = getattr(args, "num_beams", 10)
    diversity_penalty = getattr(args, "diversity_penalty", 0.0)
    use_diverse = diversity_penalty > 0.0
    num_beam_groups = num_beams if use_diverse else 1

    answer, inputs, outputs, answers = lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp,
        return_more=True,
        return_mode=1,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        length_penalty=0.0,
    )

    nlls = sequence_nlls_from_beam_outputs(outputs)
    probs = [math.exp(-nll) for nll in nlls]
    avg_nlls = average_nlls_from_beam_outputs(inputs, outputs, nlls)
    best_idx = int(np.argmax(probs))
    answer = answers[best_idx]
    return {
        "answer": answer,
        "answers": list(answers),
        "inputs": inputs,
        "outputs": outputs,
        "nlls": nlls,
        "avg_nlls": avg_nlls,
        "probs": probs,
        "best_idx": best_idx,
    }


def generate_temperature_samples(args, lvlm, sample):
    """Generate N answers via multinomial sampling with temperature (RDS paper).

    N defaults to sampling_time if > 0, else num_beams.
    Temperature defaults to sampling_temp if > 0, else 1.0.

    Samples are drawn sequentially (batch size 1). Batched num_return_sequences
    tiles vision inputs and OOMs on Qwen2.5-VL with eager attention.
    """
    n_samples = getattr(args, "sampling_time", 0)
    if n_samples is None or n_samples <= 0:
        n_samples = getattr(args, "num_beams", 10)
    temp = getattr(args, "sampling_temp", 0.0)
    if temp is None or temp <= 0.0:
        temp = 1.0

    answers = []
    nlls = []
    avg_nlls = []
    last_inputs = None
    last_outputs = None
    stop_token_ids = None

    for _ in range(n_samples):
        # return_mode=1: sequences + scores only (no attentions/hidden_states).
        _, inputs, outputs, sample_answers = lvlm.generate(
            sample["img"],
            sample["question"],
            temp,
            return_more=True,
            return_mode=1,
            num_beams=1,
            num_return_sequences=1,
            num_beam_groups=1,
            diversity_penalty=0.0,
            length_penalty=1.0,
        )
        if stop_token_ids is None:
            stop_token_ids = _stop_token_ids_from_lvlm(lvlm, outputs)
        sample_nlls, sample_avg_nlls = sequence_nlls_from_sample_outputs(
            inputs, outputs, stop_token_ids=stop_token_ids
        )
        answers.extend(sample_answers)
        nlls.extend(sample_nlls)
        avg_nlls.extend(sample_avg_nlls)
        last_inputs, last_outputs = inputs, outputs
        # Free large score tensors before the next sample.
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clamp non-finite NLLs so weighted RDS stays well-defined.
    nlls = [nll if math.isfinite(nll) else 1e6 for nll in nlls]
    avg_nlls = [nll if math.isfinite(nll) else 1e6 for nll in avg_nlls]
    probs = [math.exp(-nll) for nll in nlls]
    best_idx = int(np.argmin(avg_nlls)) if avg_nlls else 0
    answer = answers[best_idx]
    return {
        "answer": answer,
        "answers": list(answers),
        "inputs": last_inputs,
        "outputs": last_outputs,
        "nlls": nlls,
        "avg_nlls": avg_nlls,
        "probs": probs,
        "best_idx": best_idx,
        "n_samples": n_samples,
        "sampling_temp": temp,
    }


def sequence_nlls_from_beam_outputs(outputs):
    """Return sum-NLL for each returned beam (requires length_penalty=0.0)."""
    sequences_scores = getattr(outputs, "sequences_scores", None)
    if sequences_scores is None:
        raise ValueError(
            "Beam search outputs must include sequences_scores. "
            "Call generate with num_beams > 1, return_more=True, and length_penalty=0.0."
        )
    return (-sequences_scores).detach().float().cpu().tolist()


def sequence_nlls_from_sample_outputs(inputs, outputs, stop_token_ids=None):
    """Return (sum_nlls, avg_nlls) for each sampled sequence from generation scores.

    Important: stop at EOS/pad. After early stopping, HF right-pads sequences; pad
    positions often have -inf logits, which would make NLL become Infinity.
    """
    scores = getattr(outputs, "scores", None)
    if scores is None:
        raise ValueError(
            "Sampling outputs must include scores. "
            "Call generate with return_more=True and output_scores enabled."
        )

    prompt_len = inputs["input_ids"].shape[-1]
    sequences = outputs.sequences
    if stop_token_ids is None:
        stop_token_ids = _stop_token_ids_from_outputs(outputs)
    sum_nlls = []
    avg_nlls = []

    for seq_idx in range(sequences.shape[0]):
        gen_ids = sequences[seq_idx, prompt_len:]
        token_nlls = []
        for step_idx, step_logits in enumerate(scores):
            if step_idx >= gen_ids.shape[0]:
                break
            token_id = int(gen_ids[step_idx].item())
            # Skip pad/eos-only trailing positions if stop set was empty somehow.
            logits = step_logits[seq_idx].float()
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_prob = log_probs[token_id]
            if not torch.isfinite(token_log_prob):
                # Trailing pad / masked token — stop rather than poisoning the sum.
                break
            token_nlls.append(-token_log_prob)
            if stop_token_ids and token_id in stop_token_ids:
                break
        if not token_nlls:
            sum_nlls.append(0.0)
            avg_nlls.append(0.0)
        else:
            stacked = torch.stack(token_nlls)
            sum_nlls.append(stacked.sum().item())
            avg_nlls.append(stacked.mean().item())
    return sum_nlls, avg_nlls


def _as_token_id_set(token_ids):
    if token_ids is None:
        return set()
    if isinstance(token_ids, (list, tuple, set)):
        return {int(x) for x in token_ids}
    return {int(token_ids)}


def _stop_token_ids_from_outputs(outputs):
    stop_ids = set()
    stop_ids |= _as_token_id_set(getattr(outputs, "eos_token_id", None))
    stop_ids |= _as_token_id_set(getattr(outputs, "pad_token_id", None))
    return stop_ids or None


def _stop_token_ids_from_lvlm(lvlm, outputs=None):
    """Collect EOS/pad ids from model config; outputs alone often lack these fields."""
    stop_ids = set()
    if outputs is not None:
        stop_ids |= _as_token_id_set(getattr(outputs, "eos_token_id", None))
        stop_ids |= _as_token_id_set(getattr(outputs, "pad_token_id", None))

    model = getattr(lvlm, "model", None)
    if model is not None:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            stop_ids |= _as_token_id_set(getattr(gen_cfg, "eos_token_id", None))
            stop_ids |= _as_token_id_set(getattr(gen_cfg, "pad_token_id", None))
        cfg = getattr(model, "config", None)
        if cfg is not None:
            stop_ids |= _as_token_id_set(getattr(cfg, "eos_token_id", None))
            stop_ids |= _as_token_id_set(getattr(cfg, "pad_token_id", None))

    processor = getattr(lvlm, "processor", None)
    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
    if tokenizer is None:
        tokenizer = getattr(lvlm, "tokenizer", None)
    if tokenizer is not None:
        stop_ids |= _as_token_id_set(getattr(tokenizer, "eos_token_id", None))
        stop_ids |= _as_token_id_set(getattr(tokenizer, "pad_token_id", None))

    return stop_ids or None


def _eos_token_ids(outputs):
    return _stop_token_ids_from_outputs(outputs)


def generation_lengths(inputs, outputs):
    """Number of generated tokens per sequence (cut at first EOS when available)."""
    prompt_len = inputs["input_ids"].shape[-1]
    eos_ids = _eos_token_ids(outputs)
    lengths = []
    for seq in outputs.sequences:
        gen = seq[prompt_len:].tolist()
        if not gen:
            lengths.append(1)
            continue
        if eos_ids is None:
            lengths.append(len(gen))
            continue
        cut = len(gen)
        for i, tok in enumerate(gen):
            if tok in eos_ids:
                cut = i + 1
                break
        lengths.append(max(cut, 1))
    return lengths


def average_nlls_from_beam_outputs(inputs, outputs, nlls=None):
    """Length-normalized NLL (ANLL) per beam."""
    if nlls is None:
        nlls = sequence_nlls_from_beam_outputs(outputs)
    lengths = generation_lengths(inputs, outputs)
    return [nll / length for nll, length in zip(nlls, lengths)]
