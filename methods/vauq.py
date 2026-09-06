import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm
from utils.constants import is_choice_question
from methods.vauq_utils import compute_attention_over_visual_tokens
from utils.visual_statistics import maybe_log_visual_statistics


def compute_entropy(outputs):
    entropies = []
    for step_logits in outputs["scores"]:
        log_probs = F.log_softmax(step_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
        entropies.append(entropy)
    mean_entropy = torch.mean(torch.stack(entropies))
    return mean_entropy


def _infer_patch_grid(n_tokens, image_width, image_height):
    """Factor n_tokens into (grid_h, grid_w) closest to the image aspect ratio."""
    if n_tokens <= 0:
        raise ValueError(f"n_visual_tokens must be positive, got {n_tokens}.")
    best = None
    target_ratio = image_height / max(image_width, 1)
    for grid_h in range(1, int(math.isqrt(n_tokens)) + 1):
        if n_tokens % grid_h != 0:
            continue
        grid_w = n_tokens // grid_h
        ratio = grid_h / grid_w
        score = abs(math.log(ratio + 1e-9) - math.log(target_ratio + 1e-9))
        if best is None or score < best[0]:
            best = (score, grid_h, grid_w)
    if best is None:
        raise ValueError(f"Could not factor n_visual_tokens={n_tokens} into a 2D grid.")
    return best[1], best[2]


def _resolve_patch_grid(lvlm, inputs, n_tokens, image):
    """Prefer Qwen image_grid_thw; otherwise infer from image aspect ratio."""
    img_w, img_h = image.size
    grid_thw = None
    if inputs is not None and "image_grid_thw" in inputs:
        grid_thw = inputs["image_grid_thw"]
    if grid_thw is not None:
        _, grid_h, grid_w = [int(x) for x in grid_thw[0].tolist()]
        merge_size = 2
        processor = getattr(lvlm, "processor", None)
        if processor is not None and hasattr(processor, "image_processor"):
            merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        token_h, token_w = grid_h // merge_size, grid_w // merge_size
        if token_h * token_w == n_tokens:
            return token_h, token_w
    return _infer_patch_grid(n_tokens, img_w, img_h)


def _masked_visual_token_indices(visual_token_positions, positions_to_zero):
    """Map absolute sequence positions -> indices in the visual-token raster."""
    pos_to_idx = {
        int(pos): i for i, pos in enumerate(visual_token_positions.tolist())
    }
    return [
        pos_to_idx[int(pos)]
        for pos in positions_to_zero.tolist()
        if int(pos) in pos_to_idx
    ]


def render_image_with_blacked_masked_tokens(
    image,
    masked_token_indices,
    grid_h,
    grid_w,
):
    """Divide image into a visual-token patch grid and black out masked patches."""
    img = image.convert("RGB")
    width, height = img.size
    out = np.asarray(img, dtype=np.uint8).copy()
    patch_h = height / grid_h
    patch_w = width / grid_w
    for idx in masked_token_indices:
        row, col = divmod(int(idx), grid_w)
        y0 = int(round(row * patch_h))
        y1 = int(round((row + 1) * patch_h))
        x0 = int(round(col * patch_w))
        x1 = int(round((col + 1) * patch_w))
        out[y0:y1, x0:x1] = 0
    return Image.fromarray(out)


def build_masked_image(
    lvlm,
    image,
    inputs,
    visual_token_positions,
    positions_to_zero,
):
    """Build a PIL image with VAUQ-masked patches set to black."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image)).convert("RGB")
    n_tokens = int(visual_token_positions.numel())
    grid_h, grid_w = _resolve_patch_grid(lvlm, inputs, n_tokens, image)
    masked_indices = _masked_visual_token_indices(
        visual_token_positions, positions_to_zero
    )
    rendered = render_image_with_blacked_masked_tokens(
        image, masked_indices, grid_h, grid_w
    )
    return rendered, grid_h, grid_w, masked_indices


def save_masked_visual_image(
    sample,
    log_dict,
    masked_image,
    grid_h,
    grid_w,
    n_masked_tokens,
    exp_dir="exp",
):
    """Save a patch visualization of VAUQ-masked visual tokens (blacked out)."""
    begin_time = log_dict.get("begin_time_str", "unknown")
    out_dir = Path(exp_dir) / f"masked_images_{begin_time}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{int(sample['idx']):04d}_masked.png"
    masked_image.save(out_path)

    log_dict[sample["idx"]]["masked_image_path"] = str(out_path)
    log_dict[sample["idx"]]["masked_image_grid"] = [int(grid_h), int(grid_w)]
    log_dict[sample["idx"]]["n_masked_visual_tokens"] = int(n_masked_tokens)
    return out_path


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


def _resolve_positions_to_zero(
    top_k_visual_positions, visual_token_positions, blur_key_regions
):
    top_k_visual_positions = top_k_visual_positions.to(dtype=torch.long)
    visual_token_positions = visual_token_positions.to(dtype=torch.long)
    if blur_key_regions:
        return top_k_visual_positions
    keep = torch.isin(visual_token_positions, top_k_visual_positions)
    return visual_token_positions[~keep]


def _decode_generate_answer(model, model_type, inputs, outputs):
    if model_type == "llava":
        answer = outputs["sequences"]
        return (
            model.processor.decode(answer[0], skip_special_tokens=True)
            .split("ASSISTANT: ")[-1]
            .strip()
        )
    if model_type in ["qwen", "qwen2.5"]:
        generated_ids = outputs["sequences"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = model.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return answer[0]
    if model_type == "gemma":
        input_len = inputs["input_ids"].shape[-1]
        trimmed_ids = outputs["sequences"][0][input_len:]
        return model.processor.decode(trimmed_ids, skip_special_tokens=True).strip()
    raise ValueError(f"Unsupported model: {model_type}")


def generate_with_masked_visual_tokens(
    model,
    inputs,
    top_k_visual_positions,
    visual_token_positions,
    model_type="llava",
    blur_key_regions=True,
    mask_mode="hook",
    masked_image=None,
    question=None,
    inference_temp=0.0,
):
    """Generate with VAUQ masking via layer-0 hook or a blacked-out image.

    mask_mode:
      - "hook": zero selected visual-token hidden states at layer 0 (original VAUQ).
      - "image": re-generate from ``masked_image`` with blacked-out patches (no hook).
    """
    mask_mode = str(mask_mode).lower()
    if mask_mode not in ("hook", "image"):
        raise ValueError(f"Unsupported vauq mask_mode={mask_mode!r}; expected hook|image.")

    model_type = _normalize_model_type(model_type)
    positions_to_zero = _resolve_positions_to_zero(
        top_k_visual_positions, visual_token_positions, blur_key_regions
    )

    if mask_mode == "image":
        if masked_image is None:
            raise ValueError("mask_mode='image' requires masked_image.")
        if question is None:
            raise ValueError("mask_mode='image' requires question.")
        answer, _, outputs, _ = model.generate(
            masked_image,
            question,
            inference_temp,
            return_more=True,
            return_mode=1,
        )
        return answer, outputs, positions_to_zero

    def pre_hook(module, input_):
        hidden_states = input_[0]
        if positions_to_zero.numel() == 0:
            return input_
        pos = positions_to_zero.to(hidden_states.device)
        if hidden_states.shape[1] > int(pos.max().item()):
            hidden_states[:, pos, :] = 0.0
        return (hidden_states,) + input_[1:]

    if model_type == "llava":
        layer_0 = model.model.language_model.model.layers[0]
    elif model_type == "qwen2.5":
        layer_0 = model.model.model.layers[0]
    elif model_type == "qwen":
        layer_0 = model.model.model.language_model.layers[0]
    elif model_type == "gemma":
        layer_0 = model.model.language_model.model.layers[0]

    handle = layer_0.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            output_scores=True,
            return_dict_in_generate=True,
        )
    handle.remove()

    final_answer = _decode_generate_answer(model, model_type, inputs, outputs)
    return final_answer, outputs, positions_to_zero


def estimate_uncertainty_by_vauq(args, lvlm, sample, llm, log_dict):
    k_percent = 40
    alpha = 1.0
    device = lvlm.device
    log_masked_image = False
    mask_mode = str(getattr(args, "vauq_mask_mode", "hook")).lower()

    # Generate answer
    answer, inputs, outputs, _answers = lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp,
        return_more=True,
        return_mode=2,
        # needs attentions over visual tokens
    )
    log_dict[sample["idx"]]["answer"] = answer
    flag_answer_correct = True
    if is_choice_question(args, sample):
        flag_answer_correct = str(sample["gt_answer"]) in answer
    else:
        flag_answer_correct, llm_answer_check = evaluate_answer_correctness_by_llm(llm, sample, answer)
        log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
    log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct
    log_dict[sample["idx"]]["answer_sampling_list"] = [answer]

    # Compute entropy
    clean_entropy = compute_entropy(outputs).item()

    # Get visual tokens
    image_token_id, visual_token_positions, visual_token_start_index, visual_token_end_index, sum_attention_over_visual_tokens, _ = compute_attention_over_visual_tokens(
        lvlm.model, lvlm.processor, inputs, outputs, args.lvlm, device)

    # Masking: blur_key_regions=True zeros top-K; False keeps top-K and zeros the rest.
    blur_key_regions = getattr(args, "blur_key_regions", True)
    k_patches = int(k_percent / 100 * visual_token_positions.shape[0])
    if not blur_key_regions:
        k_patches = int((100 - k_percent) / 100 * visual_token_positions.shape[0])
    top_k_indices = torch.topk(sum_attention_over_visual_tokens, k_patches).indices
    top_k_visual_token_positions = visual_token_positions[top_k_indices]
    positions_to_zero = _resolve_positions_to_zero(
        top_k_visual_token_positions, visual_token_positions, blur_key_regions
    )

    masked_image = None
    if log_masked_image or mask_mode == "image":
        masked_image, grid_h, grid_w, masked_indices = build_masked_image(
            lvlm,
            sample["img"],
            inputs,
            visual_token_positions,
            positions_to_zero,
        )
        if log_masked_image:
            save_masked_visual_image(
                sample,
                log_dict,
                masked_image,
                grid_h,
                grid_w,
                len(masked_indices),
                exp_dir="exp",
            )

    masked_answer, outputs_with_masked_visual_tokens, _ = generate_with_masked_visual_tokens(
        lvlm,
        inputs,
        top_k_visual_token_positions,
        visual_token_positions,
        args.lvlm,
        blur_key_regions=blur_key_regions,
        mask_mode=mask_mode,
        masked_image=masked_image,
        question=sample["question"],
        inference_temp=args.inference_temp,
    )
    masked_entropy = compute_entropy(outputs_with_masked_visual_tokens).item()

    # Log the results
    log_dict[sample["idx"]]["blur_key_regions"] = blur_key_regions
    log_dict[sample["idx"]]["vauq_mask_mode"] = mask_mode
    log_dict[sample["idx"]]["masked_answer"] = masked_answer
    log_dict[sample["idx"]]["clean_entropy"] = clean_entropy
    log_dict[sample["idx"]]["masked_entropy"] = masked_entropy
    log_dict[sample["idx"]]["image_score"] = masked_entropy - clean_entropy

    log_dict[sample["idx"]]["uncertainty"] = (alpha + 1.0) * clean_entropy - alpha * masked_entropy
    log_dict[sample["idx"]]["uncertainty_threshold"] = args.uncertainty_threshold
    flag_predict_hallucination = log_dict[sample["idx"]]["uncertainty"] >= args.uncertainty_threshold
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        log_dict[sample["idx"]]["flag_answer_correct"] and not flag_predict_hallucination
    ) or (
        not log_dict[sample["idx"]]["flag_answer_correct"] and flag_predict_hallucination
    )
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct

    # Visual stats: hook mode masks activations; image mode uses the blacked-out image.
    if mask_mode == "image":
        sample_for_stats = dict(sample)
        sample_for_stats["img"] = masked_image
        maybe_log_visual_statistics(
            args, lvlm, sample_for_stats, log_dict, positions_to_zero=None
        )
    else:
        maybe_log_visual_statistics(
            args, lvlm, sample, log_dict, positions_to_zero=positions_to_zero
        )
    return log_dict
