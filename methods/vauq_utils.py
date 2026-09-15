import torch

from utils.model_utils import resolve_image_token_id


def _layer_range_for_lvlm(lvlm_type):
    name = str(lvlm_type).lower()
    if "llava" in name:
        return [10, 25]
    elif "qwen2.5" in name:
        return [10, 25]
    elif "qwen" in name:
        return [20, 40]
    elif "gemma" in name:
        return [20, 33]
    raise ValueError(f"Unsupported model: {lvlm_type}")


def num_decoder_layers(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config; cannot infer layer count.")
    for attr in ("num_hidden_layers", "n_layer"):
        if hasattr(cfg, attr) and getattr(cfg, attr) is not None:
            return int(getattr(cfg, attr))
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for attr in ("num_hidden_layers", "n_layer"):
            if hasattr(text_cfg, attr) and getattr(text_cfg, attr) is not None:
                return int(getattr(text_cfg, attr))
    raise ValueError("Could not infer number of decoder layers from model config.")


def resolve_visual_token_span(model, processor, inputs, lvlm_type):
    image_token_id = resolve_image_token_id(
        model=model, processor=processor, name=lvlm_type
    )
    visual_token_positions = (inputs["input_ids"][0] == image_token_id).nonzero(
        as_tuple=True
    )[0]
    if visual_token_positions.numel() == 0:
        raise ValueError("No visual tokens found in input_ids.")
    visual_token_start_index = visual_token_positions.min().item()
    visual_token_end_index = visual_token_positions.max().item() + 1
    return (
        image_token_id,
        visual_token_positions,
        visual_token_start_index,
        visual_token_end_index,
    )


def _attention_to_visual_slice(attn_layer, visual_start, visual_end, query_mode):
    """Reduce one layer attention tensor to per-visual-token mass.

    ``query_mode``:
      - ``\"all\"``: sum over all query positions (overall / all tokens)
      - ``\"last\"``: only the last query position (newly generated token)
    """
    # attn_layer: (batch, heads, q_len, kv_len)
    visual = attn_layer[:, :, :, visual_start:visual_end]
    if query_mode == "all":
        return visual.sum(dim=(0, 1, 2))
    if query_mode == "last":
        return visual[:, :, -1:, :].sum(dim=(0, 1, 2))
    raise ValueError(f"Unsupported query_mode: {query_mode}")


def compute_attention_maps_over_visual_tokens(
    model,
    processor,
    inputs,
    outputs,
    lvlm_type,
    layer_indices=None,
    device=None,
):
    """Per-layer attention mass on visual tokens for two query modes.

    Modes (matching the plotting script labels):
      - ``generated``: attention from generated-token queries only
        (prefill last query + each decode step), analogous to VAUQ's use of
        generation attentions restricted to answer tokens.
      - ``overall``: attention from all query positions in every step
        (full prompt self-attention + generated tokens).

    Returns a dict with tensors of shape ``(n_layers, n_visual_tokens)``.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (
        image_token_id,
        visual_token_positions,
        visual_start,
        visual_end,
    ) = resolve_visual_token_span(model, processor, inputs, lvlm_type)

    attentions = outputs["attentions"]
    n_layers_model = len(attentions[0])
    if layer_indices is None:
        layer_indices = list(range(n_layers_model))
    else:
        layer_indices = [int(i) for i in layer_indices]
        for li in layer_indices:
            if li < 0 or li >= n_layers_model:
                raise ValueError(
                    f"Layer index {li} out of range for model with {n_layers_model} layers."
                )

    n_visual = visual_end - visual_start
    n_sel = len(layer_indices)
    generated = torch.zeros(n_sel, n_visual, device=device)
    overall = torch.zeros(n_sel, n_visual, device=device)
    layer_to_row = {li: row for row, li in enumerate(layer_indices)}

    generation_steps = len(attentions)
    for step_index in range(generation_steps):
        step_attns = attentions[step_index]
        for layer_index in layer_indices:
            row = layer_to_row[layer_index]
            attn = step_attns[layer_index].to(device)
            # Prefill (step 0) has q_len == prompt_len: "overall" sums all prompt
            # queries; "generated" keeps only the last query (first answer token).
            # Decode steps have q_len == 1, so both modes coincide there.
            overall[row] += _attention_to_visual_slice(
                attn, visual_start, visual_end, query_mode="all",
            )
            if step_index > 0:
                generated[row] += _attention_to_visual_slice(
                    attn, visual_start, visual_end, query_mode="last",
                )

    return {
        "image_token_id": image_token_id,
        "visual_token_positions": visual_token_positions,
        "visual_token_start_index": visual_start,
        "visual_token_end_index": visual_end,
        "layer_indices": layer_indices,
        "generated": generated,
        "overall": overall,
    }


def compute_attention_over_visual_tokens(model, processor, inputs, outputs, lvlm_type, device=None):
    """VAUQ helper: sum generated-token attention over a model-specific layer band."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layer_range = _layer_range_for_lvlm(lvlm_type)
    layer_indices = list(range(layer_range[0], layer_range[1]))

    (
        image_token_id,
        visual_token_positions,
        visual_token_start_index,
        visual_token_end_index,
    ) = resolve_visual_token_span(model, processor, inputs, lvlm_type)

    n_visual = visual_token_end_index - visual_token_start_index
    sum_attention_over_visual_tokens = torch.zeros(n_visual, device=device)
    sum_attention_over_layers = torch.zeros(len(layer_indices), device=device)

    # Preserve original VAUQ behavior: skip prefill (step 0), sum steps 1+.
    generation_steps = len(outputs["attentions"])
    for step_index in range(1, generation_steps):
        for row, layer_index in enumerate(layer_indices):
            attention_values = outputs["attentions"][step_index][layer_index][
                :, :, :, visual_token_start_index:visual_token_end_index
            ]
            sum_attention_over_visual_tokens += attention_values.sum(dim=(0, 1, 2))
            sum_attention_over_layers[row] += attention_values.sum()

    return (
        image_token_id,
        visual_token_positions,
        visual_token_start_index,
        visual_token_end_index,
        sum_attention_over_visual_tokens,
        sum_attention_over_layers,
    )
