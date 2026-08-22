import torch

from utils.model_utils import resolve_image_token_id


def _layer_range_for_lvlm(lvlm_type):
    name = str(lvlm_type).lower()
    if "llava" in name:
        return [10, 25]
    if "qwen" in name:
        return [20, 40]
    if "gemma" in name:
        return [20, 33]
    raise ValueError(f"Unsupported model: {lvlm_type}")


def compute_attention_over_visual_tokens(model, processor, inputs, outputs, lvlm_type, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_token_id = resolve_image_token_id(
        model=model, processor=processor, name=lvlm_type
    )
    layer_range = _layer_range_for_lvlm(lvlm_type)

    visual_token_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    visual_token_start_index = visual_token_positions.min().item()
    visual_token_end_index = visual_token_positions.max().item() + 1
    generation_steps = len(outputs["attentions"])
    sum_attention_over_visual_tokens = torch.zeros(visual_token_end_index - visual_token_start_index).to(device)
    sum_attention_over_layers = torch.zeros(layer_range[1] - layer_range[0]).to(device)
    for step_index in range(1, generation_steps):
        for layer_index in range(layer_range[0], layer_range[1]):
            attention_values = outputs["attentions"][step_index][layer_index][
                :, :, :, visual_token_start_index:visual_token_end_index]
            sum_attention_over_visual_tokens += attention_values.sum(dim=(0, 1, 2))
            sum_attention_over_layers[layer_index - layer_range[0]] += attention_values.sum()

    return image_token_id, visual_token_positions, visual_token_start_index, visual_token_end_index, sum_attention_over_visual_tokens, sum_attention_over_layers
