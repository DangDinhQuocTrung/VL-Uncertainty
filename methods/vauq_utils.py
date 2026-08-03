import torch


def compute_attention_over_visual_tokens(model, processor, inputs, outputs, lvlm_type, device=None):
    layer_range = [10, 25] if lvlm_type == "llava" else [20, 40]
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if "llava" in lvlm_type:
        image_token_id = model.config.image_token_index
    elif "Qwen" in lvlm_type:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    elif "gemma" in lvlm_type:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    else:
        raise ValueError(f"Unsupported model: {lvlm_type}")

    visual_token_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    visual_token_start_index = visual_token_positions.min().item()
    visual_token_end_index = visual_token_positions.max().item() + 1
    generation_steps = len(outputs["attentions"])
    num_layers = len(outputs["attentions"][0])
    sum_attention_over_visual_tokens = torch.zeros(visual_token_end_index - visual_token_start_index).to(device)
    sum_attention_over_layers = torch.zeros(layer_range[1] - layer_range[0]).to(device)
    for step_index in range(1, generation_steps):
        for layer_index in range(layer_range[0], layer_range[1]):
            attention_values = outputs["attentions"][step_index][layer_index][
                :, :, :, visual_token_start_index:visual_token_end_index]
            sum_attention_over_visual_tokens += attention_values.sum(dim=(0, 1, 2))
            sum_attention_over_layers[layer_index - layer_range[0]] += attention_values.sum()

    return image_token_id, visual_token_positions, visual_token_start_index, visual_token_end_index, sum_attention_over_visual_tokens, sum_attention_over_layers
