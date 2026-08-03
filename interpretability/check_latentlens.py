import torch
import latentlens
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from PIL import Image

from methods.vauq import compute_attention_over_visual_tokens


def load_model(model_name, dtype=torch.float32, device=None, trust_remote_code=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load_kwargs = dict(trust_remote_code=trust_remote_code, dtype=dtype, attn_implementation="eager")
    load_kwargs = dict(trust_remote_code=trust_remote_code, torch_dtype=dtype, attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except (ValueError, KeyError, TypeError):
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        except (ValueError, KeyError, TypeError):
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model = model.to(device).eval()
    return model, tokenizer


def check_latentlens():
    # Load a VLM and its processor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model("Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16, device=device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # Load pre-built index for this VLM
    index = latentlens.ContextualIndex.from_pretrained(f"McGill-NLP/contextual_embeddings-qwen2.5-vl-7b")

    # Process image + text, apply_chat_template inserts the image placeholder tokens
    image = Image.open("/work3/dida/outputs_LVLM/ViLP/images/000_1.png")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image."},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Text prompt:", text_prompt)
    inputs = processor(images=[image], text=[text_prompt], return_tensors="pt", padding=True).to(device)
    print("Inputs:", len(inputs["input_ids"][0]))

    # Extract hidden states — pass all processor outputs
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            # return_dict_in_generate=True,
            return_dict=True,
        )
    hidden_states = outputs["hidden_states"]
    print("Hidden states:", len(hidden_states), hidden_states[0].shape)

    # Compute attention over visual tokens
    with torch.no_grad():
        generated_outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            output_attentions=True,
            # return_dict_in_generate=True,
            # return_dict=True,
        )
    image_token_id, visual_token_positions, visual_token_start_index, visual_token_end_index, sum_attention_over_visual_tokens, sum_attention_over_layers = compute_attention_over_visual_tokens(
        model, processor, inputs, generated_outputs, "Qwen2.5-VL-7B-Instruct", device)
    top_k_indices = torch.topk(sum_attention_over_visual_tokens, 40).indices
    top_k_visual_token_positions = visual_token_positions[top_k_indices]
    print("Top visual token positions:", len(top_k_visual_token_positions), top_k_visual_token_positions)
    for layer_index in range(len(sum_attention_over_layers)):
        print(f"Layer {layer_index}: {sum_attention_over_layers[layer_index].item()}")

    chosen_position = top_k_visual_token_positions[0]
    chosen_hidden_states = [hidden_state[:, chosen_position, :] for hidden_state in hidden_states]
    print("Chosen hidden states:", chosen_position, len(chosen_hidden_states), chosen_hidden_states[0].shape)

    # Interpret the hidden states of some layers for the chosen token
    chosen_depths = [0, 5, 10, 15, 20, 23, 27]
    for depth in chosen_depths:
        results = index.search(chosen_hidden_states[depth], top_k=3)
        print(f"Depth {depth}:")
        print(results[0][0])
        print(results[0][1])
        print(results[0][2])
        print()
    return


if __name__ == "__main__":
    check_latentlens()
