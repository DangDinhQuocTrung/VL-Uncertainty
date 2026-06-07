import torch
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

    load_kwargs = dict(trust_remote_code=trust_remote_code, dtype=dtype, attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except (ValueError, KeyError, TypeError):
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        except (ValueError, KeyError, TypeError):
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model = model.to(device).eval()
    return model, tokenizer


def get_pre_hook(position, patched_hidden_state, generation_mode=True):
    def pre_hook(module, input_):
        hidden_states = input_[0]
        input_len = len(hidden_states[0])
        if generation_mode and input_len == 1:
            return (hidden_states,) + input_[1:]
        diff = torch.sum(torch.abs(hidden_states[:, position] - patched_hidden_state))
        hidden_states[:, position] = patched_hidden_state
        print("Patched", position, diff)
        return (hidden_states,) + input_[1:]

    return pre_hook


def check_patchscopes():
    # Load a VLM and its processor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model("Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16, device=device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # Process image + text, apply_chat_template inserts the image placeholder tokens
    image = Image.open("/work3/dida/outputs_LVLM/ViLP/images/000_1.png")
    # question = "Describe this image."
    question = "Modern drones typically have four propellers. How many propellers does the drone in the picture have?"
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text_prompt], return_tensors="pt", padding=True).to(device)
    print("Inputs:", len(inputs["input_ids"][0]))
    # print(inputs["input_ids"][0])
    # print("-" + processor.decode(torch.tensor([26713]), skip_special_tokens=True) + "-")
    chosen_token_id = 2168  # " image" token
    chosen_token_id = 37232  # " drones" token
    chosen_token_id = 26713  # " drone" token
    chosen_token_id = 26713

    # Extract hidden states — pass all processor outputs
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict_in_generate=True,
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
            return_dict_in_generate=True,
        )
    image_token_id, visual_token_positions, visual_token_start_index, visual_token_end_index, sum_attention_over_visual_tokens, sum_attention_over_layers = compute_attention_over_visual_tokens(
        model, processor, inputs, generated_outputs, "Qwen2.5-VL-7B-Instruct", device)
    top_k_indices = torch.topk(sum_attention_over_visual_tokens, 40).indices
    top_k_visual_token_positions = visual_token_positions[top_k_indices]
    print("Top visual token positions:", len(top_k_visual_token_positions), top_k_visual_token_positions)

    # Hooking
    chosen_position = top_k_visual_token_positions[0]
    chosen_position = inputs["input_ids"][0].tolist().index(chosen_token_id)
    target_layer = 23
    chosen_hidden_states = hidden_states[target_layer][:, chosen_position, :]
    print("Chosen hidden states:", target_layer, chosen_position, chosen_hidden_states.shape, chosen_hidden_states.mean(), chosen_hidden_states.std())
    prompt = "dog -> dog\ncat -> cat\n1135 -> 1135\nJapan -> Japan\nTaylor -> Taylor\nhello -> hello\n?"
    # prompt = f"Syria: Country in the Middle East\nLeonardo DiCaprio: American actor\nSamsung: South Korean multinational major appliance and consumer electronics corporation\n?"
    # prompt = "?"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    print("Inputs:", len(inputs["input_ids"][0]))
    # print(inputs["input_ids"][0])
    # print(processor.decode(inputs["input_ids"][0], skip_special_tokens=True))
    # print("-" + processor.decode(torch.tensor([30]), skip_special_tokens=True) + "-")
    # print("-" + processor.decode(torch.tensor([937]), skip_special_tokens=True) + "-")
    position_x = inputs["input_ids"][0].tolist().index(30)
    print("Inputs:", len(inputs["input_ids"][0]), position_x)

    pre_hook = get_pre_hook(position_x, chosen_hidden_states, generation_mode=True)
    layer_m = model.model.language_model.layers[target_layer]
    handle = layer_m.register_forward_pre_hook(pre_hook)

    with torch.no_grad():
        outputs = model.generate(**inputs, output_hidden_states=True, return_dict_in_generate=True)
    handle.remove()

    generated_ids = outputs["sequences"]
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answer = answer[0]
    print("Answer:", answer)

    hidden_states = outputs["hidden_states"]
    patch_hidden_states = hidden_states[0][-1][:, position_x, :]
    print("Patch hidden states:", patch_hidden_states.mean(), patch_hidden_states.std())


if __name__ == "__main__":
    check_patchscopes()
