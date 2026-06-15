import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from PIL import Image

from methods.vauq import compute_attention_over_visual_tokens
from interpretability.check_patchscopes_text import get_transformer_layers, get_num_layers


def load_model(model_name, dtype=torch.float32, device=None, trust_remote_code=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


def get_pre_hook(name, position, patched_hidden_state, generation_mode=True):
    def pre_hook(module, input_):
        hidden_states = input_[0]
        input_len = len(hidden_states[0])
        if generation_mode and input_len == 1:
            return
        diff = torch.sum(torch.abs(hidden_states[:, position] - patched_hidden_state))
        hidden_states[:, position] = patched_hidden_state
        print("Patched", position, diff)

    return pre_hook


def get_post_hook(name, position, patched_hidden_state, generation_mode=True):
    def post_hook(module, input_, output_):
        if "skip_ln" in name:
            # output_: (batch, sequence, hidden_state)
            output_len = len(output_[0])
        else:
            # output_[0]: (batch, sequence, hidden_state)
            output_len = len(output_[0][0])

        if generation_mode and output_len == 1:
            return
        hs = output_[0][position] if "skip_ln" in name else output_[0][0, position]
        diff = torch.sum(torch.abs(hs - patched_hidden_state))
        if "skip_ln" in name:
            output_[0][position] = patched_hidden_state
        else:
            output_[0][0, position] = patched_hidden_state
        print("Patched", position, diff)

    return post_hook


def check_patchscopes():
    # Load a VLM and its processor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model("Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16, device=device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    num_layers = get_num_layers(model)
    print("Model number of layers:", num_layers)

    # Process image + text, apply_chat_template inserts the image placeholder tokens
    image = Image.open("/work3/dida/outputs_LVLM/ViLP/images/000_1.png")
    question = "Describe this image."
    # question = "Modern drones typically have four propellers. How many propellers does the drone in the picture have?"
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
    chosen_token_id = tokenizer.encode(" image", add_special_tokens=False)[0]

    # Extract hidden states — pass all processor outputs
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
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
            return_dict_in_generate=True,
        )
    image_token_id, visual_token_positions, visual_token_start_index, visual_token_end_index, sum_attention_over_visual_tokens, sum_attention_over_layers = compute_attention_over_visual_tokens(
        model, processor, inputs, generated_outputs, "Qwen2.5-VL-7B-Instruct", device)
    top_k_indices = torch.topk(sum_attention_over_visual_tokens, 40).indices
    top_k_visual_token_positions = visual_token_positions[top_k_indices]
    print("Top visual token positions:", len(top_k_visual_token_positions), top_k_visual_token_positions)

    # Hooking
    chosen_position = top_k_visual_token_positions[0]
    # chosen_position = inputs["input_ids"][0].tolist().index(chosen_token_id)
    # chosen_position = inputs["input_ids"][0].tolist().index(151653)
    # chosen_position = max(i for i, x in enumerate(inputs["input_ids"][0].tolist()) if x == 151655)
    target_layer = num_layers - 2
    chosen_hidden_states = hidden_states[target_layer + 1][:, chosen_position, :]
    print("Chosen hidden states:", target_layer, chosen_position, chosen_hidden_states.shape, chosen_hidden_states.mean(), chosen_hidden_states.std())

    prompt = "cat -> cat\n1135 -> 1135\nhello -> hello\n? ->"
    # prompt = f"Syria: Country in the Middle East\nLeonardo DiCaprio: American actor\nSamsung: South Korean multinational major appliance and consumer electronics corporation\n?"
    # prompt = "?"
    # messages = [{"role": "user", "content": [
    #     {"type": "text", "text": prompt},
    # ]}]
    # prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    print("Inputs:", len(inputs["input_ids"][0]))
    # print(inputs["input_ids"][0])
    print(processor.decode(inputs["input_ids"][0], skip_special_tokens=True))
    # print("-" + processor.decode(torch.tensor([30]), skip_special_tokens=True) + "-")
    # print("-" + processor.decode(torch.tensor([937]), skip_special_tokens=True) + "-")

    # No patch
    with torch.no_grad():
        # Generate N tokens in a loop
        N = 5
        generated_text = prompt

        for _ in range(N):
            # Get the current input
            current_input = tokenizer(generated_text, return_tensors="pt").to(device)
            # Run the model
            outputs = model(**current_input)

            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Get the predicted token
            next_token_id = torch.argmax(next_token_probs, dim=-1).item()
            next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
            prob = next_token_probs[0, next_token_id].item()

            # Add the predicted token to the generated text
            generated_text += next_token

    print(f"\nFull generated text: {generated_text}")
    print()

    # Patch the last token position ("? ->") during generation.
    position_x = len(inputs["input_ids"][0]) - 1
    print("Patch position:", position_x, tokenizer.decode(inputs["input_ids"][0, position_x]))
    if target_layer == num_layers - 1:
        skip_final_ln = True
    else:
        skip_final_ln = False
    skip_ln_name = "_skip_ln" if skip_final_ln else "ln"

    # pre_hook = get_pre_hook(f"layer_{target_layer}", position_x, chosen_hidden_states, generation_mode=True)
    post_hook = get_post_hook(f"layer_{target_layer}{skip_ln_name}", position_x, chosen_hidden_states, generation_mode=True)
    layer_m = get_transformer_layers(model)[target_layer]
    handle = layer_m.register_forward_hook(post_hook)

    with torch.no_grad():
        # Generate N tokens in a loop
        N = 10
        generated_text = prompt

        for _ in range(N):
            # Get the current input
            current_input = tokenizer(generated_text, return_tensors="pt").to(device)
            # Run the model
            outputs = model(**current_input)

            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Get the predicted token
            next_token_id = torch.argmax(next_token_probs, dim=-1).item()
            next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
            prob = next_token_probs[0, next_token_id].item()

            # Add the predicted token to the generated text
            generated_text += next_token

    print(f"\nFull generated text: {generated_text}")
    print()
    handle.remove()

    # with torch.no_grad():
    #     outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    #     answer_probs, answer_token_id = torch.max(
    #         torch.softmax(outputs.logits[0, -1, :], dim=0), dim=0
    #     )
    #     answer = (
    #         tokenizer.decode([answer_token_id], skip_special_tokens=True),
    #         round(answer_probs.cpu().item(), 4),
    #     )
    #     print("Answer:", f"[{answer[0]}]")
    #     print("Probability:", answer[1])
    # handle.remove()

    # patch_hidden_states = outputs.hidden_states[-1][:, position_x, :]
    # print(
    #     "Patch hidden states:",
    #     patch_hidden_states.mean().item(),
    #     patch_hidden_states.std().item(),
    # )


if __name__ == "__main__":
    check_patchscopes()
