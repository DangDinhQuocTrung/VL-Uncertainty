from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from PIL import Image

from methods.vauq_utils import compute_attention_over_visual_tokens
from interpretability.scopes_lens_utils import load_model, get_num_layers, get_transformer_layers, get_post_hook, logit_lens_on_token


MODEL_NAME = "google/gemma-3-12b-it"


def check_patchscopes():
    # Load a VLM and its processor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(MODEL_NAME, dtype=torch.bfloat16, device=device)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
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
    # print("-" + processor.decode(torch.tensor([262144]), skip_special_tokens=False) + "-")
    # print(processor.tokenizer.all_special_tokens)
    chosen_token_id = 2168  # " image" token
    chosen_token_id = 37232  # " drones" token
    chosen_token_id = 26713  # " drone" token
    chosen_token_id = tokenizer.encode(" image", add_special_tokens=False)[0]
    chosen_token_id = tokenizer.encode("<end_of_image>", add_special_tokens=False)[0]

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
        model, processor, inputs, generated_outputs, MODEL_NAME, device)
    top_k_indices = torch.topk(sum_attention_over_visual_tokens, 40).indices
    top_k_visual_token_positions = visual_token_positions[top_k_indices]
    print("Top visual token positions:", len(top_k_visual_token_positions), top_k_visual_token_positions)

    # Hooking
    # chosen_position = top_k_visual_token_positions[0]
    chosen_position = inputs["input_ids"][0].tolist().index(chosen_token_id)
    # chosen_position = inputs["input_ids"][0].tolist().index(151653)
    # chosen_position = max(i for i, x in enumerate(inputs["input_ids"][0].tolist()) if x == 151655)
    target_layer = num_layers - 2
    chosen_hidden_states = hidden_states[target_layer + 1][:, chosen_position, :]
    print("Chosen hidden states:", target_layer, chosen_position, chosen_hidden_states.shape, chosen_hidden_states.mean(), chosen_hidden_states.std())

    # Lens
    weight_dir = Path("/work3/dida/outputs_LVLM/patchscopes_full_pile/google/gemma-3-12b-it_mappings_pile")
    last_layer = num_layers - 1
    mapping_file = weight_dir / f"mapping_{target_layer:02d}-{last_layer:02d}.npy"
    mapping = np.load(mapping_file)
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    transform = lambda x: torch.tensor(
        np.squeeze(unpad(np.dot(pad(np.expand_dims(x.detach().cpu().float().numpy(), 0)), mapping))),
        dtype=torch.bfloat16,
    ).to(device)
    U_lh = model.language_model.lm_head.weight
    mapped_hidden_states = transform(chosen_hidden_states[0]).unsqueeze(0)
    lens_tokens = logit_lens_on_token(U_lh, mapped_hidden_states, tokenizer)
    print("Logit lens tokens:", lens_tokens)

    # prompt = "cat -> cat\n1135 -> 1135\nhello -> hello\n? ->"
    prompt = f"Syria: Country in the Middle East\nLeonardo DiCaprio: American actor\nSamsung: South Korean multinational major appliance and consumer electronics corporation\n?"
    # prompt = "?"
    # messages = [{"role": "user", "content": [
    #     {"type": "text", "text": prompt},
    # ]}]
    # prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    print("Inputs:", len(inputs["input_ids"][0]))
    # print(inputs["input_ids"][0])
    # print(processor.decode(inputs["input_ids"][0], skip_special_tokens=True))
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
