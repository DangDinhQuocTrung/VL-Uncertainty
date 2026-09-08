import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from interpretability.scopes_lens_utils import load_model, get_num_layers, get_transformer_layers, get_post_hook, logit_lens_on_token


# MODEL_NAME = "google/gemma-3-12b-it"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def check_lens():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(MODEL_NAME, dtype=torch.bfloat16, device=device)
    num_layers = get_num_layers(model)
    print("Model number of layers:", num_layers)
    U_et = model.language_model.model.embed_tokens.weight
    U_lh = model.language_model.lm_head.weight
    print(torch.equal(U_lh, U_et), U_lh.shape, U_et.shape)

    # Source prompt: extract a hidden state from a specific token position.
    # prompt_source = "Patchscopes is robust. It helps interpret..."
    prompt_source = "Amazon's former CEO attented Oscars"
    inputs_source = tokenizer(prompt_source, return_tensors="pt").to(device)
    print("Source inputs:", len(inputs_source["input_ids"][0]))
    print(tokenizer.decode(inputs_source["input_ids"][0], skip_special_tokens=False))

    with torch.no_grad():
        outputs_source = model(**inputs_source, output_hidden_states=True, return_dict=True)
    hidden_states = outputs_source.hidden_states
    print("Hidden states:", len(hidden_states), hidden_states[0].shape)

    # Token to patch from (e.g. " CEO" in the source sentence).
    chosen_token_id = tokenizer.encode(" CEO", add_special_tokens=False)[0]
    chosen_position = inputs_source["input_ids"][0].tolist().index(chosen_token_id)
    # Amazon Jeff: 33 37
    target_layer = 34
    chosen_hidden_states = hidden_states[target_layer + 1][:, chosen_position, :]
    print(
        "Chosen hidden states:",
        target_layer,
        chosen_position,
        chosen_hidden_states.shape,
        chosen_hidden_states.mean().item(),
        chosen_hidden_states.std().item(),
    )
    lens_tokens = logit_lens_on_token(U_lh, chosen_hidden_states, tokenizer)
    print("Logit lens tokens:", lens_tokens)
    print()

    # Loading mappings
    if MODEL_NAME == "google/gemma-3-12b-it":
        weight_dir = Path("/work3/dida/outputs_LVLM/patchscopes_full_pile/google/gemma-3-12b-it_mappings_pile")
    elif MODEL_NAME == "Qwen/Qwen2.5-VL-7B-Instruct":
        weight_dir = Path("/work3/dida/outputs_LVLM/patchscopes_full_pile/Qwen/Qwen2.5-VL-7B-Instruct_mappings_pile")
    last_layer = num_layers - 1
    mapping_file = weight_dir / f"mapping_{target_layer:02d}-{last_layer:02d}.npy"
    mapping = np.load(mapping_file)
    print("Mapping:", mapping.shape, mapping_file.name)
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    transform = lambda x: torch.tensor(
        np.squeeze(unpad(np.dot(pad(np.expand_dims(x.detach().cpu().float().numpy(), 0)), mapping))),
        dtype=torch.bfloat16,
    ).to(device)
    mapped_hidden_states = transform(chosen_hidden_states[0]).unsqueeze(0)
    print("Mapped hidden states:", mapped_hidden_states.shape, mapped_hidden_states.mean().item(), mapped_hidden_states.std().item())
    lens_tokens = logit_lens_on_token(U_lh, mapped_hidden_states, tokenizer)
    print("Logit lens tokens:", lens_tokens)
    print()

    # Target prompt: few-shot identity mapping, as in the patchscopes notebook.
    prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n? ->"
    inputs_target = tokenizer(prompt_target, return_tensors="pt").to(device)
    print("Target inputs:", len(inputs_target["input_ids"][0]))
    print(tokenizer.decode(inputs_target["input_ids"][0], skip_special_tokens=False))

    # No patch
    with torch.no_grad():
        # Generate N tokens in a loop
        N = 5
        generated_text = prompt_target

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
    target_layer = num_layers - 1
    position_x = len(inputs_target["input_ids"][0]) - 1
    print("Patch position:", position_x, tokenizer.decode(inputs_target["input_ids"][0, position_x]))
    if target_layer == num_layers - 1:
        skip_final_ln = True
    else:
        skip_final_ln = False
    skip_ln_name = "_skip_ln" if skip_final_ln else "ln"

    # pre_hook = get_pre_hook(f"layer_{target_layer}", position_x, chosen_hidden_states, generation_mode=True)
    post_hook = get_post_hook(f"layer_{target_layer}{skip_ln_name}", position_x, mapped_hidden_states, generation_mode=True)
    layer_m = get_transformer_layers(model)[target_layer]
    handle = layer_m.register_forward_hook(post_hook)

    with torch.no_grad():
        outputs = model(**inputs_target, output_hidden_states=True, return_dict=True)
        answer_probs, answer_token_id = torch.max(
            torch.softmax(outputs.logits[0, -1, :], dim=0), dim=0
        )
        answer = (
            tokenizer.decode([answer_token_id], skip_special_tokens=True),
            round(answer_probs.cpu().item(), 4),
        )
        print("Answer:", answer[0])
        print("Probability:", answer[1])
    handle.remove()

    patch_hidden_states = outputs.hidden_states[-1][:, position_x, :]
    print(
        "Patch hidden states:",
        patch_hidden_states.mean().item(),
        patch_hidden_states.std().item(),
    )


if __name__ == "__main__":
    check_lens()
