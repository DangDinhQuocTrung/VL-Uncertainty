import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interpretability.scopes_lens_utils import load_model, get_num_layers, get_transformer_layers, get_post_hook, logit_lens_on_token


MODEL_NAME = "google/gemma-3-12b-it"


def get_transformer_layers(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    raise AttributeError("Could not find transformer layers on model.")


def get_num_layers(model):
    return len(get_transformer_layers(model))


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(MODEL_NAME, dtype=torch.bfloat16, device=device)
    num_layers = get_num_layers(model)
    print("Model number of layers:", num_layers)
    U_lh = model.language_model.lm_head.weight

    # Source prompt: extract a hidden state from a specific token position.
    # prompt_source = "Patchscopes is robust. It helps interpret..."
    prompt_source = "Amazon's former CEO attented Oscars"
    # prompt_source = "The president of the US attended Oscars"
    inputs_source = tokenizer(prompt_source, return_tensors="pt").to(device)
    print("Source inputs:", len(inputs_source["input_ids"][0]))
    print([tokenizer.decode(token_id, skip_special_tokens=False) for token_id in inputs_source["input_ids"][0]])

    with torch.no_grad():
        outputs_source = model(**inputs_source, output_hidden_states=True, return_dict=True)
    hidden_states = outputs_source.hidden_states
    print("Hidden states:", len(hidden_states), hidden_states[0].shape)

    # Token to patch from (e.g. " CEO" in the source sentence).
    chosen_token_id = tokenizer.encode(" CEO", add_special_tokens=False)[0]
    # chosen_token_id = tokenizer.encode(" US", add_special_tokens=False)[0]
    chosen_position = inputs_source["input_ids"][0].tolist().index(chosen_token_id)
    # Amazon Jeff: 25 33; Denmark Copenhagen: 26
    target_layer = 26
    chosen_hidden_states = hidden_states[target_layer + 1][:, chosen_position, :]
    print(
        "Chosen hidden states:",
        target_layer,
        chosen_position, tokenizer.decode(chosen_token_id, skip_special_tokens=False),
        chosen_hidden_states.shape,
        chosen_hidden_states.mean().item(),
        chosen_hidden_states.std().item(),
    )
    lens_tokens = logit_lens_on_token(U_lh, chosen_hidden_states, tokenizer)
    print("Logit lens tokens:", lens_tokens)
    print()

    # Target prompt: few-shot identity mapping, as in the patchscopes notebook.
    prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n? ->"
    # prompt_target = f"Syria: Country in the Middle East\nLeonardo DiCaprio: American actor\nSamsung: South Korean multinational major appliance and consumer electronics corporation\n?"
    # prompt_target = "?"
    inputs_target = tokenizer(prompt_target, return_tensors="pt").to(device)
    print("Target inputs:", len(inputs_target["input_ids"][0]))
    print(tokenizer.decode(inputs_target["input_ids"][0], skip_special_tokens=False))
    N = 5

    # No patch
    with torch.no_grad():
        # Generate N tokens in a loop
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
    position_x = len(inputs_target["input_ids"][0]) - 1
    # position_x = len(inputs_target["input_ids"][0]) - 2
    print("Patch position:", position_x, tokenizer.decode(inputs_target["input_ids"][0, position_x]))
    if target_layer == num_layers - 1:
        skip_final_ln = True
    else:
        skip_final_ln = False
    skip_ln_name = "_skip_ln" if skip_final_ln else "ln"

    # pre_hook = get_pre_hook(f"layer_{target_layer}", position_x, chosen_hidden_states, generation_mode=True)
    post_hook = get_post_hook(f"layer_{target_layer}{skip_ln_name}", position_x, chosen_hidden_states, generation_mode=True)
    layer_m = get_transformer_layers(model)[target_layer]
    handle = layer_m.register_forward_hook(post_hook)
    N = 10 if len(prompt_target) <= 3 else 1

    if N == 1:
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
        lens_tokens = logit_lens_on_token(U_lh, patch_hidden_states, tokenizer)
        print("Logit lens tokens:", lens_tokens)

        last_hidden_states = outputs.hidden_states[-1][:, -2, :]
        lens_tokens = logit_lens_on_token(U_lh, last_hidden_states, tokenizer)
        print("Logit lens tokens:", lens_tokens)
    else:
        with torch.no_grad():
            # Generate N tokens in a loop
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

                # Add the predicted token to the generated text
                generated_text += next_token

        print(f"\nFull generated text: {generated_text}")
        handle.remove()
    return


if __name__ == "__main__":
    check_patchscopes()
