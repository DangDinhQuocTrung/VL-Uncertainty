import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name, dtype=torch.bfloat16, device=None, trust_remote_code=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = model.to(device).eval()
    return model, tokenizer


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


def logit_lens_on_token(matrix, hidden_state, tokenizer, k=10):
    tokens = []
    with torch.no_grad():
        hidden_state = hidden_state[0] if hidden_state.ndim == 2 else hidden_state
        logits = matrix @ hidden_state
        topk = torch.topk(logits, k)
        for score, token_id in zip(topk.values, topk.indices):
            tokens.append(tokenizer.decode([token_id]))
    return tokens
