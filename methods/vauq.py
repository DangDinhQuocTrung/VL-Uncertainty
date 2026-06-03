import torch
import torch.nn.functional as F
from utils.constants import BENCHMARK_TYPE


def compute_entropy(outputs):
    entropies = []
    for step_logits in outputs["scores"]:
        log_probs = F.log_softmax(step_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
        entropies.append(entropy)
    mean_entropy = torch.mean(torch.stack(entropies))
    return mean_entropy


def generate_with_masked_visual_tokens(model, inputs, top_k_visual_positions, model_type="llava"):
    if "llava" in model_type:
        model_type = "llava"
    elif "Qwen" in model_type:
        model_type = "qwen"
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    # better: hook the input to layer 0 using register_forward_pre_hook
    def pre_hook(module, input_):
        # input_ is a tuple, first element is hidden states
        hidden_states = input_[0]
        if hidden_states.shape[1] > top_k_visual_positions.shape[0]:
            hidden_states[:, top_k_visual_positions, :] = 0.0
        return (hidden_states,) + input_[1:]

    if model_type == "llava":
        layer_0 = model.model.language_model.model.layers[0]
    elif model_type == "qwen":
        layer_0 = model.model.model.layers[0]

    # inputs["attention_mask"][0, top_k_visual_positions] = 0
    handle = layer_0.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            output_scores=True,
            return_dict_in_generate=True,
        )
    handle.remove()

    if model_type == "llava":
        answer = outputs["sequences"]
        final_answer = (
            model.processor.decode(answer[0], skip_special_tokens=True)
            .split("ASSISTANT: ")[-1]
            .strip()
        )
    elif model_type == "qwen":
        generated_ids = outputs["sequences"]
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = model.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        final_answer = answer[0]

    return final_answer, outputs


def estimate_uncertainty_by_vauq(args, lvlm, sample, llm, log_dict):
    K = 40
    alpha = 1.0
    device = lvlm.device

    # Generate answer
    answer, inputs, outputs = lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp,
        return_more=True,
    )
    log_dict[sample["idx"]]["answer"] = answer
    flag_answer_correct = True
    if BENCHMARK_TYPE[args.benchmark] == "MULTI_CHOICE":
        flag_answer_correct = str(sample["gt_answer"]) in answer
    else:
        question = f"Ground truth: {sample['gt_answer']}. Model answer: {answer}. Please verify if the model answer matches the ground truth. Respond with either 'Correct' or 'Wrong' only."
        llm_answer_check = llm.generate(question, 0.1)
        log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
        flag_answer_correct = (
            "Correct" in llm_answer_check
            or "correct" in llm_answer_check
            or "C" in llm_answer_check
            or "c" in llm_answer_check
        )
    log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct
    log_dict[sample["idx"]]["answer_sampling_list"] = [answer]

    # Compute entropy
    clean_entropy = compute_entropy(outputs).item()

    # Get visual tokens
    if "llava" in args.lvlm:
        image_token_id = lvlm.model.config.image_token_index
    elif "Qwen" in args.lvlm:
        image_token_id = lvlm.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    else:
        raise ValueError(f"Unsupported model: {args.lvlm}")

    visual_token_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    vision_token_start_index = visual_token_positions.min().item()
    vision_token_end_index = visual_token_positions.max().item() + 1
    generation_steps = len(outputs["attentions"])
    num_layers = len(outputs["attentions"][0])
    layer_range = [10, 25]
    sum_attention_over_vision_tokens = torch.zeros(vision_token_end_index - vision_token_start_index).to(device)
    for step_index in range(1, generation_steps):
        for layer_index in range(layer_range[0], layer_range[1]):
            attention_values = outputs["attentions"][step_index][layer_index][
                :, :, :, vision_token_start_index:vision_token_end_index]
            sum_attention_over_vision_tokens += attention_values.sum(dim=(0, 1, 2))

    # Masking
    top_k_indices = torch.topk(sum_attention_over_vision_tokens, K).indices
    top_k_vision_token_positions = visual_token_positions[top_k_indices]
    masked_answer, outputs_with_masked_visual_tokens = generate_with_masked_visual_tokens(
        lvlm, inputs, top_k_vision_token_positions, args.lvlm)
    masked_entropy = compute_entropy(outputs_with_masked_visual_tokens).item()

    # Log the results
    log_dict[sample["idx"]]["masked_answer"] = masked_answer
    log_dict[sample["idx"]]["clean_entropy"] = clean_entropy
    log_dict[sample["idx"]]["masked_entropy"] = masked_entropy
    log_dict[sample["idx"]]["image_score"] = masked_entropy - clean_entropy

    log_dict[sample["idx"]]["uncertainty"] = (alpha + 1.0) * clean_entropy - alpha * masked_entropy
    log_dict[sample["idx"]]["uncertainty_threshold"] = args.uncertainty_threshold
    flag_predict_hallucination = log_dict[sample["idx"]]["uncertainty"] >= args.uncertainty_threshold
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        log_dict[sample["idx"]]["flag_answer_correct"] and not flag_predict_hallucination
    ) or (
        not log_dict[sample["idx"]]["flag_answer_correct"] and flag_predict_hallucination
    )
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct
    return log_dict
