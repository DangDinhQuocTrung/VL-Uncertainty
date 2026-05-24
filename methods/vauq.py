from utils.constants import BENCHMARK_TYPE


def estimate_uncertainty_by_vauq(args, lvlm, sample, llm, log_dict):
    ans, inputs, outputs = lvlm.generate(
        sample["img"],
        sample["question"],
        0.2,
        return_more=True,
    )
    log_dict[sample["idx"]]["ans"] = ans
    flag_answer_correct = True
    if BENCHMARK_TYPE[args.benchmark] == "MULTI_CHOICE":
        flag_answer_correct = str(sample["gt_answer"]) in ans
    else:
        question = f"Ground truth: {sample['gt_answer']}. Model answer: {ans}. Please verify if the model ans matches the ground truth. Respond with either 'Correct' or 'Wrong' only."
        llm_answer_check = llm.generate(question, 0.1)
        log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
        flag_answer_correct = (
            "Correct" in llm_answer_check
            or "correct" in llm_answer_check
            or "C" in llm_answer_check
            or "c" in llm_answer_check
        )
    log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct
    log_dict[sample["idx"]]["answer_sampling_list"] = [ans]

    # Get visual tokens
    if "llava" in args.lvlm:
        image_token_id = lvlm.model.config.image_token_index
        visual_token_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    elif "Qwen" in args.lvlm:
        image_token_id = lvlm.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        visual_token_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    else:
        raise ValueError(f"Unsupported model: {args.lvlm}")

    print(visual_token_positions.shape, visual_token_positions[:5], visual_token_positions[-5:])
    print(len(outputs["attentions"][0]), len(outputs["attentions"]), outputs["attentions"][0][0].shape)

    return
