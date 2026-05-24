import spacy
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from methods.svar.utils import *
from utils.constants import BENCHMARK_TYPE


def estimate_uncertainty_by_svar(args, model_manager, sample, llm, log_dict):
    # Inference
    answer, input_ids, outputs = model_manager.generate(
        sample["img"],
        sample["question"],
        0.0,
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

    # Get some constants
    vision_token_start = model_manager.img_start_idx
    vision_token_end = model_manager.img_end_idx
    input_token_len = (model_manager.llm_model.get_vision_tower().num_patches +
        len(input_ids[0]) - 1
        # -1 for the <image> token
    )
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sample["gt_answer"])
    gt_words = [token.lemma_.lower() for token in doc if not token.is_punct]
    doc = nlp(answer)
    generated_words = [token.lemma_.lower() for token in doc if not token.is_punct]
    print(answer)

    # Real words Calculation
    log_dict[sample["idx"]]["real_attn_contribution_across_layers"] = []
    log_dict[sample["idx"]]["visual_attn_weights"] = []
    log_dict[sample["idx"]]["real_SVAR_5_18"] = []
    # words_to_calculate = set(generated_words) & set(gt_words)
    words_to_calculate = set(generated_words)
    for ri, real_word in enumerate(words_to_calculate):
        # Calculate attn sublayer contribution for each real word
        print(real_word)
        try:
            # Get attn sublayer contribution
            _records = get_only_attn_out_contribution(
                model_manager.llm_model, model_manager.tokenizer,
                outputs, real_word, input_token_len-1,
            )
            # print(len(_records), _records[0].shape)
            # log_dict[sample["idx"]]["real_attn_contribution_across_layers"].append([float(x) for x in _records])

            # Get visual attention weights
            real_word_attnw_matrix, _ = attnw_over_vision_layer_head_selected_text(
                real_word, outputs, model_manager.tokenizer,
                vision_token_start, vision_token_end,
            )
            # print(real_word_attnw_matrix.shape)
            # log_dict[sample["idx"]]["visual_attn_weights"].append(real_word_attnw_matrix)
            real_word_layer_attnw = real_word_attnw_matrix.mean(axis=1)[::-1]
            log_dict[sample["idx"]]["real_SVAR_5_18"].append(real_word_layer_attnw[5:19].sum().item())
        except Exception as e:
            print(e)
            print(f"'{real_word}' not found in the generated text.")

    # Log the results
    log_dict[sample["idx"]]["uncertainty"] = sum(log_dict[sample["idx"]]["real_SVAR_5_18"]) / len(log_dict[sample["idx"]]["real_SVAR_5_18"])
    log_dict[sample["idx"]]["uncertainty_threshold"] = args.uncertainty_threshold
    flag_predict_hallucination = log_dict[sample["idx"]]["uncertainty"] >= args.uncertainty_threshold
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        log_dict[sample["idx"]]["flag_answer_correct"] and not flag_predict_hallucination
    ) or (
        not log_dict[sample["idx"]]["flag_answer_correct"] and flag_predict_hallucination
    )
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct

    # Lens
    discrete_range = [
        [391, 396],
        [415, 420],
        [328, 330],
        [352, 354],
        [376, 378],
        [379, 382],
        [403, 406],
    ]
    layer_range = [0, 5, 7, 10, 12, 15, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28, 30, 31]
    logits_warper = TopKLogitsWarper(top_k=50, filter_value=float("-inf"))
    logits_processor = LogitsProcessorList([])

    logitLens_of_vision_tokens_with_discrete_range(
        model_manager.llm_model, model_manager.tokenizer, input_ids, outputs,
        model_manager.img_start_idx, discrete_range,
        layer_range,
        logits_warper, logits_processor
    )
    return log_dict
