import torch
import gc
from utils.constants import BENCHMARK_TYPE
from methods.euq.evidence import EvidenceModel


def estimate_uncertainty_by_euq(args, lvlm, sample, llm, log_dict):
    # Inference
    answer, down_proj_features, llm_head_features = lvlm.generate(
        sample["img"],
        sample["question"],
        0.2,
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

    # Evidence model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(f"{lvlm.weight_dir}/{lvlm.version}_attention_weights.pth")
    head_state_dict = torch.load(f"{lvlm.weight_dir}/{lvlm.version}_head_weights.pth")
    evidence_model = EvidenceModel(state_dict)
    head_evidence_model = EvidenceModel(head_state_dict)

    # Compute evidence
    conflict_value = 0.0
    ig_value = 0.0
    length_down_proj_features = len(down_proj_features)
    processed_features = []
    for feature in down_proj_features:
        processed_features.append(feature.squeeze(0))
    for feature in processed_features:
        evidence_weights = evidence_model.get_evidence_weights(feature.squeeze(0).T)
        conflict_value += evidence_model.get_evidence_conflict().item()
        ig_value += evidence_model.get_evidence_ignorance().item()
    del evidence_model, state_dict, down_proj_features, processed_features
    torch.cuda.empty_cache()
    gc.collect()

    head_conflict_value = 0.0
    head_ig_value = 0.0
    length_llm_head_features = len(llm_head_features)
    processed_features_head = []
    for feature in llm_head_features:
        processed_features_head.append(feature)
    for feature in processed_features_head:
        head_evidence_weights = head_evidence_model.get_evidence_weights(feature.T)
        head_conflict_value += head_evidence_model.get_evidence_conflict().item()
        head_ig_value += head_evidence_model.get_evidence_ignorance().item()
    del head_evidence_model, head_state_dict, llm_head_features, processed_features_head
    torch.cuda.empty_cache()
    gc.collect()

    mean_conflict_value = conflict_value / length_down_proj_features
    mean_ig_value = ig_value / length_down_proj_features
    mean_head_conflict_value = head_conflict_value / length_llm_head_features
    mean_head_ig_value = head_ig_value / length_llm_head_features
    log_dict[sample["idx"]]["mean_conflict_value"] = mean_conflict_value
    log_dict[sample["idx"]]["mean_ignorance_value"] = mean_ig_value
    log_dict[sample["idx"]]["mean_head_conflict_value"] = mean_head_conflict_value
    log_dict[sample["idx"]]["mean_head_ignorance_value"] = mean_head_ig_value
    sample_conflict_value = (mean_conflict_value + mean_head_conflict_value) / 2
    sample_ignorance_value = (mean_ig_value + mean_head_ig_value) / 2
    total_uncertainty = sample_conflict_value + sample_ignorance_value

    # Log the results
    log_dict[sample["idx"]]["uncertainty"] = total_uncertainty
    log_dict[sample["idx"]]["uncertainty_thres"] = args.uncertainty_thres
    flag_predict_hallucination = log_dict[sample["idx"]]["uncertainty"] >= args.uncertainty_thres
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        log_dict[sample["idx"]]["flag_answer_correct"] and not flag_predict_hallucination
    ) or (
        not log_dict[sample["idx"]]["flag_answer_correct"] and flag_predict_hallucination
    )
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct
    return log_dict
