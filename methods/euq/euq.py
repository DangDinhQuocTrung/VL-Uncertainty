import os
import torch
import gc
from pathlib import Path
from utils.constants import is_choice_question
from methods.euq.evidence import EvidenceModel
from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm, evaluate_multiple_choice_answer_correctness
from utils.visual_statistics import maybe_log_visual_statistics


def estimate_uncertainty_by_euq(args, lvlm, sample, llm, log_dict):
    index = sample["idx"]
    if getattr(llm, "model", None) is not None and hasattr(llm.model, "device"):
        device = llm.model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dir = Path(lvlm.weight_dir) if lvlm is not None else Path(log_dict["weight_dir"])
    lvlm_version = lvlm.version if lvlm is not None else log_dict["lvlm_version"]
    feature_weight_dir = weight_dir / "features"
    feature_weight_dir.mkdir(parents=True, exist_ok=True)

    # Inference
    if args.split_inference_quantification in [0, 1]:
        log_dict["weight_dir"] = str(lvlm.weight_dir)
        log_dict["lvlm_version"] = lvlm.version
        answer, down_proj_features, llm_head_features = lvlm.generate(
            sample["img"],
            sample["question"],
            args.inference_temp,
            return_more=True,
        )
        log_dict[sample["idx"]]["answer"] = answer
        flag_answer_correct = True
        if is_choice_question(args, sample):
            flag_answer_correct, llm_answer_check = evaluate_multiple_choice_answer_correctness(llm, sample, answer)
        else:
            flag_answer_correct, llm_answer_check = evaluate_answer_correctness_by_llm(llm, sample, answer)
        log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
        log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct
        log_dict[sample["idx"]]["answer_sampling_list"] = [answer]
        maybe_log_visual_statistics(args, lvlm, sample, log_dict)
    else:
        answer = log_dict[sample["idx"]]["answer"]
        down_proj_features = torch.load(feature_weight_dir / f"{lvlm_version}_sample_{index:04d}_down_proj_features.pth", map_location=device)
        llm_head_features = torch.load(feature_weight_dir / f"{lvlm_version}_sample_{index:04d}_head_features.pth", map_location=device)

    # Evidence model
    state_dict = torch.load(weight_dir / f"{lvlm_version}_attention_weights.pth", map_location=device)
    head_state_dict = torch.load(weight_dir / f"{lvlm_version}_head_weights.pth", map_location=device)
    evidence_model = EvidenceModel(state_dict)
    head_evidence_model = EvidenceModel(head_state_dict)

    # Compute evidence
    conflict_value = 0.0
    ig_value = 0.0
    length_down_proj_features = len(down_proj_features)
    processed_features = []
    if args.split_inference_quantification == 1:
        torch.save(down_proj_features, feature_weight_dir / f"{lvlm_version}_sample_{index:04d}_down_proj_features.pth")
    else:
        for feature in down_proj_features:
            processed_features.append(feature.squeeze(0))
        for feature in processed_features:
            evidence_weights = evidence_model.get_evidence_weights(feature.squeeze(0).T)
            conflict_value += evidence_model.get_evidence_conflict().item()
            ig_value += evidence_model.get_evidence_ignorance().item()
        del evidence_model, state_dict, down_proj_features, processed_features
        gc.collect()
        torch.cuda.empty_cache()
        (feature_weight_dir / f"{lvlm_version}_sample_{index:04d}_down_proj_features.pth").unlink(missing_ok=True)

    head_conflict_value = 0.0
    head_ig_value = 0.0
    length_llm_head_features = len(llm_head_features)
    processed_features_head = []
    if args.split_inference_quantification == 1:
        torch.save(llm_head_features, feature_weight_dir / f"{lvlm_version}_sample_{index:04d}_head_features.pth")
    else:
        for feature in llm_head_features:
            feature.to("cpu")
            processed_features_head.append(feature)
        for feature_index in range(length_llm_head_features):
            feature = processed_features_head[feature_index]
            head_evidence_weights = head_evidence_model.get_evidence_weights(feature.T)
            head_conflict_value += head_evidence_model.get_evidence_conflict().item()
            head_ig_value += head_evidence_model.get_evidence_ignorance().item()
            processed_features_head[feature_index] = None
            feature.to("cpu")
            del feature
            gc.collect()
            torch.cuda.empty_cache()
        del head_evidence_model, head_state_dict, llm_head_features, processed_features_head
        gc.collect()
        torch.cuda.empty_cache()
        (feature_weight_dir / f"{lvlm_version}_sample_{index:04d}_head_features.pth").unlink(missing_ok=True)

    if args.split_inference_quantification == 1:
        return log_dict

    mean_conflict_value = conflict_value / max(length_down_proj_features, 1)
    mean_ig_value = ig_value / max(length_down_proj_features, 1)
    mean_head_conflict_value = head_conflict_value / max(length_llm_head_features, 1)
    mean_head_ig_value = head_ig_value / max(length_llm_head_features, 1)
    log_dict[sample["idx"]]["mean_conflict_value"] = mean_conflict_value
    log_dict[sample["idx"]]["mean_ignorance_value"] = mean_ig_value
    log_dict[sample["idx"]]["mean_head_conflict_value"] = mean_head_conflict_value
    log_dict[sample["idx"]]["mean_head_ignorance_value"] = mean_head_ig_value
    sample_conflict_value = mean_conflict_value + mean_head_conflict_value
    sample_ignorance_value = mean_ig_value + mean_head_ig_value
    total_uncertainty = sample_conflict_value + sample_ignorance_value

    # Log the results
    log_dict[sample["idx"]]["uncertainty"] = total_uncertainty
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
