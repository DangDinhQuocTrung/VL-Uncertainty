import re
import math
import random
import collections

from utils.misc import *
from utils.textual_perturbation import *
from utils.visual_perturbation import *
from utils.constants import BENCHMARK_TYPE


def perturbation_of_visual_prompt(args, sample):
    perturbed_img_list = []
    if args.visual_perturbation == "blurring":
        for radius in args.blur_radius_list:
            perturbed_img_list.append(image_blurring(sample["img"], radius))
    elif args.visual_perturbation == "rotation":
        for degree in [-40, -20, 20, 40, 10]:
            perturbed_img_list.append(image_rotation(sample["img"], degree))
    elif args.visual_perturbation == "flipping":
        perturbed_img_list = [image_flipping(sample["img"], "horizontal")] * 2 + [
            image_flipping(sample["img"], "vertical")
        ] * 3
    elif args.visual_perturbation == "shifting":
        for dir in ["up", "down", "left", "right"]:
            perturbed_img_list.append((sample["img"], dir, 100))
        perturbed_img_list.append((sample["img"], "up", 50))
    elif args.visual_perturbation == "cropping":
        for ratio in [0.95, 0.9, 0.85, 0.8, 0.75]:
            perturbed_img_list.append(image_cropping(sample["img"], ratio))
    elif args.visual_perturbation == "erasing":
        for size in [50, 60, 70, 80, 90, 100]:
            perturbed_img_list.append(
                image_erasing(sample["img"], erase_l=size, erase_w=size)
            )
    elif args.visual_perturbation == "gaussian_noise":
        for degree in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_img_list.append(gaussian_noise(sample["img"], degree))
    elif args.visual_perturbation == "dropout":
        for degree in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_img_list.append(dropout(sample["img"], degree))
    elif args.visual_perturbation == "salt_and_pepper":
        for degree in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_img_list.append(salt_and_pepper(sample["img"], degree))
    elif args.visual_perturbation == "sharpen":
        for degree in [0.1, 0.2, 0.3, 0.4, 0.5]:
            perturbed_img_list.append(image_sharpen(sample["img"], degree))
    elif args.visual_perturbation == "adjust_brightness":
        for degree in [0.8, 0.9, 1.1, 1.2, 1.3]:
            perturbed_img_list.append(adjust_brightness(sample["img"], degree))
    elif args.visual_perturbation == "adjust_contrast":
        for degree in [0.8, 0.9, 1.1, 1.2, 1.3]:
            perturbed_img_list.append(adjust_contrast(sample["img"], degree))
    elif args.visual_perturbation == "rotate_shift":
        for degree in [-40, -20, 20, 40, 10]:
            perturbed_img_list.append(
                image_shifting(image_rotation(sample["img"], degree), "up", 100)
            )
    elif args.visual_perturbation == "crop_flip":
        for degree in [0.95, 0.9, 0.85, 0.8, 0.75]:
            perturbed_img_list.append(
                image_flipping(image_cropping(sample["img"], degree), "horizontal")
            )
    elif args.visual_perturbation == "rotate_blur":
        for degree in [-40, -20, 20, 40, 10]:
            perturbed_img_list.append(
                blur_image(image_rotation(sample["img"], degree), 1)
            )
    elif args.visual_perturbation == "crop_blur":
        for degree in [0.95, 0.9, 0.85, 0.8, 0.75]:
            perturbed_img_list.append(
                blur_image(image_cropping(sample["img"], degree), 1)
            )
    return perturbed_img_list


def perturbation_of_textual_prompt(args, sample, llm):
    perturbed_question_list = []
    original_question = parse_original_question(sample["question"])
    if args.textual_perturbation == "llm_rephrasing":
        for temp in args.textual_perturbation_temp_list:
            instruction = args.textual_perturbation_instruction_template.replace(
                "{question}", original_question
            )
            perturbed_question = llm.generate(instruction, temp)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    if args.textual_perturbation == "swapping":
        for _ in range(args.sampling_time):
            perturbed_question = word_swapping(original_question)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    elif args.textual_perturbation == "deleting":
        for _ in range(args.sampling_time):
            perturbed_question = word_deleting(original_question)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    elif args.textual_perturbation == "inserting":
        for _ in range(args.sampling_time):
            perturbed_question = word_inserting(original_question)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    elif args.textual_perturbation == "replacing":
        for _ in range(args.sampling_time):
            perturbed_question = word_replacing(original_question)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    elif args.textual_perturbation == "text_shuffle":
        for _ in range(args.sampling_time):
            perturbed_question = text_shuffle(original_question)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    elif args.textual_perturbation == "noise_injection":
        for noise_level in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_question = noise_injection(original_question, noise_level)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    elif args.textual_perturbation == "word_dropout":
        for dropout_rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_question = word_dropout(original_question, dropout_rate)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    elif args.textual_perturbation == "character_dropout":
        for dropout_rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_question = character_dropout(original_question, dropout_rate)
            perturbed_question_list.append(
                merge_question(perturbed_question, sample["question"])
            )
    return perturbed_question_list


def combination_of_perturbed_prompt(
    args, sample, perturbed_img_list, perturbed_question_list, log_dict
):
    perturbed_prompt_list = []
    if args.pair_order == "progressively":
        pass
    elif args.pair_order.startswith("shift"):
        shift_by = int(args.pair_order.split("_")[1])
        shift_by %= len(perturbed_question_list)
        perturbed_question_list = (
            perturbed_question_list[shift_by:] + perturbed_question_list[:shift_by]
        )
    elif args.pair_order == "random_pair":
        random.shuffle(perturbed_question_list)
    if args.pair_order != "progressively":
        log_dict[sample["idx"]][
            "perturbed_question_list_after_combination"
        ] = perturbed_question_list
    N = len(perturbed_img_list)
    for i in range(N):
        perturbed_prompt = sample.copy()
        perturbed_prompt["img"] = perturbed_img_list[i]
        perturbed_prompt["question"] = perturbed_question_list[i]
        perturbed_prompt_list.append(perturbed_prompt)
    return perturbed_prompt_list


def infer_single_sample(args, lvlm, sample, is_sampling, llm, log_dict):
    ans = lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp if not is_sampling else args.sampling_temp,
    )
    if not is_sampling:
        log_dict[sample["idx"]]["ans"] = ans
        flag_ans_correct = True
        if BENCHMARK_TYPE[args.benchmark] == "MULTI_CHOICE":
            flag_ans_correct = str(sample["gt_ans"]) in ans
        else:
            question = f"Ground truth: {sample['gt_ans']}. Model answer: {ans}. Please verify if the model ans matches the ground truth. Respond with either 'Correct' or 'Wrong' only."
            llm_ans_check = llm.generate(question, 0.1)
            log_dict[sample["idx"]]["llm_ans_check"] = llm_ans_check
            flag_ans_correct = (
                "Correct" in llm_ans_check
                or "correct" in llm_ans_check
                or "C" in llm_ans_check
                or "c" in llm_ans_check
            )
        log_dict[sample["idx"]]["flag_ans_correct"] = flag_ans_correct
    else:
        log_dict[sample["idx"]]["ans_sampling_list"].append(ans)
    return


def uncertainty_estimation(args, sample, llm, log_dict):
    ans_sampling_list = log_dict[sample["idx"]]["ans_sampling_list"]
    ans_cluster_idx = []
    if BENCHMARK_TYPE[args.benchmark] == "MULTI_CHOICE":
        for ans in ans_sampling_list:
            if (
                re.search(r"\d+", ans) is None
                or int(re.search(r"\d+", ans).group()) >= sample["num_c"]
            ):
                ans_cluster_idx.append(-1)
            else:
                ans_cluster_idx.append(int(re.search(r"\d+", ans).group()))
    else:
        ans_cluster_idx = [-1] * len(ans_sampling_list)
        cur_cluster_idx = 0
        log_dict[sample["idx"]]["entailment"] = {}
        for i in range(len(ans_sampling_list)):
            if ans_cluster_idx[i] == -1:
                ans_cluster_idx[i] = cur_cluster_idx
                for j in range(i + 1, len(ans_sampling_list)):
                    if ans_cluster_idx[j] == -1:
                        entailment_ij = llm.generate(
                            f"Does '{ans_sampling_list[i]}' entail '{ans_sampling_list[j]}'? Respond with either 'Yes' or 'No' only.",
                            0.1,
                        )
                        entailment_ji = llm.generate(
                            f"Does '{ans_sampling_list[j]}' entail '{ans_sampling_list[i]}'? Respond with either 'Yes' or 'No' only.",
                            0.1,
                        )
                        log_dict[sample["idx"]]["entailment"][
                            f"{i}_{j}"
                        ] = entailment_ij
                        log_dict[sample["idx"]]["entailment"][
                            f"{j}_{i}"
                        ] = entailment_ji
                        i_to_j = (
                            "Yes" in entailment_ij
                            or "yes" in entailment_ij
                            or "Y" in entailment_ij
                            or "y" in entailment_ij
                        )
                        j_to_i = (
                            "Yes" in entailment_ji
                            or "yes" in entailment_ji
                            or "Y" in entailment_ji
                            or "y" in entailment_ji
                        )
                        if i_to_j and j_to_i:
                            ans_cluster_idx[j] = cur_cluster_idx
                cur_cluster_idx += 1

    log_dict[sample["idx"]]["ans_cluster_idx"] = ans_cluster_idx

    cluster_dis = collections.Counter(ans_cluster_idx)
    log_dict[sample["idx"]]["cluster_dis"] = cluster_dis
    uncertainty = -sum(
        (cnt / args.sampling_time) * math.log2(cnt / args.sampling_time)
        for cnt in cluster_dis.values()
    )
    log_dict[sample["idx"]]["uncertainty"] = uncertainty


def hallucination_detection(args, sample, log_dict):
    flag_predict_hallucination = (
        log_dict[sample["idx"]]["uncertainty"] >= args.uncertainty_thres
    )
    log_dict[sample["idx"]]["uncertainty_thres"] = args.uncertainty_thres
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination

    flag_detection_correct = (
        log_dict[sample["idx"]]["flag_ans_correct"] and not flag_predict_hallucination
    ) or (
        not log_dict[sample["idx"]]["flag_ans_correct"] and flag_predict_hallucination
    )
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct


def vl_uncertainty(args, lvlm, sample, llm, log_dict):
    perturbed_img_list = perturbation_of_visual_prompt(args, sample)
    perturbed_question_list = perturbation_of_textual_prompt(args, sample, llm)
    log_dict[sample["idx"]]["perturbed_question_list"] = perturbed_question_list
    perturbed_prompt_list = combination_of_perturbed_prompt(
        args, sample, perturbed_img_list, perturbed_question_list, log_dict
    )

    log_dict[sample["idx"]]["ans_sampling_list"] = []
    for i in range(args.sampling_time):
        infer_single_sample(args, lvlm, perturbed_prompt_list[i], True, llm, log_dict)

    uncertainty_estimation(args, sample, llm, log_dict)
    hallucination_detection(args, sample, log_dict)


def semantic_entropy(args, lvlm, sample, llm, log_dict):
    log_dict[sample["idx"]]["ans_sampling_list"] = []
    for _ in range(args.sampling_time):
        infer_single_sample(args, lvlm, sample, True, llm, log_dict)

    uncertainty_estimation(args, sample, llm, log_dict)
    hallucination_detection(args, sample, log_dict)


def estimate_uncertainty_by_vl_or_semantic_entropy(args, lvlm, sample, llm, log_dict):
    infer_single_sample(args, lvlm, sample, False, llm, log_dict)
    if args.uncertainty == "vl_uncertainty":
        vl_uncertainty(args, lvlm, sample, llm, log_dict)
    elif args.uncertainty == "semantic_entropy":
        semantic_entropy(args, lvlm, sample, llm, log_dict)
    return log_dict
