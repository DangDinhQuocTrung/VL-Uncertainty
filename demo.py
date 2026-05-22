import warnings

warnings.filterwarnings("ignore")
import collections
import math
import random

import torch

from llm.Qwen import Qwen
from lvlm.InternVL import InternVL
from utils.misc import *
from utils.textual_perturbation import *
from utils.visual_perturbation import *


def handle_demo(lvlm, llm):
    sample = {}
    sample["img"] = Image.open(".asset/img/titanic.png")
    sample["question"] = "What is the name of this movie?"
    sample["gt_ans"] = "Titanic."
    print("-" * 50)
    print("- Demo image: .asset/img/titanic.png")
    print("- Question: What is the name of this movie?")
    print("- GT answer: Titanic.")
    print("-" * 50)

    ans = lvlm.generate(sample["img"], sample["question"], 0.1)
    print(f"- LVLM answer: {ans}")
    flag_ans_correct = True
    question = f"Ground truth: {sample['gt_ans']}. Model answer: {ans}. Please verify if the model ans matches the ground truth. Respond with either 'Correct' or 'Wrong' only."
    llm_ans_check = llm.generate(question, 0.1)
    flag_ans_correct = (
        "Correct" in llm_ans_check
        or "correct" in llm_ans_check
        or "C" in llm_ans_check
        or "c" in llm_ans_check
    )
    print(f"- LVLM answer accuracy: {'Correct' if flag_ans_correct else 'Wrong'}")
    print("-" * 50)

    ans_sampling_list = []
    perturbed_img_list = []
    for radius in [1, 2, 3, 4, 5]:
        perturbed_img_list.append(image_blurring(sample["img"], radius))
    perturbed_question_list = []
    original_question = parse_original_question(sample["question"])
    textual_perturbation_instruction_template = "Given the input question: '{question}', generate a semantically equivalent variation by changing the wording, structure, grammar, or narrative. Ensure the perturbed question maintains the same meaning as the original. Provide only the rephrased question as the output."
    for temp in [0.6, 0.7, 0.8, 0.9, 1.0]:
        instruction = textual_perturbation_instruction_template.replace(
            "{question}", original_question
        )
        perturbed_question = llm.generate(instruction, temp)
        perturbed_question_list.append(
            merge_question(perturbed_question, sample["question"])
        )
    for i in range(5):
        ans = lvlm.generate(perturbed_img_list[i], perturbed_question_list[i], 1.0)
        ans_sampling_list.append(ans)

    ans_cluster_idx = [-1] * len(ans_sampling_list)
    cur_cluster_idx = 0
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
    cluster_dis = collections.Counter(ans_cluster_idx)
    uncertainty = -sum((cnt / 5) * math.log2(cnt / 5) for cnt in cluster_dis.values())
    print(f"- Estimated uncertianty: {uncertainty}")
    flag_predict_hallucination = uncertainty >= 1.0
    print(f"- Uncertianty threshold: 1.0")
    print("-" * 50)

    print(
        f"- Hallucination prediction: {'Is hallucination' if flag_predict_hallucination else 'Is not hallucination'}"
    )
    flag_detection_correct = (flag_ans_correct and not flag_predict_hallucination) or (
        not flag_ans_correct and flag_predict_hallucination
    )
    print(
        f"- Hallucination detection: {'Success!' if flag_detection_correct else 'Fail'}"
    )
    print("-" * 50)


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    fix_seed(0)
    lvlm = InternVL("InternVL2-1B")
    llm = Qwen("Qwen2.5-3B-Instruct")
    handle_demo(lvlm, llm)


if __name__ == "__main__":
    main()


"""
--------------------------------------------------
- Demo image: .asset/img/titanic.png
- Question: What is the name of this movie?
- GT answer: Titanic.
--------------------------------------------------
- LVLM answer: The movie in the image is "Coco."
- LVLM answer accuracy: Wrong
--------------------------------------------------
- Estimated uncertianty: 2.321928094887362
- Uncertianty threshold: 1.0
--------------------------------------------------
- Hallucination prediction: Is hallucination
- Hallucination detection: Success!
--------------------------------------------------
"""
