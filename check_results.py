import json
from pathlib import Path

import torch
from torchmetrics.functional import auroc, average_precision


def check_results():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    json_path = Path("/zhome/05/8/227717/VL-Uncertainty/exp/log_2026_05_31_00_44_20.json")
    with open(json_path, "r") as f:
        log_dict = json.load(f)

    flag_answer_correct = []
    uncertainty_scores = []
    for key, value in log_dict.items():
        if not key.isdigit():
            continue
        flag_answer_correct.append(value["flag_answer_correct"])
        uncertainty_scores.append(value["clean_entropy"] + 0.01 * value["image_score"])

    flag_answer_correct = torch.tensor(flag_answer_correct, dtype=torch.long).to(device)
    uncertainty_scores = torch.tensor(uncertainty_scores, dtype=torch.float32).to(device)

    task = "binary"
    incorrectness_gt = 1 - flag_answer_correct
    auroc_score = auroc(uncertainty_scores, incorrectness_gt, task=task)
    auprc_score = average_precision(uncertainty_scores, incorrectness_gt, task=task)

    print(f"AUROC: {auroc_score:.4f}")
    print(f"AUPRC: {auprc_score:.4f}")


if __name__ == "__main__":
    check_results()
