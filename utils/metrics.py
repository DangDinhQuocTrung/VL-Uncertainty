from torchmetrics.functional import f1_score
from sklearn.metrics import precision_recall_curve


def compute_f1_score(outputs, targets, seeking=False, eps=1e-8):
    if not seeking:
        f1_score_result = f1_score(outputs, targets.long(), task="binary")
        best_threshold = None
    else:
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        outputs = outputs[..., -1] if outputs.ndim >= 2 and outputs.shape[-1] == 2 else outputs
        # print(predictions.shape, targets.shape, torch.sum(targets), (predictions == targets).sum())
        precisions, recalls, thresholds = precision_recall_curve(targets, outputs)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + eps)
        best_index = f1_scores.argmax()
        best_threshold = thresholds[best_index].item()
        f1_score_result = f1_scores[best_index].item()
    return f1_score_result, best_threshold
