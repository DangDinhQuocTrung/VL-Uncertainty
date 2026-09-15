"""DeBERTa-large MNLI classifier for semantic clustering.

Matches Kuhn et al. (ICML 2023, "Semantic Uncertainty"): concatenate the
question with each generated answer, then treat two answers as equivalent
iff they bidirectionally entail each other under microsoft/deberta-large-mnli.
MNLI labels: 0=contradiction, 1=neutral, 2=entailment.
"""

import numpy as np
import torch

_NLI_CLASSIFIER = None

MNLI_ID2LABEL = {
    0: "contradiction",
    1: "neutral",
    2: "entailment",
}


class DebertaNLIClassifier:
    def __init__(self, model_name="microsoft/deberta-large-mnli", device=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {
            int(k): str(v).lower() for k, v in self.model.config.id2label.items()
        }

    def _with_question(self, text, question):
        text = "" if text is None else str(text).strip()
        if not question:
            return text
        return f"{question} {text}".strip()

    @torch.no_grad()
    def classify_pair(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0]
        pred_id = int(torch.argmax(logits, dim=-1).item())
        return pred_id, self.id2label.get(pred_id, MNLI_ID2LABEL.get(pred_id, str(pred_id)))

    @torch.no_grad()
    def classify_bidirectional(self, text1, text2):
        inputs = self.tokenizer(
            [text1, text2],
            [text2, text1],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        labels = [
            self.id2label.get(int(pred_id), MNLI_ID2LABEL.get(int(pred_id), str(pred_id)))
            for pred_id in pred_ids
        ]
        return pred_ids, labels

    def _entailment_class_id(self):
        for idx, label in self.id2label.items():
            if "entail" in str(label).lower():
                return int(idx)
        return 2

    @torch.no_grad()
    def classify_pairs(self, premises, hypotheses, batch_size=8):
        """Return logits of shape (N, C) for paired (premise, hypothesis) inputs."""
        if not premises:
            return torch.zeros((0, self.model.config.num_labels), device=self.device)

        logit_chunks = []
        for start in range(0, len(premises), batch_size):
            inputs = self.tokenizer(
                premises[start : start + batch_size],
                hypotheses[start : start + batch_size],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logit_chunks.append(self.model(**inputs).logits)
        return torch.cat(logit_chunks, dim=0)

    def pairwise_semantic_distance(self, texts, question=None, batch_size=8):
        """Symmetric DeBERTa-MNLI distance matrix.

        d(a, a') = 1 - 0.5 * (P_entail(a -> a') + P_entail(a' -> a)).
        Identical strings have distance 0.
        """
        n = len(texts)
        dist = np.zeros((n, n), dtype=np.float64)
        if n <= 1:
            return dist

        cleaned = ["" if t is None else str(t).strip() for t in texts]
        pair_index = []
        premises = []
        hypotheses = []
        for i in range(n):
            for j in range(i + 1, n):
                if cleaned[i] == cleaned[j]:
                    continue
                pair_index.append((i, j))
                left = self._with_question(cleaned[i], question)
                right = self._with_question(cleaned[j], question)
                premises.extend([left, right])
                hypotheses.extend([right, left])

        if pair_index:
            logits = self.classify_pairs(premises, hypotheses, batch_size=batch_size)
            probs = torch.softmax(logits.float(), dim=-1)
            ent_id = self._entailment_class_id()
            for k, (i, j) in enumerate(pair_index):
                p_ij = probs[2 * k, ent_id].item()
                p_ji = probs[2 * k + 1, ent_id].item()
                distance = 1.0 - 0.5 * (p_ij + p_ji)
                dist[i, j] = dist[j, i] = max(0.0, distance)

        return dist

    def bidirectional_entailment(self, text1, text2, question=None):
        """Return (equivalent, label_12, label_21) using strict bidirectional entailment."""
        ans1 = "" if text1 is None else str(text1).strip()
        ans2 = "" if text2 is None else str(text2).strip()
        if ans1 == ans2:
            return True, "entailment", "entailment"

        premise1 = self._with_question(ans1, question)
        premise2 = self._with_question(ans2, question)
        pred_ids, labels = self.classify_bidirectional(premise1, premise2)
        equivalent = pred_ids[0] == 2 and pred_ids[1] == 2
        return equivalent, labels[0], labels[1]


def get_nli_classifier(model_name="microsoft/deberta-large-mnli", device=None):
    global _NLI_CLASSIFIER
    cache_key = (model_name, str(device))
    if (
        _NLI_CLASSIFIER is not None
        and getattr(_NLI_CLASSIFIER, "_cache_key", None) == cache_key
    ):
        return _NLI_CLASSIFIER
    classifier = DebertaNLIClassifier(model_name=model_name, device=device)
    classifier._cache_key = cache_key
    _NLI_CLASSIFIER = classifier
    return classifier
