import torch
import torch.nn.functional as F
import numpy as np

from methods.nli import get_nli_classifier

_EMBED_MODEL = None


def _get_embed_model(model_name, device):
    global _EMBED_MODEL
    if _EMBED_MODEL is not None and getattr(_EMBED_MODEL, "_rds_name", None) == model_name:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "RDS requires the sentence-transformers package. "
            "Install with: pip install sentence-transformers"
        ) from exc
    model = SentenceTransformer(model_name, device=str(device))
    model._rds_name = model_name
    _EMBED_MODEL = model
    return model


def embed_answers(answers, model_name="all-MiniLM-L6-v2", device=None):
    """Embed answers and L2-normalize onto the unit hypersphere."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = _get_embed_model(model_name, device)
    texts = [a if (a is not None and str(a).strip()) else " " for a in answers]
    embeddings = embed_model.encode(texts, convert_to_tensor=True, device=device)
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    return embeddings


def cosine_distance_matrix(answers, embed_model_name, device):
    embeddings = embed_answers(answers, model_name=embed_model_name, device=device)
    similarity = embeddings @ embeddings.T
    dist = (1.0 - similarity).clamp(min=0.0).detach().cpu().numpy().astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    return dist


def deberta_distance_matrix(args, answers, question):
    nli = get_nli_classifier(
        model_name=getattr(args, "nli_model", "microsoft/deberta-v2-xlarge-mnli"),
        device=getattr(args, "nli_device", "auto"),
    )
    return nli.pairwise_semantic_distance(answers, question=question)
