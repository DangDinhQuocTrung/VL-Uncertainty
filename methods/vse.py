"""Visual Semantic Entropy (VSE) uncertainty estimation.

Paper: Visual Semantic Entropy: Do Vision Language Models Recognize Visual Ambiguity?
       arXiv:2606.31407

VSE keeps the question fixed, perturbs only the image, clusters sampled answers
into semantic prototypes, and scores uncertainty as the mass-weighted pairwise
distance among those prototypes.
"""

import collections

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from methods.nli import get_nli_classifier
from methods.vl_uncertainty import hallucination_detection, infer_single_sample
from utils.misc import parse_original_question
from utils.visual_perturbation import gaussian_noise


def perturbation_of_visual_prompt_vse(args, sample):
    """Generate M local visual variants at a single noise scale.

    Unlike VL-Uncertainty, VSE does not increase perturbation magnitude across
    samples and does not paraphrase the question.
    """
    num_views = int(getattr(args, "sampling_time", 10))
    sigma = float(getattr(args, "vse_noise_sigma", 20.0))
    degree = sigma / 255.0
    return [gaussian_noise(sample["img"], degree) for _ in range(num_views)]


def _cosine_distance_matrix(answers, embed_model_name, device):
    from methods.rds import embed_answers

    embeddings = embed_answers(answers, model_name=embed_model_name, device=device)
    similarity = embeddings @ embeddings.T
    dist = (1.0 - similarity).clamp(min=0.0).detach().cpu().numpy().astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    return dist


def _deberta_distance_matrix(args, answers, question):
    nli = get_nli_classifier(
        model_name=getattr(
            args, "vse_nli_model", "microsoft/deberta-v2-xlarge-mnli"
        ),
        device=getattr(args, "nli_device", "auto"),
    )
    return nli.pairwise_semantic_distance(answers, question=question)


def compute_semantic_distance_matrix(args, answers, question, device=None):
    distance_fn = getattr(args, "vse_distance", "deberta").lower()
    if distance_fn == "cosine":
        embed_model = getattr(args, "vse_embed_model", None) or getattr(
            args, "rds_embed_model", "all-MiniLM-L6-v2"
        )
        return _cosine_distance_matrix(answers, embed_model, device)
    if distance_fn != "deberta":
        raise ValueError(
            f"Unsupported VSE distance '{distance_fn}'. Expected 'deberta' or 'cosine'."
        )
    return _deberta_distance_matrix(args, answers, question)


def hierarchical_cluster(distance_matrix, threshold, linkage="average"):
    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=int)
    if n == 1:
        return np.zeros((1,), dtype=int)

    dist = np.array(distance_matrix, dtype=np.float64, copy=True)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    kwargs = {
        "n_clusters": None,
        "linkage": linkage,
        "distance_threshold": float(threshold),
    }
    clustering = AgglomerativeClustering(metric="precomputed", **kwargs)
    return clustering.fit_predict(dist)


def _cluster_indices(labels):
    groups = collections.OrderedDict()
    for idx, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), []).append(idx)
    return list(groups.values())


def select_prototypes(distance_matrix, cluster_groups):
    """Medoid of each cluster: argmin_a sum_{a' in cluster} d(a, a')."""
    prototype_indices = []
    for members in cluster_groups:
        if len(members) == 1:
            prototype_indices.append(members[0])
            continue
        sub = distance_matrix[np.ix_(members, members)]
        prototype_indices.append(members[int(np.argmin(sub.sum(axis=1)))])
    return prototype_indices


def prototype_dispersion(distance_matrix, cluster_groups, prototype_indices):
    """Mass-weighted pairwise distance among prototypes (Eq. 8)."""
    n_samples = distance_matrix.shape[0]
    if n_samples == 0 or len(cluster_groups) <= 1:
        weights = [len(members) / max(n_samples, 1) for members in cluster_groups]
        return 0.0, weights

    weights = np.array(
        [len(members) / n_samples for members in cluster_groups], dtype=np.float64
    )
    proto_dist = distance_matrix[np.ix_(prototype_indices, prototype_indices)]
    np.fill_diagonal(proto_dist, 0.0)
    uncertainty = float(weights @ proto_dist @ weights)
    return uncertainty, weights.tolist()


def prototype_semantic_aggregation(args, sample, log_dict, device=None):
    answers = log_dict[sample["idx"]]["answer_sampling_list"]
    n_samples = len(answers)
    log_dict[sample["idx"]]["vse_distance"] = getattr(args, "vse_distance", "deberta")
    log_dict[sample["idx"]]["vse_cluster_threshold"] = float(
        getattr(args, "vse_cluster_threshold", 0.5)
    )

    if n_samples == 0:
        log_dict[sample["idx"]]["answer_cluster_idx"] = []
        log_dict[sample["idx"]]["cluster_dis"] = {}
        log_dict[sample["idx"]]["vse_prototypes"] = []
        log_dict[sample["idx"]]["vse_weights"] = []
        log_dict[sample["idx"]]["uncertainty"] = 0.0
        return

    question = parse_original_question(sample["question"])
    distance_matrix = compute_semantic_distance_matrix(
        args, answers, question, device=device
    )
    labels = hierarchical_cluster(
        distance_matrix,
        threshold=getattr(args, "vse_cluster_threshold", 0.5),
    )
    cluster_groups = _cluster_indices(labels)
    prototype_indices = select_prototypes(distance_matrix, cluster_groups)
    uncertainty, weights = prototype_dispersion(
        distance_matrix, cluster_groups, prototype_indices
    )

    cluster_idx = [int(label) for label in labels.tolist()]
    log_dict[sample["idx"]]["answer_cluster_idx"] = cluster_idx
    log_dict[sample["idx"]]["cluster_dis"] = collections.Counter(cluster_idx)
    log_dict[sample["idx"]]["vse_distance_matrix"] = np.round(
        distance_matrix, 6
    ).tolist()
    log_dict[sample["idx"]]["vse_prototype_indices"] = prototype_indices
    log_dict[sample["idx"]]["vse_prototypes"] = [answers[i] for i in prototype_indices]
    log_dict[sample["idx"]]["vse_weights"] = weights
    log_dict[sample["idx"]]["uncertainty"] = uncertainty


def visual_semantic_entropy(args, lvlm, sample, llm, log_dict):
    perturbed_img_list = perturbation_of_visual_prompt_vse(args, sample)
    log_dict[sample["idx"]]["vse_noise_sigma"] = float(
        getattr(args, "vse_noise_sigma", 20.0)
    )
    log_dict[sample["idx"]]["vse_num_views"] = len(perturbed_img_list)

    log_dict[sample["idx"]]["answer_sampling_list"] = []
    for perturbed_img in perturbed_img_list:
        perturbed_sample = sample.copy()
        perturbed_sample["img"] = perturbed_img
        infer_single_sample(args, lvlm, perturbed_sample, True, llm, log_dict)

    device = getattr(lvlm, "device", None)
    prototype_semantic_aggregation(args, sample, log_dict, device=device)
    hallucination_detection(args, sample, log_dict)


def estimate_uncertainty_by_vse(args, lvlm, sample, llm, log_dict):
    infer_single_sample(args, lvlm, sample, False, llm, log_dict)
    visual_semantic_entropy(args, lvlm, sample, llm, log_dict)
    return log_dict
