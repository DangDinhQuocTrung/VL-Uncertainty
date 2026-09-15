from methods.nli import get_nli_classifier
from methods.distance_utils import cosine_distance_matrix


def check_nli():
    nli_classifier = get_nli_classifier()
    print(nli_classifier.pairwise_semantic_distance(["The sky is blue.", "The sky is red."]))
    print(nli_classifier.pairwise_semantic_distance(["The sky is not blue.", "The sky is red."]))
    print(nli_classifier.pairwise_semantic_distance(["The sky is blue.", "The sky is not blue."]))

    print(cosine_distance_matrix(["The sky is blue.", "The sky is red."], "all-MiniLM-L6-v2", "cuda"))
    print(cosine_distance_matrix(["The sky is not blue.", "The sky is red."], "all-MiniLM-L6-v2", "cuda"))
    print(cosine_distance_matrix(["The sky is blue.", "The sky is not blue."], "all-MiniLM-L6-v2", "cuda"))


if __name__ == "__main__":
    check_nli()
