"""Select VSE (or similar) samples that are confidently wrong and homogeneous.

Finds cases where:
  - the answer is incorrect
  - uncertainty is low
  - answer_sampling_list is nearly uniform (string-level and/or cluster-level)

Useful for inspecting VSE failure modes on ViLP (prior-locked wrong answers).

Example:
  PYTHONPATH=. python plotting/select_confident_wrong_homogeneous.py \\
      --log exp_02/ViLP_vse_2026_09_03_22_51_14_revised.json \\
      --max_uncertainty 0.0 \\
      --min_majority 1.0 \\
      --top_k 20
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import LOG_META_KEYS


def load_log(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r") as f:
        return json.load(f)


def _is_sample_entry(key: str, value: Any) -> bool:
    if key in LOG_META_KEYS or not isinstance(value, dict):
        return False
    if "Hallucination detection" in key or "Perturbation detection" in key:
        return False
    return bool(value.get("flag_sample_valid", True)) and "uncertainty" in value


def _normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def sampling_homogeneity(answers: List[Any]) -> Dict[str, float]:
    """String-level homogeneity of the sampled answer list."""
    norms = [_normalize_answer(a) for a in answers]
    n = len(norms)
    if n == 0:
        return {
            "n_samples": 0.0,
            "n_unique": 0.0,
            "majority_frac": 0.0,
            "unique_frac": 1.0,
            "majority_answer": "",
        }
    counts = Counter(norms)
    majority_answer, majority_count = counts.most_common(1)[0]
    return {
        "n_samples": float(n),
        "n_unique": float(len(counts)),
        "majority_frac": majority_count / n,
        "unique_frac": len(counts) / n,
        "majority_answer": majority_answer,
    }


def cluster_homogeneity(value: Dict[str, Any]) -> Dict[str, float]:
    """Cluster-level homogeneity from VSE fields when present."""
    cluster_dis = value.get("cluster_dis")
    cluster_idx = value.get("answer_cluster_idx")
    if isinstance(cluster_dis, dict) and cluster_dis:
        counts = [int(c) for c in cluster_dis.values()]
        total = sum(counts)
        return {
            "n_clusters": float(len(counts)),
            "cluster_majority_frac": (max(counts) / total) if total else 0.0,
        }
    if isinstance(cluster_idx, list) and cluster_idx:
        counts = Counter(cluster_idx)
        total = len(cluster_idx)
        return {
            "n_clusters": float(len(counts)),
            "cluster_majority_frac": max(counts.values()) / total,
        }
    return {"n_clusters": float("nan"), "cluster_majority_frac": float("nan")}


def select_samples(
    log: Dict[str, Any],
    max_uncertainty: float,
    min_majority: float,
    max_unique_frac: float,
    max_clusters: Optional[int],
    require_single_cluster: bool,
) -> List[Dict[str, Any]]:
    selected = []
    for key, value in log.items():
        if not _is_sample_entry(key, value):
            continue
        if bool(value.get("flag_answer_correct", True)):
            continue

        uncertainty = float(value["uncertainty"])
        if uncertainty > max_uncertainty:
            continue

        answers = value.get("answer_sampling_list", [])
        if not isinstance(answers, list) or not answers:
            continue

        string_stats = sampling_homogeneity(answers)
        if string_stats["majority_frac"] < min_majority:
            continue
        if string_stats["unique_frac"] > max_unique_frac:
            continue

        cluster_stats = cluster_homogeneity(value)
        n_clusters = cluster_stats["n_clusters"]
        if require_single_cluster and n_clusters == n_clusters:  # not NaN
            if n_clusters > 1:
                continue
        if max_clusters is not None and n_clusters == n_clusters:
            if n_clusters > max_clusters:
                continue

        selected.append(
            {
                "idx": int(key) if str(key).isdigit() else key,
                "uncertainty": uncertainty,
                "answer": value.get("answer"),
                "gt_answer": value.get("gt_answer"),
                "question": value.get("question"),
                "answer_sampling_list": answers,
                "answer_cluster_idx": value.get("answer_cluster_idx"),
                "cluster_dis": value.get("cluster_dis"),
                "llm_answer_check": value.get("llm_answer_check"),
                **string_stats,
                **cluster_stats,
            }
        )

    # Most homogeneous first, then lowest uncertainty.
    selected.sort(
        key=lambda s: (
            -s["majority_frac"],
            s["unique_frac"],
            s["n_clusters"] if s["n_clusters"] == s["n_clusters"] else 99.0,
            s["uncertainty"],
            s["idx"] if isinstance(s["idx"], int) else str(s["idx"]),
        )
    )
    return selected


def _shorten(text: Any, max_len: int = 160) -> str:
    s = " ".join(str(text).split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def print_examples(samples: List[Dict[str, Any]], top_k: int) -> None:
    show = samples[:top_k]
    print(f"Selected {len(samples)} samples; showing top {len(show)}:\n")
    for i, s in enumerate(show, 1):
        n_cl = s["n_clusters"]
        n_cl_str = f"{int(n_cl)}" if n_cl == n_cl else "n/a"
        print(f"[{i}] idx={s['idx']}  U={s['uncertainty']:.4f}  "
              f"majority={s['majority_frac']:.2f}  unique={int(s['n_unique'])}  "
              f"clusters={n_cl_str}")
        print(f"    pred: {s['answer']}   gt: {s['gt_answer']}")
        print(f"    samples: {s['answer_sampling_list']}")
        print(f"    question: {_shorten(s['question'])}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select incorrect, low-uncertainty, homogeneous sampling examples."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to a VSE (or similar) log JSON, preferably *_revised.json.",
    )
    parser.add_argument(
        "--max_uncertainty",
        type=float,
        default=0.1,
        help="Keep samples with uncertainty <= this value (default: 0.0).",
    )
    parser.add_argument(
        "--min_majority",
        type=float,
        default=1.0,
        help="Minimum fraction of samples sharing the majority string (default: 1.0).",
    )
    parser.add_argument(
        "--max_unique_frac",
        type=float,
        default=1.0,
        help="Maximum (#unique answers / #samples). Use with --min_majority.",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=None,
        help="Optional max number of VSE semantic clusters.",
    )
    parser.add_argument(
        "--require_single_cluster",
        action="store_true",
        help="Require answer_cluster_idx / cluster_dis to be a single cluster.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many examples to print.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write selected examples as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log = load_log(args.log)
    selected = select_samples(
        log,
        max_uncertainty=args.max_uncertainty,
        min_majority=args.min_majority,
        max_unique_frac=args.max_unique_frac,
        max_clusters=args.max_clusters,
        require_single_cluster=args.require_single_cluster,
    )

    n_incorrect = sum(
        1
        for k, v in log.items()
        if _is_sample_entry(k, v) and not bool(v.get("flag_answer_correct", True))
    )
    print(f"log: {args.log}")
    print(f"incorrect samples: {n_incorrect}")
    print(
        f"filters: U<={args.max_uncertainty}, majority>={args.min_majority}, "
        f"unique_frac<={args.max_unique_frac}"
        + (", single_cluster" if args.require_single_cluster else "")
        + (f", max_clusters<={args.max_clusters}" if args.max_clusters is not None else "")
    )
    print_examples(selected, top_k=args.top_k)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(selected, f, indent=2)
        print(f"Wrote {len(selected)} samples to {out_path}")


if __name__ == "__main__":
    main()
