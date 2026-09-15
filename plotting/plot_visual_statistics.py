"""Plot per-token visual statistics as patch heatmaps on ViLP images.

Reads experiment logs produced with --compute_visual_statistics true and
log_per_image arrays (see exp_03/values_plot_code for examples).

For EUQ: overlays head conflict and ignorance per visual token.
For semantic entropy: overlays per-token LogitLens entropy (H_vis).

Example:
  python plotting/plot_visual_statistics.py \\
      --log exp_03/values_plot_code/log_2026_08_26_02_13_07_revised.json \\
      --num_samples 3 \\
      --out exp/fig_visual_statistics_euq.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

from benchmark.ViLP import ViLP
from utils.constants import LOG_META_KEYS


EUQ_STAT_SPECS = (
    ("visual_token_head_conflict_values", "Head Conflict"),
    ("visual_token_head_ignorance_values", "Head Ignorance"),
)
SE_STAT_SPECS = (("visual_token_entropies", r"Token Entropy ($H_{vis}$)"),)


def load_log(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r") as f:
        return json.load(f)


def detect_method(log: Dict[str, Any]) -> str:
    method = str(log.get("uncertainty_method", "")).lower()
    if "semantic_entropy" in method:
        return "semantic_entropy"
    if method == "euq":
        return "euq"
    raise ValueError(
        f"Unsupported uncertainty_method '{log.get('uncertainty_method')}'. "
        "Expected semantic_entropy or euq."
    )


def _is_sample_entry(key: str, value: Any) -> bool:
    if key in LOG_META_KEYS or not isinstance(value, dict):
        return False
    if "Hallucination detection" in key or "Perturbation detection" in key:
        return False
    return True


def stat_specs_for_method(method: str) -> Tuple[Tuple[str, str], ...]:
    if method == "euq":
        return EUQ_STAT_SPECS
    if method == "semantic_entropy":
        return SE_STAT_SPECS
    raise ValueError(f"Unsupported method: {method}")


def iter_sample_indices(log: Dict[str, Any], num_samples: int, start_idx: int) -> List[int]:
    indices = sorted(int(k) for k in log if str(k).isdigit())
    if start_idx > 0:
        indices = [idx for idx in indices if idx >= start_idx]
    if not indices:
        raise ValueError("No numeric sample indices found in log.")
    return indices[:num_samples]


def infer_grid_shape(
    n_tokens: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """Return (grid_height, grid_width) with grid_h * grid_w == n_tokens."""
    if n_tokens <= 0:
        raise ValueError(f"n_visual_tokens must be positive, got {n_tokens}.")

    best: Optional[Tuple[float, int, int]] = None
    target_ratio = image_height / max(image_width, 1)
    max_side = int(math.isqrt(n_tokens))
    for grid_h in range(1, max_side + 1):
        if n_tokens % grid_h != 0:
            continue
        grid_w = n_tokens // grid_h
        ratio = grid_h / grid_w
        score = abs(math.log(ratio + 1e-9) - math.log(target_ratio + 1e-9))
        if best is None or score < best[0]:
            best = (score, grid_h, grid_w)
    if best is None:
        raise ValueError(f"Could not factor n_visual_tokens={n_tokens} into a 2D grid.")
    return best[1], best[2]


def _get_processor_grid_shape(
    lvlm_version: str,
    image: Image.Image,
    question: str,
    n_tokens: int,
) -> Optional[Tuple[int, int]]:
    """Use the LVLM processor grid when transformers/qwen_vl_utils are available."""
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
    except ImportError:
        return None

    model_id = lvlm_version if "/" in lvlm_version else f"Qwen/{lvlm_version}"
    processor = AutoProcessor.from_pretrained(model_id)
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        return None

    t, grid_h, grid_w = [int(x) for x in grid_thw[0].tolist()]
    merge_unit = merge_size * merge_size
    computed_tokens = (t * grid_h * grid_w) // merge_unit
    if computed_tokens != n_tokens:
        return None
    return grid_h // merge_size, grid_w // merge_size


def resolve_grid_shape(
    sample_entry: Dict[str, Any],
    image: Image.Image,
    question: str,
    lvlm_version: Optional[str],
    use_processor_grid: bool,
) -> Tuple[int, int]:
    n_tokens = int(sample_entry["n_visual_tokens"])
    img_w, img_h = image.size

    if use_processor_grid and lvlm_version:
        proc_shape = _get_processor_grid_shape(lvlm_version, image, question, n_tokens)
        if proc_shape is not None:
            return proc_shape

    return infer_grid_shape(n_tokens, img_w, img_h)


def values_to_grid(values: Sequence[float], grid_h: int, grid_w: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    expected = grid_h * grid_w
    if arr.size != expected:
        raise ValueError(
            f"Expected {expected} values for grid {grid_h}x{grid_w}, got {arr.size}."
        )
    return arr.reshape(grid_h, grid_w)


def upsample_grid(grid: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    grid_img = Image.fromarray(grid.astype(np.float32), mode="F")
    upsampled = grid_img.resize((out_width, out_height), Image.BILINEAR)
    return np.asarray(upsampled, dtype=np.float64)


def render_heatmap_overlay(
    image: Image.Image,
    grid: np.ndarray,
    cmap_name: str = "viridis",
    alpha: float = 0.55,
) -> np.ndarray:
    img_w, img_h = image.size
    upsampled = upsample_grid(grid, img_w, img_h)
    cmap = plt.colormaps[cmap_name]
    vmin = float(grid.min())
    vmax = float(grid.max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    norm = (upsampled - vmin) / (vmax - vmin)
    heatmap_rgba = cmap(np.clip(norm, 0.0, 1.0))
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = heatmap_rgb.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * overlay
    return np.clip(blended, 0, 255).astype(np.uint8)


def _lvlm_version_from_log(log: Dict[str, Any]) -> Optional[str]:
    args = log.get("args")
    if args is None:
        return None
    if hasattr(args, "lvlm"):
        return str(args.lvlm)
    if isinstance(args, dict):
        return args.get("lvlm")
    return None


def plot_sample_row(
    axes,
    image: Image.Image,
    sample_entry: Dict[str, Any],
    stat_specs: Sequence[Tuple[str, str]],
    grid_shape: Tuple[int, int],
    sample_idx: int,
    cmap: str,
    alpha: float,
) -> None:
    grid_h, grid_w = grid_shape
    axes[0].imshow(image)
    axes[0].set_title(f"Sample {sample_idx}")
    axes[0].axis("off")

    uncertainty = sample_entry.get("uncertainty")
    subtitle_parts = []
    if uncertainty is not None:
        subtitle_parts.append(f"uncertainty={float(uncertainty):.3f}")
    if "visual_entropy" in sample_entry:
        subtitle_parts.append(f"H_vis={float(sample_entry['visual_entropy']):.3f}")
    if subtitle_parts:
        axes[0].text(
            0.5,
            -0.08,
            " | ".join(subtitle_parts),
            transform=axes[0].transAxes,
            ha="center",
            fontsize=8,
            color="#444444",
        )

    for ax, (value_key, title) in zip(axes[1:], stat_specs):
        if value_key not in sample_entry:
            raise KeyError(
                f"Sample {sample_idx} is missing '{value_key}'. "
                "Re-run with --compute_visual_statistics true and per-image logging."
            )
        grid = values_to_grid(sample_entry[value_key], grid_h, grid_w)
        overlay = render_heatmap_overlay(image, grid, cmap_name=cmap, alpha=alpha)
        ax.imshow(overlay)
        ax.set_title(title)
        ax.axis("off")

        vmin = float(grid.min())
        vmax = float(grid.max())
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)


def plot_visual_statistics(
    log: Dict[str, Any],
    vilp: ViLP,
    sample_indices: Sequence[int],
    out_path: str,
    cmap: str = "viridis",
    alpha: float = 0.55,
    use_processor_grid: bool = False,
) -> None:
    method = detect_method(log)
    stat_specs = stat_specs_for_method(method)
    lvlm_version = _lvlm_version_from_log(log)
    n_cols = 1 + len(stat_specs)
    n_rows = len(sample_indices)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    for row, sample_idx in enumerate(sample_indices):
        key = str(sample_idx)
        if key not in log or not _is_sample_entry(key, log[key]):
            raise KeyError(f"Sample {sample_idx} not found in log.")
        sample_entry = log[key]
        if "n_visual_tokens" not in sample_entry:
            raise KeyError(
                f"Sample {sample_idx} is missing 'n_visual_tokens'. "
                "Re-run with --compute_visual_statistics true."
            )

        retrieved = vilp.retrieve(sample_idx)
        image = retrieved["img"]
        question = retrieved.get("question") or sample_entry.get("question", "")
        grid_shape = resolve_grid_shape(
            sample_entry,
            image,
            question,
            lvlm_version=lvlm_version,
            use_processor_grid=use_processor_grid,
        )
        plot_sample_row(
            axes[row],
            image,
            sample_entry,
            stat_specs,
            grid_shape,
            sample_idx,
            cmap=cmap,
            alpha=alpha,
        )

    method_label = "EUQ" if method == "euq" else "Semantic Entropy"
    dataset = log.get("dataset_name", "ViLP")
    fig.suptitle(
        f"Visual Token Statistics ({method_label}, {dataset})",
        fontsize=14,
        y=1.01 if n_rows == 1 else 1.0,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-token visual statistics as patch heatmaps (ViLP only)."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to exp/log_*.json with per-token visual statistics.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples from the log to visualize (default: 5).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Only consider log samples with idx >= start_idx (default: 0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output figure path (default: exp/fig_visual_statistics_<method>.png).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for heatmaps (default: viridis).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Heatmap overlay alpha on the original image (default: 0.55).",
    )
    parser.add_argument(
        "--use_processor_grid",
        action="store_true",
        help=(
            "Infer patch grid from the LVLM processor (Qwen2.5-VL). "
            "Falls back to aspect-ratio factorization if unavailable."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log = load_log(args.log)

    dataset = log.get("dataset_name")
    if dataset is not None and dataset != "ViLP":
        raise ValueError(
            f"This script supports ViLP only (log dataset_name={dataset!r})."
        )

    method = detect_method(log)
    sample_indices = iter_sample_indices(log, args.num_samples, args.start_idx)
    if not sample_indices:
        raise ValueError("No samples selected for plotting.")

    out_path = args.out
    if out_path is None:
        out_path = f"exp/fig_visual_statistics_{method}.png"

    print(f"Dataset: ViLP | method: {method} | samples: {sample_indices}")
    vilp = ViLP()
    plot_visual_statistics(
        log,
        vilp,
        sample_indices,
        out_path=out_path,
        cmap=args.cmap,
        alpha=args.alpha,
        use_processor_grid=args.use_processor_grid,
    )


if __name__ == "__main__":
    main()
