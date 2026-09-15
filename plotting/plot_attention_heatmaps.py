"""Plot side-by-side visual attention heatmaps for VQA generation.

For each sample index in ``[sample_start, sample_end]``, runs the LVLM once with
``output_attentions``, then overlays attention mass on the image for:

  - **Generated tokens**: attention from answer-token queries onto visual tokens
  - **Overall**: attention from all query positions (prompt + generated) onto
    visual tokens

Layer range is configurable (default: all decoder layers). Each selected layer
gets a 1x2 figure; an additional mean-over-layers figure is always written.

Requires eager attention (flash attention does not expose weights). The script
loads the LVLM with ``use_flash_attention=False``.

Example:
  PYTHONPATH=. ~/venv/health/bin/python plotting/plot_attention_heatmaps.py \\
      --benchmark ViLP \\
      --lvlm Qwen2.5-VL-7B-Instruct \\
      --sample_start 0 --sample_end 0 \\
      --layer_start 10 --layer_end 16 \\
      --out_dir exp/attn_heatmaps
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

from methods.vauq import _resolve_patch_grid
from methods.vauq_utils import (
    compute_attention_maps_over_visual_tokens,
    num_decoder_layers,
)
from plotting.plot_visual_statistics import render_heatmap_overlay, values_to_grid
from utils.constants import BENCHMARK_MAP, LVLM_MAP
from utils.model_utils import resolve_image_token_id


def parse_layer_spec(
    layer_start: Optional[int],
    layer_end: Optional[int],
    layers: Optional[Sequence[int]],
    n_layers: int,
) -> List[int]:
    """Resolve layer indices. Default is all layers ``[0, n_layers)``."""
    if layers is not None and len(layers) > 0:
        idxs = [int(x) for x in layers]
    else:
        start = 0 if layer_start is None else int(layer_start)
        end = n_layers if layer_end is None else int(layer_end)
        if end <= start:
            raise ValueError(f"Empty layer range: start={start}, end={end}.")
        idxs = list(range(start, end))

    for li in idxs:
        if li < 0 or li >= n_layers:
            raise ValueError(f"Layer {li} out of range [0, {n_layers}).")
    return idxs


def normalize_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


def resolve_visual_token_row_height_px(lvlm, image: Image.Image, question: str) -> int:
    """Pixel height of one visual-token row on the current image.

    Dry-runs ``prepare_inputs`` to resolve the token grid, then returns
    ``round(image_height / grid_h)``.
    """
    img = normalize_image(image)
    if not hasattr(lvlm, "prepare_inputs"):
        raise AttributeError(
            f"{type(lvlm).__name__} has no prepare_inputs(); cannot resolve patch row height."
        )
    inputs = lvlm.prepare_inputs(img, question)
    image_token_id = resolve_image_token_id(lvlm=lvlm)
    n_visual = int((inputs["input_ids"][0] == image_token_id).sum().item())
    grid_h, _grid_w = _resolve_patch_grid(lvlm, inputs, n_visual, img)
    if grid_h <= 0:
        raise ValueError(f"Invalid visual token grid_h={grid_h}.")
    return max(1, int(round(img.height / grid_h)))


def prepend_black_patch_rows(
    image: Image.Image,
    artificial_rows: int,
    row_height_px: int,
) -> Image.Image:
    """Prepend ``artificial_rows`` black visual-token rows (each ``row_height_px`` tall).

    If ``artificial_rows`` is 0, return the original image unchanged.
    """
    n = int(artificial_rows)
    if n < 0:
        raise ValueError(f"artificial_rows must be >= 0, got {n}.")
    img = normalize_image(image)
    if n == 0:
        return img
    if row_height_px <= 0:
        raise ValueError(f"row_height_px must be > 0, got {row_height_px}.")
    pad_px = n * int(row_height_px)
    width, height = img.size
    canvas = Image.new("RGB", (width, height + pad_px), (0, 0, 0))
    canvas.paste(img, (0, pad_px))
    return canvas


def clip_to_percentile(grid: np.ndarray, percentile: float) -> np.ndarray:
    """Clip values above the given percentile (e.g. 99) before min-max coloring."""
    if percentile is None or percentile >= 100.0:
        return grid
    if percentile <= 0.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}.")
    hi = float(np.percentile(grid, percentile))
    return np.minimum(grid, hi)


def topk_visual_positions(
    values: Sequence[float],
    grid_h: int,
    grid_w: int,
    top_k: int,
) -> List[Tuple[int, int, int, float]]:
    """Return top-k visual tokens as (rank, row, col, value), 0-indexed row/col."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    expected = grid_h * grid_w
    if arr.size != expected:
        raise ValueError(
            f"Expected {expected} values for grid {grid_h}x{grid_w}, got {arr.size}."
        )
    k = min(int(top_k), arr.size)
    if k <= 0:
        return []
    # Stable ranking among ties: higher value first, then lower flat index.
    order = np.argsort(-arr, kind="stable")[:k]
    out: List[Tuple[int, int, int, float]] = []
    for rank, flat in enumerate(order, start=1):
        row, col = divmod(int(flat), grid_w)
        out.append((rank, row, col, float(arr[flat])))
    return out


def _annotate_topk(
    ax,
    image: Image.Image,
    topk: Sequence[Tuple[int, int, int, float]],
    grid_h: int,
    grid_w: int,
) -> None:
    """Mark top-k patch centers and list (row, col) under the panel."""
    img_w, img_h = image.size
    patch_h = img_h / grid_h
    patch_w = img_w / grid_w
    for rank, row, col, _value in topk:
        x = (col + 0.5) * patch_w
        y = (row + 0.5) * patch_h
        ax.scatter(
            [x],
            [y],
            s=36,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            x,
            y,
            str(rank),
            color="black",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
    if topk:
        lines = [f"{rank}: (r={row}, c={col})" for rank, row, col, _ in topk]
        ax.text(
            0.0,
            -0.02,
            f"Top-{len(topk)}: " + "  |  ".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#222222",
            wrap=False,
        )


def save_side_by_side(
    image: Image.Image,
    generated_values: Sequence[float],
    overall_values: Sequence[float],
    grid_h: int,
    grid_w: int,
    out_path: Path,
    title: str,
    question: str,
    answer: str,
    cmap: str,
    alpha: float,
    clip_percentile: float = 99.0,
    top_k: int = 5,
) -> None:
    # print(torch.sort(torch.tensor(generated_values), dim=0, descending=True)[0][:20])
    # print(torch.sort(torch.tensor(overall_values), dim=0, descending=True)[0][:20])
    # Rank on raw attention; clip only for colormap display.
    gen_topk = topk_visual_positions(generated_values, grid_h, grid_w, top_k)
    ovr_topk = topk_visual_positions(overall_values, grid_h, grid_w, top_k)
    gen_grid = clip_to_percentile(values_to_grid(generated_values, grid_h, grid_w), clip_percentile)
    ovr_grid = clip_to_percentile(values_to_grid(overall_values, grid_h, grid_w), clip_percentile)
    gen_overlay = render_heatmap_overlay(image, gen_grid, cmap_name=cmap, alpha=alpha)
    ovr_overlay = render_heatmap_overlay(image, ovr_grid, cmap_name=cmap, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2))
    panels = (
        (axes[0], gen_overlay, gen_grid, gen_topk, "Generated tokens → visual"),
        (axes[1], ovr_overlay, ovr_grid, ovr_topk, "Overall → visual"),
    )
    for ax, overlay, grid, topk, panel_title in panels:
        ax.imshow(overlay)
        ax.set_title(panel_title)
        ax.axis("off")
        _annotate_topk(ax, image, topk, grid_h, grid_w)
        vmin = float(grid.min())
        vmax = float(grid.max())
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    q = textwrap.fill(question.replace("\n", " "), width=100)
    a = textwrap.fill(str(answer).replace("\n", " "), width=100)
    fig.suptitle(f"{title}\nQ: {q}\nA: {a}", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def obtain_lvlm(lvlm_name: str, use_fastest: bool):
    lvlm_class = LVLM_MAP.get(lvlm_name)
    if not lvlm_class:
        raise ValueError(f"Unsupported LVLM: {lvlm_name}")
    # Attentions require eager implementation.
    return lvlm_class(lvlm_name, use_fastest=use_fastest, use_flash_attention=False)


def obtain_benchmark(benchmark_name: str):
    benchmark_class = BENCHMARK_MAP.get(benchmark_name)
    if not benchmark_class:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")
    return benchmark_class()


def process_sample(
    lvlm,
    sample,
    layer_indices: Sequence[int],
    out_dir: Path,
    cmap: str,
    alpha: float,
    inference_temp: float,
    save_per_layer: bool,
    clip_percentile: float = 99.0,
    top_k: int = 5,
    artificial_rows: int = 0,
) -> str:
    image = normalize_image(sample["img"])
    row_height_px = 0
    if artificial_rows:
        row_height_px = resolve_visual_token_row_height_px(
            lvlm, image, sample["question"]
        )
        image = prepend_black_patch_rows(image, artificial_rows, row_height_px)
        print(
            f"Sample {sample['idx']}: prepended {artificial_rows} patch row(s) "
            f"({row_height_px}px each, {artificial_rows * row_height_px}px total)"
        )
    answer, inputs, outputs, _ = lvlm.generate(
        image,
        sample["question"],
        inference_temp,
        return_more=True,
        return_mode=2,
    )

    maps = compute_attention_maps_over_visual_tokens(
        lvlm.model,
        lvlm.processor,
        inputs,
        outputs,
        getattr(lvlm, "version", type(lvlm).__name__),
        layer_indices=layer_indices,
        device=lvlm.device,
    )
    n_visual = int(maps["visual_token_positions"].numel())
    grid_h, grid_w = _resolve_patch_grid(lvlm, inputs, n_visual, image)

    generated = maps["generated"].detach().float().cpu()
    overall = maps["overall"].detach().float().cpu()

    idx = sample["idx"]
    mean_gen = generated.mean(dim=0).tolist()
    mean_ovr = overall.mean(dim=0).tolist()
    pad_tag = f"_arows{artificial_rows}" if artificial_rows else ""
    title_pad = (
        f" | artificial_rows={artificial_rows}" if artificial_rows else ""
    )
    save_side_by_side(
        image,
        mean_gen,
        mean_ovr,
        grid_h,
        grid_w,
        out_dir / f"sample_{idx:04d}_layers_mean{pad_tag}.png",
        title=f"Sample {idx} | mean over layers {list(layer_indices)}{title_pad}",
        question=sample["question"],
        answer=answer,
        cmap=cmap,
        alpha=alpha,
        clip_percentile=clip_percentile,
        top_k=top_k,
    )

    if save_per_layer:
        for row, layer_idx in enumerate(maps["layer_indices"]):
            save_side_by_side(
                image,
                generated[row].tolist(),
                overall[row].tolist(),
                grid_h,
                grid_w,
                out_dir / f"sample_{idx:04d}_layer_{layer_idx:02d}{pad_tag}.png",
                title=f"Sample {idx} | layer {layer_idx}{title_pad}",
                question=sample["question"],
                answer=answer,
                cmap=cmap,
                alpha=alpha,
                clip_percentile=clip_percentile,
                top_k=top_k,
            )

    # Free large attention tensors before the next sample.
    del outputs, maps, generated, overall
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side attention heatmaps (generated vs overall)."
    )
    parser.add_argument("--benchmark", type=str, default="ViLP")
    parser.add_argument("--lvlm", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument(
        "--sample_end",
        type=int,
        default=8,
        help="Inclusive end sample index (one image/sample per index).",
    )
    parser.add_argument(
        "--layer_start",
        type=int,
        default=None,
        help="Inclusive start layer (default: 0).",
    )
    parser.add_argument(
        "--layer_end",
        type=int,
        default=None,
        help="Exclusive end layer (default: all layers).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Explicit layer indices (overrides --layer_start/--layer_end).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="exp",
    )
    parser.add_argument("--inference_temp", type=float, default=0.0)
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Clip attention values to this percentile before min-max coloring (default: 99).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Mark the top-k visual tokens (by raw attention) with row/col on each heatmap.",
    )
    parser.add_argument(
        "--artificial_rows",
        type=int,
        default=1,
        help=(
            "Prepend this many black visual-token rows (patch-row height, not "
            "pixels) to the top of the input image before generation "
            "(0 = original image)."
        ),
    )
    parser.add_argument(
        "--per_layer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save one 1x2 figure per selected layer (default: False). Use --per-layer to enable.",
    )
    parser.add_argument("--use_fastest", action="store_true")
    args = parser.parse_args()

    if args.sample_end < args.sample_start:
        raise ValueError(
            f"sample_end ({args.sample_end}) < sample_start ({args.sample_start})"
        )

    if args.artificial_rows < 0:
        raise ValueError(
            f"artificial_rows must be >= 0, got {args.artificial_rows}."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = obtain_benchmark(args.benchmark)
    lvlm = obtain_lvlm(args.lvlm, use_fastest=args.use_fastest)
    n_layers = num_decoder_layers(lvlm.model)
    layer_indices = parse_layer_spec(
        args.layer_start, args.layer_end, args.layers, n_layers
    )
    print(
        f"Model layers={n_layers}, using {len(layer_indices)} layers: {layer_indices}"
    )
    print(
        f"Samples [{args.sample_start}, {args.sample_end}] "
        f"artificial_rows={args.artificial_rows} -> {out_dir}"
    )

    for idx in range(args.sample_start, args.sample_end + 1):
        sample = benchmark.retrieve(idx)
        if sample is None or sample.get("img") is None or sample.get("question") is None:
            print(f"Skip invalid sample {idx}")
            continue
        print(f"Sample {idx}: generating with attentions...")
        answer = process_sample(
            lvlm,
            sample,
            layer_indices=layer_indices,
            out_dir=out_dir,
            cmap=args.cmap,
            alpha=args.alpha,
            inference_temp=args.inference_temp,
            save_per_layer=args.per_layer,
            clip_percentile=args.clip_percentile,
            top_k=args.top_k,
            artificial_rows=args.artificial_rows,
        )
        print(f"Sample {idx}: answer={answer!r}")
    return


if __name__ == "__main__":
    main()
