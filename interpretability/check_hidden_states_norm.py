"""Check whether HF hidden_states[-1] is already final-normed (LogitLens caveat).

Compares:
  - hidden_states[-1]  (what visual_statistics.py currently uses)
  - final_norm(hidden_states[-1])
  - outputs.last_hidden_state / outputs[0] when available
  - lm_head logits from raw vs re-normalized hidden states

Example:
  python interpretability/check_hidden_states_norm.py \\
      --lvlm Qwen2.5-VL-7B-Instruct \\
      --sample_idx 0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from benchmark.ViLP import ViLP
from lvlm.Qwen2FVL import Qwen2FVL
from utils.model_utils import get_final_norm, get_lm_head, resolve_image_token_id


def _tensor_report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a_f = a.detach().float()
    b_f = b.detach().float()
    diff = (a_f - b_f).abs()
    cos = F.cosine_similarity(a_f.flatten(), b_f.flatten(), dim=0).item()
    equal = torch.allclose(a_f, b_f, rtol=1e-3, atol=1e-3)
    print(
        f"{name}: equal={equal}  "
        f"max_abs_diff={diff.max().item():.6g}  "
        f"mean_abs_diff={diff.mean().item():.6g}  "
        f"cosine={cos:.8f}"
    )


def _rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item())


def main():
    parser = argparse.ArgumentParser(
        description="Check if LM hidden_states[-1] already includes final norm."
    )
    parser.add_argument("--lvlm", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Override question text (default: ViLP sample question).",
    )
    args = parser.parse_args()

    vilp = ViLP()
    sample = vilp.retrieve(args.sample_idx)
    image = sample["img"]
    question = args.question or sample["question"]

    print(f"Loading {args.lvlm} ...")
    lvlm = Qwen2FVL(args.lvlm)
    model = lvlm.model
    model.eval()

    final_norm = get_final_norm(model)
    lm_head = get_lm_head(model)
    if final_norm is None:
        raise RuntimeError("Could not locate final norm on this model.")

    print(f"final_norm: {final_norm.__class__.__name__} @ {final_norm}")
    print(f"lm_head: {lm_head}")

    inputs = lvlm.prepare_inputs(image, question)
    image_token_id = resolve_image_token_id(lvlm)
    visual_pos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    print(
        f"seq_len={inputs['input_ids'].shape[-1]}  "
        f"n_visual_tokens={visual_pos.numel()}  "
        f"image_token_id={image_token_id}"
    )

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        hs = outputs.hidden_states
        raw_last = hs[-1]
        renormed = final_norm(raw_last)

        print(f"\n# hidden_states layers: {len(hs)}")
        print(f"hidden_states[-1] shape: {tuple(raw_last.shape)}")
        print(f"RMS(raw_last)     = {_rms(raw_last):.6g}")
        print(f"RMS(norm(raw))    = {_rms(renormed):.6g}")

        print("\n=== hidden_states[-1] vs final_norm(hidden_states[-1]) ===")
        _tensor_report("all tokens", raw_last, renormed)
        if visual_pos.numel() > 0:
            _tensor_report(
                "visual tokens",
                raw_last[0, visual_pos],
                renormed[0, visual_pos],
            )
            _tensor_report(
                "last text token",
                raw_last[0, -1],
                renormed[0, -1],
            )

        # last_hidden_state / logits when exposed by this HF class
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            print("\n=== vs outputs.last_hidden_state ===")
            _tensor_report("raw_last vs last_hidden_state", raw_last, outputs.last_hidden_state)
            _tensor_report(
                "renormed vs last_hidden_state", renormed, outputs.last_hidden_state
            )

        if hasattr(outputs, "logits") and outputs.logits is not None:
            print("\n=== lm_head logits (full vocab, all tokens) ===")
            logits_raw = lm_head(raw_last)
            logits_norm = lm_head(renormed)
            _tensor_report("logits(raw) vs logits(norm(raw))", logits_raw, logits_norm)
            _tensor_report("logits(raw) vs outputs.logits", logits_raw, outputs.logits)
            _tensor_report("logits(norm) vs outputs.logits", logits_norm, outputs.logits)

            # Ranking agreement on a few visual tokens (LogitLens relevance)
            if visual_pos.numel() > 0:
                idx = visual_pos[: min(8, visual_pos.numel())]
                topk = 5
                agree = 0
                for p in idx.tolist():
                    t_raw = logits_raw[0, p].topk(topk).indices
                    t_norm = logits_norm[0, p].topk(topk).indices
                    agree += int(torch.equal(t_raw, t_norm))
                print(
                    f"visual-token top-{topk} rank agree "
                    f"(raw vs renorm): {agree}/{len(idx)}"
                )

        already_normalized = torch.allclose(
            raw_last.float(), renormed.float(), rtol=1e-3, atol=1e-3
        )
        print("\n=== verdict ===")
        if already_normalized:
            print(
                "hidden_states[-1] ALREADY matches final_norm(...).\n"
                "Applying norm again would be wrong; current visual_statistics.py is OK."
            )
        else:
            print(
                "hidden_states[-1] is NOT final-normalized.\n"
                "For LogitLens / H_vis you should apply final_norm before lm_head.\n"
                "Current visual_statistics.py skips that and may be incorrect."
            )


if __name__ == "__main__":
    main()
