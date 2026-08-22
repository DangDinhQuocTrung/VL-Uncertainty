"""Shared LVLM introspection helpers (image token ids, LM head, final norm)."""

from __future__ import annotations


def resolve_image_token_id(lvlm=None, model=None, processor=None, name=None):
    """Return the token id used as visual placeholders in input_ids.

    Accepts either an LVLM wrapper (with ``.model`` / ``.processor`` / ``.version``)
    or explicit ``model`` / ``processor`` / ``name`` kwargs, so callers like VAUQ
    and SVAR can share the same lookup.
    """
    if lvlm is not None:
        if name is None:
            name = getattr(lvlm, "version", None) or type(lvlm).__name__
        if processor is None:
            processor = getattr(lvlm, "processor", None)
        if model is None:
            model = getattr(lvlm, "model", None)

    name = str(name or "").lower()

    if processor is not None:
        tok = getattr(processor, "tokenizer", processor)
        if "qwen" in name:
            return int(tok.convert_tokens_to_ids("<|image_pad|>"))
        if "gemma" in name:
            return int(tok.convert_tokens_to_ids("<image_soft_token>"))
        if hasattr(tok, "convert_tokens_to_ids"):
            for special in ("<|image_pad|>", "<image_soft_token>"):
                tid = tok.convert_tokens_to_ids(special)
                if tid is not None and tid != getattr(tok, "unk_token_id", None):
                    return int(tid)

    if model is not None and hasattr(model, "config"):
        if hasattr(model.config, "image_token_index"):
            return int(model.config.image_token_index)
        if hasattr(model.config, "image_token_id"):
            return int(model.config.image_token_id)

    raise ValueError(
        f"Could not resolve image token id for model '{name or type(lvlm)}'."
    )


def get_final_norm(model):
    """Return the final RMS/LayerNorm before the LM head, if present."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "norm")
    ):
        return model.model.language_model.norm
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        lm = model.language_model.model
        if hasattr(lm, "norm"):
            return lm.norm
    return None


def get_lm_head(model):
    """Return the language modeling head module."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    raise AttributeError("Could not find lm_head on model.")
