def build_generation_kwargs(
    temp,
    num_beams=1,
    num_return_sequences=None,
    num_beam_groups=1,
    diversity_penalty=0.0,
    length_penalty=1.0,
    repetition_penalty=1.05,
    top_k=50,
    top_p=0.95,
    min_temperature=None,
):
    """Build HuggingFace GenerationConfig kwargs for greedy/sampling/beam search."""
    if num_return_sequences is None:
        num_return_sequences = num_beams if num_beams > 1 else 1

    use_beam_search = num_beams > 1
    do_sample = (not use_beam_search) and temp > 0.0
    temperature = temp
    if min_temperature is not None and do_sample:
        temperature = max(temp, min_temperature)

    gen_kwargs = {
        "do_sample": do_sample,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
    }

    if use_beam_search:
        gen_kwargs["num_beam_groups"] = num_beam_groups
        if num_beam_groups > 1:
            gen_kwargs["diversity_penalty"] = diversity_penalty
    else:
        gen_kwargs["top_k"] = top_k
        gen_kwargs["top_p"] = top_p
        if do_sample:
            gen_kwargs["temperature"] = temperature
        elif min_temperature is not None:
            # Gemma currently always passes a temperature; keep that behavior.
            gen_kwargs["temperature"] = max(temp, min_temperature)

    return gen_kwargs
