import torch
import latentlens
from transformers import AutoProcessor
from PIL import Image


def check_latentlens(lvlm, llm):
    # Load a VLM and its processor
    model, tokenizer = latentlens.load_model("Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.float16)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # Load pre-built index for this VLM
    index = latentlens.ContextualIndex.from_pretrained(f"McGill-NLP/contextual_embeddings-qwen2.5-vl-7b")

    # Process image + text — apply_chat_template inserts the image placeholder tokens
    image = Image.open("example.jpg")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image."},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text_prompt], return_tensors="pt", padding=True).to("cuda")

    # Extract hidden states — pass all processor outputs
    hidden_states = latentlens.get_hidden_states(model, **inputs)

    # Interpret the last layer's hidden states for all tokens
    results = index.search(hidden_states[27].squeeze(0), top_k=5)


if __name__ == "__main__":
    check_latentlens()
