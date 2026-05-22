import warnings

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

warnings.filterwarnings("ignore")
USE_FASTEST = True


class LLaVA:

    def __init__(self, version):
        self.version = version
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        if USE_FASTEST:
            model_name = "llava-hf/llava-1.5-7b-hf"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            ).to(self.device)
        else:
            model_name = f"llava-hf/{self.version}"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, image, question, temp):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}, {"type": "image"}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            0, torch.float16
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=temp,
        )
        final_ans = (
            self.processor.decode(output[0], skip_special_tokens=True)
            .split("ASSISTANT: ")[-1]
            .strip()
        )
        return final_ans
