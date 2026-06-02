import warnings

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)

warnings.filterwarnings("ignore")


class LLaVA:

    def __init__(self, version, use_fastest=False, use_flash_attention=True):
        self.version = version
        self.use_fastest = use_fastest
        self.use_flash_attention = use_flash_attention
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        if self.use_fastest:
            model_name = f"llava-hf/{self.version}"
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
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
            )
        else:
            model_name = f"llava-hf/{self.version}"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, image, question, temp, return_more=False):
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
        inputs = (
            self.processor(images=image, text=prompt, return_tensors="pt")
            .to(0, torch.float16)
            .to(self.device)
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=temp > 0.0,
            temperature=temp,
            output_scores=return_more,
            output_attentions=return_more,
            return_dict_in_generate=return_more,
        )
        answer = outputs["sequences"] if return_more else outputs
        final_answer = (
            self.processor.decode(answer[0], skip_special_tokens=True)
            .split("ASSISTANT: ")[-1]
            .strip()
        )
        if return_more:
            return final_answer, inputs, outputs
        return final_answer
