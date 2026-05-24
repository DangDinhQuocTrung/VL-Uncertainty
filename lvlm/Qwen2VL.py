import warnings

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

warnings.filterwarnings("ignore")


class Qwen2VL:

    def __init__(self, version, use_fastest=False):
        self.version = version
        self.use_fastest = use_fastest
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        model_name = f"Qwen/{self.version}"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, image, question, temp, return_more=False):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=temp,
            repetition_penalty=1.05,
            top_k=50,
            top_p=0.95,
            output_attentions=return_more,
            output_scores=return_more,
            return_dict_in_generate=return_more,
        )
        generated_ids = outputs["sequences"] if return_more else outputs
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = answer[0]

        if return_more:
            return answer, inputs, outputs
        return answer
