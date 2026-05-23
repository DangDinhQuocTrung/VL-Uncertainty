import os
import warnings

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

warnings.filterwarnings("ignore")
USE_FASTEST = True


class Qwen2FVL:

    def __init__(self, version):
        self.version = version
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        if USE_FASTEST:
            model_name = f"Qwen/{self.version}"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            model_name = f"Qwen/{self.version}"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.save_head_weights()

    def save_head_weights(self):
        self.weight_dir = "/work3/dida/outputs_LVLM/EUQ_weights"
        os.makedirs(self.weight_dir, exist_ok=True)
        head_weights_path = f"{self.weight_dir}/{self.version}_head_weights.pth"
        attention_weights_path = f"{self.weight_dir}/{self.version}_attention_weights.pth"
        if (not os.path.exists(head_weights_path)) or (not os.path.exists(attention_weights_path)):
            head_weight = self.model.lm_head.weight
            print(self.model.lm_head)
            head_weight_cpu = head_weight.cpu()
            torch.save(head_weight_cpu, head_weights_path)
            # attention_weight = self.model.model.language_model.layers[0].mlp.down_proj.weight
            attention_weight = self.model.model.layers[0].mlp.down_proj.weight
            print(self.model.model.layers[0].mlp.down_proj)
            # attention_weight = attention_weight.view(1792, 18944)
            attention_weight_cpu = attention_weight.cpu()
            torch.save(attention_weight_cpu, attention_weights_path)
        return

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

        # Hooking
        down_proj_features = []
        llm_head_features = []

        def down_proj_hook(module, inputs, outputs):
            intermediate = inputs[0]
            down_proj_features.append(intermediate.detach().cpu())

        def lm_head_hook(module, inputs, outputs):
            full_hidden = inputs[0].detach().cpu()
            last_token_hidden = full_hidden[:, -1, :]
            llm_head_features.append(last_token_hidden)

        down_proj_handle = self.model.model.layers[0].mlp.down_proj.register_forward_hook(down_proj_hook)
        lm_head_handle = self.model.lm_head.register_forward_hook(lm_head_hook)

        # Generation
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=32,
            output_hidden_states=True,
            do_sample=True,
            temperature=temp,
            repetition_penalty=1.05,
            top_k=50,
            top_p=0.95,
        )
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

        # Post-processing
        down_proj_features = [x[:,-1:,:].cpu() for x in down_proj_features]
        llm_head_feature_temp = []
        for inputs in llm_head_features:
            if(inputs.dim() == 3):
                inputs = inputs.unsqueeze(0)
            llm_head_feature_temp.append(inputs.cpu())
        llm_head_features = llm_head_feature_temp

        down_proj_handle.remove()
        lm_head_handle.remove()
        del llm_head_feature_temp

        if return_more:
            return answer, down_proj_features, llm_head_features
        return answer
