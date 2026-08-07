import os
import warnings

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    GenerationConfig,
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
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.save_head_weights()

    def save_head_weights(self):
        self.weight_dir = "/work3/dida/outputs_LVLM/EUQ_weights"
        os.makedirs(self.weight_dir, exist_ok=True)
        head_weights_path = f"{self.weight_dir}/{self.version}_head_weights.pth"
        attention_weights_path = f"{self.weight_dir}/{self.version}_attention_weights.pth"
        if (not os.path.exists(head_weights_path)) or (not os.path.exists(attention_weights_path)):
            head_weight = self.model.language_model.lm_head.weight
            print(self.model.language_model.lm_head)
            head_weight_cpu = head_weight.cpu()
            torch.save(head_weight_cpu, head_weights_path)
            attention_weight = self.model.language_model.model.layers[0].mlp.down_proj.weight
            print(self.model.language_model.model.layers[0].mlp.down_proj)
            attention_weight_cpu = attention_weight.cpu()
            torch.save(attention_weight_cpu, attention_weights_path)
        return

    def generate(self, image, question, temp, return_more=False, return_mode=0):
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

        # EUQ hook
        down_proj_handle = self.model.language_model.model.layers[0].mlp.down_proj.register_forward_hook(down_proj_hook)
        lm_head_handle = self.model.language_model.lm_head.register_forward_hook(lm_head_hook)

        # Generation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            output_scores=return_more,
            output_attentions=return_more,
            output_hidden_states=return_more,
            return_dict_in_generate=return_more,
            generation_config=GenerationConfig(
                do_sample=temp > 0.0,
                temperature=temp,
                repetition_penalty=1.05,
                top_k=50,
                top_p=0.95,
            ),
        )
        generated_ids = outputs["sequences"] if return_more else outputs
        final_answer = (
            self.processor.decode(generated_ids[0], skip_special_tokens=True)
            .split("ASSISTANT: ")[-1]
            .strip()
        )

        # Post-processing
        down_proj_features = [x[:, -1:, :].cpu() for x in down_proj_features]
        llm_head_feature_temp = []
        for head_inputs in llm_head_features:
            if(head_inputs.dim() == 3):
                head_inputs = head_inputs.unsqueeze(0)
            llm_head_feature_temp.append(head_inputs.cpu())
        llm_head_features = llm_head_feature_temp

        # Remove temporary variables
        down_proj_handle.remove()
        lm_head_handle.remove()
        del llm_head_feature_temp

        if return_more and return_mode == 0:
            return final_answer, down_proj_features, llm_head_features
        elif return_more and return_mode == 1:
            return final_answer, inputs, outputs
        return final_answer
