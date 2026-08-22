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

from lvlm.generation_utils import build_generation_kwargs

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

    def _decode_answers(self, generated_ids):
        answers = []
        for seq in generated_ids:
            answer = (
                self.processor.decode(seq, skip_special_tokens=True)
                .split("ASSISTANT: ")[-1]
                .strip()
            )
            answers.append(answer)
        return answers

    def prepare_inputs(self, image, question):
        """Build processor inputs for (image, question), matching generate()."""
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
        return (
            self.processor(images=image, text=prompt, return_tensors="pt")
            .to(0, torch.float16)
            .to(self.device)
        )

    def generate(
        self,
        image,
        question,
        temp,
        return_more=False,
        return_mode=0,
        num_beams=1,
        num_return_sequences=None,
        num_beam_groups=1,
        diversity_penalty=0.0,
        length_penalty=1.0,
    ):
        inputs = self.prepare_inputs(image, question)
        use_euq_hooks = return_more and return_mode == 0

        down_proj_features = []
        llm_head_features = []
        down_proj_handle = None
        lm_head_handle = None

        if use_euq_hooks:
            def down_proj_hook(module, inputs_, outputs):
                intermediate = inputs_[0]
                down_proj_features.append(intermediate.detach().cpu())

            def lm_head_hook(module, inputs_, outputs):
                full_hidden = inputs_[0].detach().cpu()
                last_token_hidden = full_hidden[:, -1, :]
                llm_head_features.append(last_token_hidden)

            down_proj_handle = self.model.language_model.model.layers[0].mlp.down_proj.register_forward_hook(down_proj_hook)
            lm_head_handle = self.model.language_model.lm_head.register_forward_hook(lm_head_hook)

        gen_kwargs = build_generation_kwargs(
            temp,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            length_penalty=length_penalty,
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            output_scores=return_more,
            output_attentions=return_more and return_mode == 1,
            output_hidden_states=return_more and return_mode == 1,
            return_dict_in_generate=return_more,
            generation_config=GenerationConfig(**gen_kwargs),
        )
        generated_ids = outputs["sequences"] if return_more else outputs
        answers = self._decode_answers(generated_ids)
        final_answer = answers[0]

        if use_euq_hooks:
            down_proj_features = [x[:, -1:, :].cpu() for x in down_proj_features]
            llm_head_feature_temp = []
            for head_inputs in llm_head_features:
                if head_inputs.dim() == 3:
                    head_inputs = head_inputs.unsqueeze(0)
                llm_head_feature_temp.append(head_inputs.cpu())
            llm_head_features = llm_head_feature_temp
            down_proj_handle.remove()
            lm_head_handle.remove()
            del llm_head_feature_temp

        if return_more and return_mode == 0:
            return final_answer, down_proj_features, llm_head_features
        elif return_more and return_mode == 1:
            return final_answer, inputs, outputs, answers
        return final_answer
