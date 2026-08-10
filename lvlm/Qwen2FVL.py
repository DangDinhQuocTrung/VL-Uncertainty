import os
import warnings

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig, GenerationConfig

from custom_llava.conversation import conv_templates, SeparatorStyle
from utils.text_constants import DEFAULT_IMAGE_TOKEN
from lvlm.generation_utils import build_generation_kwargs

warnings.filterwarnings("ignore")


def make_prompt(context, question):
    question = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return prompt


class Qwen2FVL:

    def __init__(self, version, use_fastest=False, use_flash_attention=True):
        self.version = version
        self.use_fastest = use_fastest
        self.use_flash_attention = use_flash_attention
        self.use_flash_attention = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def _model_id(self):
        if "/" in self.version:
            return self.version
        return f"Qwen/{self.version}"

    def build_model(self):
        model_name = self._model_id()
        if self.use_fastest:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                # load_in_8bit=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
                device_map="auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                # Use sdpa for matching the results of previous flash_attention_2
                attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
                device_map="auto",
            )
        self.model.eval()
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

    def _get_temp(self, temp):
        return temp

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
        prompt = question
        # prompt = make_prompt(None, question)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
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

            down_proj_handle = self.model.model.layers[0].mlp.down_proj.register_forward_hook(down_proj_hook)
            lm_head_handle = self.model.lm_head.register_forward_hook(lm_head_hook)

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
        prompt_len = inputs.input_ids.shape[-1]
        generated_ids_trimmed = [out_ids[prompt_len:] for out_ids in generated_ids]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = answers[0]

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
            return answer, down_proj_features, llm_head_features
        elif return_more and return_mode == 1:
            return answer, inputs, outputs, answers
        return answer
