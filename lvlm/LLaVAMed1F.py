import os
import warnings

import torch
from PIL import Image
from transformers import GenerationConfig

from custom_llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from custom_llava.conversation import conv_templates
from custom_llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from custom_llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")


class LLaVAMed1F:

    def __init__(self, version, use_fastest=False, use_flash_attention=True):
        self.version = version
        self.use_fastest = use_fastest
        self.use_flash_attention = use_flash_attention
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def _model_id(self):
        if "/" in self.version:
            return self.version
        return f"microsoft/{self.version}"

    def build_model(self):
        model_path = self._model_id()
        model_name = get_model_name_from_path(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=self.use_fastest,
            device=device,
            use_flash_attn=self.use_flash_attention,
        )
        self.model.eval()
        self.conv_mode = "mistral_instruct"
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
            attention_weight = self.model.model.layers[0].mlp.down_proj.weight
            print(self.model.model.layers[0].mlp.down_proj)
            attention_weight_cpu = attention_weight.cpu()
            torch.save(attention_weight_cpu, attention_weights_path)
        return

    def _prepare_inputs(self, image, question):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if self.model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_sizes = [image.size]
        images_tensor = process_images(
            [image], self.image_processor, self.model.config
        )
        if isinstance(images_tensor, list):
            images_tensor = [
                img.to(self.model.device, dtype=torch.float16) for img in images_tensor
            ]
        else:
            images_tensor = images_tensor.to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        return {
            "input_ids": input_ids,
            "images": images_tensor,
            "image_sizes": image_sizes,
        }

    def _decode_answer(self, generated_ids):
        answer = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()
        return answer

    def generate(self, image, question, temp, return_more=False, return_mode=0):
        inputs = self._prepare_inputs(image, question)

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
        down_proj_handle = self.model.model.layers[0].mlp.down_proj.register_forward_hook(down_proj_hook)
        lm_head_handle = self.model.lm_head.register_forward_hook(lm_head_hook)

        # Generation
        outputs = self.model.generate(
            inputs["input_ids"],
            images=inputs["images"],
            image_sizes=inputs["image_sizes"],
            max_new_tokens=64,
            output_scores=return_more,
            output_attentions=return_more,
            output_hidden_states=return_more,
            return_dict_in_generate=return_more,
            use_cache=True,
            generation_config=GenerationConfig(
                do_sample=temp > 0.0,
                temperature=temp,
                repetition_penalty=1.05,
                top_k=50,
                top_p=0.95,
            ),
        )
        generated_ids = outputs["sequences"] if return_more else outputs
        answer = self._decode_answer(generated_ids)

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
            return answer, down_proj_features, llm_head_features
        elif return_more and return_mode == 1:
            return answer, inputs, outputs
        return answer
