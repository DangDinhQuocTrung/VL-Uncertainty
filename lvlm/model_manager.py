import os
import warnings

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)

from custom_llava.mm_utils import get_model_name_from_path, process_images
from custom_llava.model.builder import load_pretrained_model
from methods.svar.utils import set_act_get_hooks, remove_hooks

warnings.filterwarnings("ignore")

# LLaVA-1.5
IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_LENGTH = 576
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"

SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere>\n<question> ASSISTANT:",
    "llava-1.5-7b-hf": "USER: <ImageHere>\n<question> ASSISTANT:",
    "internvl": "USER: <ImageHere> <question> ASSISTANT:",
    "idefics2": "User:<ImageHere><question><end_of_utterance>\nAssistant:",
}


def load_llava_model(model_path, use_fastest=False):
    # load the model
    load_8bit = False
    load_4bit = use_fastest
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        load_8bit, load_4bit, device=device,
    )
    return tokenizer, model, image_processor, model


def prepare_llava_inputs(template, query, image_tensor, tokenizer):
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1)
    kwargs = {}

    kwargs["images"] = image_tensor.half()
    return qu, input_ids, img_start_idx, img_end_idx, kwargs


class LLaVAModelManager:

    def __init__(self, model_name, use_fastest=False):
        self.model_name = model_name.lower()
        self.use_fastest = use_fastest
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.load_model()
        self.beam = 1
        self.max_tokens = 512

    def load_model(self):
        if self.model_name == "llava-1.5-7b-hf":
            # model_path = "/work3/dida/cache/hub/models--liuhaotian--llava-v1.5-7b"
            model_path = "liuhaotian/llava-v1.5-7b"
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path, self.use_fastest)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return

    def construct_template(self):
        if self.model_name == "llava-1.5-7b-hf":
            template = SYSTEM_MESSAGE + " " + INSTRUCTION_TEMPLATE[self.model_name]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return template

    def prepare_inputs_for_model(self, query, image, use_dataloader=False):
        template = self.construct_template()

        if self.model_name == "llava-1.5-7b-hf":
            if use_dataloader:
                images_tensor = image["pixel_values"][0]
            else:
                images_tensor = image
            questions, input_ids, img_start_idx, img_end_idx, kwargs = prepare_llava_inputs(
                template, query, images_tensor, self.tokenizer
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx
        return questions, input_ids, kwargs

    def decode(self, output_ids):
        # get outputs
        if self.model_name == "llava-1.5-7b-hf":
            # replace image token by pad token
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return output_text

    def generate(self, image, question, temp, return_more=False):
        images_tensor = process_images(
                [image],
                self.image_processor,
                self.llm_model.config
        ).to(self.llm_model.device, dtype=torch.float16)
        query = [question]
        questions, input_ids, kwargs = self.prepare_inputs_for_model(query, images_tensor, use_dataloader=False)

        # Use hooks to get the attention sublayers' output
        hooks = set_act_get_hooks(self.llm_model.model, attn_out=True)
        with torch.inference_mode():
            outputs = self.llm_model.generate(
                input_ids,
                do_sample=False,
                num_beams=self.beam,
                max_new_tokens=self.max_tokens,
                use_cache=True,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
                **kwargs,
            )
        remove_hooks(hooks)

        answer = self.tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)[0].strip()
        if return_more:
            return answer, input_ids, outputs
        return answer
