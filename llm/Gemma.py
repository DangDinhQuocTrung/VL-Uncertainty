import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Gemma:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def _model_id(self):
        if "/" in self.version:
            return self.version
        return f"google/{self.version}"

    def _is_gemma3_family(self):
        name = self.version.lower()
        return "gemma-3" in name or "medgemma" in name

    def build_model(self):
        model_name = self._model_id()
        model_cls = AutoModelForCausalLM
        if self._is_gemma3_family():
            try:
                from transformers import Gemma3ForCausalLM

                # Text-only load: omits the vision tower on multimodal Gemma 3 / MedGemma.
                model_cls = Gemma3ForCausalLM
            except ImportError:
                model_cls = AutoModelForCausalLM
        dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
        self.model = model_cls.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _messages(self, question):
        if self._is_gemma3_family():
            return [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                },
            ]
        return [{"role": "user", "content": question}]

    def generate(self, question, temp):
        inputs = self.tokenizer.apply_chat_template(
            self._messages(question),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        temperature = max(float(temp), 1e-5)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=temperature,
            top_p=0.8,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response = self.tokenizer.decode(
            generated_ids[0][input_len:], skip_special_tokens=True
        )
        return response.strip()
