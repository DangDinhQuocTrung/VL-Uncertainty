import os
import time

from anthropic import Anthropic, APIStatusError


class Claude:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Export it before using a Claude judge."
            )
        self.client = Anthropic(api_key=api_key)
        # API models have no local weights. EUQ falls back when this is missing.
        self.model = None

    def generate(self, question, temp):
        temperature = min(max(float(temp), 0.0), 1.0)
        last_error = None
        for attempt in range(5):
            try:
                response = self.client.messages.create(
                    model=self.version,
                    max_tokens=64,
                    # temperature=temperature,
                    messages=[{"role": "user", "content": question}],
                    output_config={"effort": "low"},
                )
                return "".join(
                    block.text
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                )
            except APIStatusError as e:
                last_error = e
                if e.status_code in (429, 500, 502, 503, 529) and attempt < 4:
                    time.sleep(2**attempt)
                    continue
                raise
        raise last_error
