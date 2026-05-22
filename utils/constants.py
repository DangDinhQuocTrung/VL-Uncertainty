from benchmark.LLaVABench import LLaVABench
from benchmark.MMMU import MMMU
from benchmark.MMVet import MMVet
from benchmark.ScienceQA import ScienceQA
from benchmark.ViLP import ViLP
from benchmark.MisbehaviorBench import MisbehaviorBench

from llm.Qwen import Qwen
from lvlm.InternVL import InternVL
from lvlm.LLaVA import LLaVA
from lvlm.LLaVANeXT import LLaVANeXT
from lvlm.Qwen2VL import Qwen2VL

BLACK_BOX_METHODS = [
    "vl_uncertainty",
    "semantic_entropy",
]
METHODS = [
    *BLACK_BOX_METHODS,
]

LVLM_MAP = {
    "Qwen2-VL-72B-Instruct": Qwen2VL,
    "Qwen2-VL-7B-Instruct": Qwen2VL,
    "Qwen2-VL-2B-Instruct": Qwen2VL,
    "InternVL2-26B": InternVL,
    "InternVL2-8B": InternVL,
    "InternVL2-1B": InternVL,
    "llava-v1.6-vicuna-13b-hf": LLaVANeXT,
    "llava-v1.6-mistral-7b-hf": LLaVANeXT,
    "llava-1.5-13b-hf": LLaVA,
    "llava-1.5-7b-hf": LLaVA,
}

BENCHMARK_MAP = {
    "MMVet": MMVet,
    "LLaVABench": LLaVABench,
    "MMMU": MMMU,
    "ScienceQA": ScienceQA,
    "ViLP": ViLP,
    "MisbehaviorBench": MisbehaviorBench,
}

LLM_MAP = {
    "Qwen2.5-0.5B-Instruct": Qwen,
    "Qwen2.5-1.5B-Instruct": Qwen,
    "Qwen2.5-3B-Instruct": Qwen,
    "Qwen2.5-7B-Instruct": Qwen,
}

BENCHMARK_TYPE = {
    "MMVet": "FREE_FORM",
    "LLaVABench": "FREE_FORM",
    "MMMU": "MULTI_CHOICE",
    "ScienceQA": "MULTI_CHOICE",
    "ViLP": "FREE_FORM",
    "MisbehaviorBench": "FREE_FORM",
}
