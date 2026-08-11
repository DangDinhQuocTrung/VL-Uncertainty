from benchmark.LLaVABench import LLaVABench
from benchmark.MMMU import MMMU
from benchmark.MMVet import MMVet
from benchmark.ScienceQA import ScienceQA
from benchmark.ViLP import ViLP
from benchmark.MisbehaviorBench import MisbehaviorBench
from benchmark.ViLP_captioning import ViLP_captioning
from benchmark.VQARAD import VQARAD
from benchmark.PathVQA import PathVQA
from benchmark.SLAKE import SLAKE
from benchmark.MedVIGIL import MedVIGIL
from benchmark.GMAIMMBench import GMAIMMBench

from llm.Claude import Claude
from llm.Qwen import Qwen
from lvlm.InternVL import InternVL
from lvlm.LLaVA import LLaVA
from lvlm.LLaVAMed1F import LLaVAMed1F
from lvlm.LLaVANeXT import LLaVANeXT
from lvlm.Qwen2VL import Qwen2VL
from lvlm.Qwen2FVL import Qwen2FVL
from lvlm.HuatuoGPTVision import HuatuoGPTVision
from lvlm.Gemma3 import Gemma3

BLACK_BOX_METHODS = [
    "vl_uncertainty",
    "semantic_entropy",
]
METHODS = [
    *BLACK_BOX_METHODS,
]

LVLM_MAP = {
    "gemma-3-4b-it": Gemma3,
    "gemma-3-12b-it": Gemma3,
    "gemma-3-27b-it": Gemma3,
    "medgemma-4b-it": Gemma3,
    "medgemma-1.5-4b-it": Gemma3,
    "Qwen2-VL-72B-Instruct": Qwen2VL,
    "Qwen2-VL-7B-Instruct": Qwen2VL,
    "Qwen2-VL-2B-Instruct": Qwen2VL,
    "Qwen2.5-VL-7B-Instruct": Qwen2FVL,
    "HuatuoGPT-Vision-7B": HuatuoGPTVision,
    "InternVL2-26B": InternVL,
    "InternVL2-8B": InternVL,
    "InternVL2-1B": InternVL,
    "llava-v1.6-vicuna-13b-hf": LLaVANeXT,
    "llava-v1.6-mistral-7b-hf": LLaVANeXT,
    "llava-1.5-13b-hf": LLaVA,
    "llava-1.5-7b-hf": LLaVA,
    "llava-med-v1.5-mistral-7b": LLaVAMed1F,
}

BENCHMARK_MAP = {
    "MMVet": MMVet,
    "LLaVABench": LLaVABench,
    "MMMU": MMMU,
    "ScienceQA": ScienceQA,
    "ViLP": ViLP,
    "MisbehaviorBench": MisbehaviorBench,
    "ViLP_captioning": ViLP_captioning,
    "VQARAD": VQARAD,
    "PathVQA": PathVQA,
    "SLAKE": SLAKE,
    "MedVIGIL": MedVIGIL,
    "GMAIMMBench": GMAIMMBench,
}

LLM_MAP = {
    "Qwen2.5-0.5B-Instruct": Qwen,
    "Qwen2.5-1.5B-Instruct": Qwen,
    "Qwen2.5-3B-Instruct": Qwen,
    "Qwen2.5-7B-Instruct": Qwen,
    "Qwen2.5-14B-Instruct": Qwen,
    "claude-sonnet-5": Claude,
}

PERTURBATION_DETECTION_DATASETS = [
    "MedVIGIL",
]

BENCHMARK_TYPE = {
    "MMVet": "FREE_FORM",
    "LLaVABench": "FREE_FORM",
    "MMMU": "MULTI_CHOICE",
    "ScienceQA": "MULTI_CHOICE",
    "ViLP": "FREE_FORM",
    "MisbehaviorBench": "FREE_FORM",
    "ViLP_captioning": "FREE_FORM",
    "VQARAD": "FREE_FORM",
    "PathVQA": "FREE_FORM",
    "SLAKE": "FREE_FORM",
    "MedVIGIL": "MULTI_CHOICE",
    "GMAIMMBench": "MULTI_CHOICE",
}


def is_choice_question(args, sample):
    return BENCHMARK_TYPE[args.benchmark] == "MULTI_CHOICE" or bool(
        sample.get("is_closed")
    )
