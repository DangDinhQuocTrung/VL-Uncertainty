import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

CLOSED_CHOICES = ["No", "Yes"]


def _format_closed_question(question, choices):
    prompt = (
        "You are an expert in radiology specializing in clinical medical image "
        "interpretation. Analyze the provided radiology image (e.g., CT, MRI, or X-ray "
        "of the head, neck, chest, abdomen, or pelvis) and answer the clinical visual "
        "question, which may involve organ recognition, abnormality detection, position, "
        "modality/plane, size/shape/color, or knowledge-enhanced clinical reasoning.\n"
    )
    choice_numbers = ""
    options = ""
    for i, choice in enumerate(choices):
        options += f"({i}): {choice}\n"
        choice_numbers += f"({i}), "
    choice_numbers = choice_numbers[:-2]
    prompt += (
        "Instructions: This is a single choice question. Answer only one word with a "
        "choice number.\n"
        "You answer must be of the format: Answer: (<choice number>).\n"
        f"Your answer must be one of: {choice_numbers}.\n"
        "Example: Answer: (1).\n"
    )
    prompt += f"Question: {str(question).strip()}\n"
    prompt += f"Options:\n{options}"
    prompt += "Answer: "
    return prompt


def _format_open_question(question):
    prompt = (
        "You are an expert in radiology specializing in clinical medical image "
        "interpretation. Analyze the provided radiology image (e.g., CT, MRI, or X-ray "
        "of the head, neck, chest, abdomen, or pelvis) and answer the clinical visual "
        "question, which may involve organ recognition, abnormality detection, position, "
        "modality/plane, size/shape/color, or knowledge-enhanced clinical reasoning.\n"
    )
    prompt += (
        "Instructions: Please give a concise answer within five words. "
        "Please use only one word to answer if possible.\n"
    )
    prompt += f"Question: {str(question).strip()}\n"
    prompt += "Answer: "
    return prompt


class SLAKE:

    def __init__(self):
        # JSON-only dataset; images live in imgs.zip on the same HF repo.
        self.ds = load_dataset("BoKelvin/SLAKE")["test"].filter(
            lambda x: x["q_lang"] == "en"
        )
        zip_path = Path(
            hf_hub_download(
                repo_id="BoKelvin/SLAKE",
                filename="imgs.zip",
                repo_type="dataset",
            )
        )
        extract_dir = zip_path.parent
        # imgs.zip may unpack as extract_dir/imgs/... or extract_dir/xmlab*/...
        candidate_roots = [extract_dir / "imgs", extract_dir]
        if not any((root / "xmlab1" / "source.jpg").exists() for root in candidate_roots):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        self.img_root = next(
            root for root in candidate_roots if (root / "xmlab1" / "source.jpg").exists()
        )

    def obtain_size(self):
        return len(self.ds)

    def retrieve(self, idx):
        row = self.ds[idx]
        answer_text = str(row["answer"]).strip()
        is_closed = (
            str(row["answer_type"]).strip().upper() == "CLOSED"
            and answer_text.lower() in {"yes", "no"}
        )
        if is_closed:
            question = _format_closed_question(row["question"], CLOSED_CHOICES)
            gt_answer = str(CLOSED_CHOICES.index("Yes" if answer_text.lower() == "yes" else "No"))
        else:
            question = _format_open_question(row["question"])
            gt_answer = row["answer"]
        image = Image.open(self.img_root / row["img_name"])
        result = {
            "idx": idx,
            "img": image,
            "question": question,
            "gt_answer": gt_answer,
            "is_closed": is_closed,
        }
        if is_closed:
            result["num_c"] = len(CLOSED_CHOICES)
        return result


if __name__ == "__main__":
    benchmark = SLAKE()
    print(benchmark.retrieve(0))
