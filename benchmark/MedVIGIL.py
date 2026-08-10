from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image


class MedVIGIL:
    """MedVIGIL MCQ probes (default format in the official evaluation harness).

    Dataset: https://huggingface.co/datasets/jhq0709/MedVIGIL
    Harness: https://github.com/hq0709/MedVIGIL
    """

    def __init__(self):
        self.root_dir = Path(
            snapshot_download(
                repo_id="jhq0709/MedVIGIL",
                repo_type="dataset",
            )
        )
        self.ds = pd.read_csv(self.root_dir / "probes_mcq.csv")

    def obtain_size(self):
        return len(self.ds)

    def _resolve_image(self, image_file):
        if pd.isna(image_file) or not str(image_file).strip():
            return None
        image_file = str(image_file).strip()
        # Perturbed probes already include the images_perturbed/ prefix.
        if image_file.startswith("images_perturbed/"):
            path = self.root_dir / image_file
        else:
            path = self.root_dir / "images" / image_file
        if not path.exists():
            return None
        return Image.open(path)

    def retrieve(self, idx):
        row = self.ds.iloc[idx]
        question = str(row["question"])
        question += "\n"
        choices = ""
        choice_numbers = ""
        num_c = 0
        for i, letter in enumerate("ABCDE"):
            choice = row.get(f"choice_{letter}", "")
            if pd.isna(choice) or not str(choice).strip():
                continue
            choices += f"({i}): {str(choice).strip()}\n"
            choice_numbers += f"{i}, "
            num_c += 1
        choice_numbers = choice_numbers[:-2]
        question += choices
        question += "\n"
        question += (
            f"This is a single choice question, answer only one word with choice number "
            f"in {choice_numbers}."
        )

        correct_letter = str(row["correct_letter"]).strip().upper()
        gt_answer = "ABCDE".index(correct_letter)

        result = {
            "idx": idx,
            "img": self._resolve_image(row["image_file"]),
            "question": question,
            "gt_answer": str(gt_answer),
            "num_c": num_c,
        }
        return result


if __name__ == "__main__":
    benchmark = MedVIGIL()
    print(benchmark.retrieve(0))
