import base64
import io
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


class GMAIMMBench:
    """GMAI-MMBench VAL split (answers available; TEST has no labels).

    Dataset: https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench
    """

    def __init__(self):
        self.modalities = [
            "CT", "MRI", "X-ray",
            "Endoscopy", "Microscopy", "Histopathology",
            "Fundus Photography", "Dermoscopy",
        ]
        # self.num_questions_per_modality = 200
        self.num_questions_per_modality = 1
        tsv_path = Path(
            hf_hub_download(
                repo_id="OpenGVLab/GMAI-MMBench",
                filename="GMAI_mm_bench_VAL.tsv",
                repo_type="dataset",
            )
        )
        # TSV embeds large base64 images; load in chunks and keep a global
        # per-modality quota (not a per-chunk slice).
        kept = {m: [] for m in self.modalities}
        counts = {m: 0 for m in self.modalities}
        all_modalities = set()
        for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=64):
            all_modalities.update(chunk["modality"].dropna().unique().tolist())
            for modality in self.modalities:
                if counts[modality] >= self.num_questions_per_modality:
                    continue
                need = self.num_questions_per_modality - counts[modality]
                subset = chunk[chunk["modality"] == modality].head(need)
                if len(subset) == 0:
                    continue
                kept[modality].append(subset)
                counts[modality] += len(subset)
            if all(
                counts[m] >= self.num_questions_per_modality for m in self.modalities
            ):
                break
        # print("Unique modality values:", sorted(all_modalities))
        frames = [pd.concat(parts, ignore_index=True) for parts in kept.values() if parts]
        self.ds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        print(self.ds.shape)

    def obtain_size(self):
        return len(self.ds)

    def _decode_image(self, image_b64):
        if pd.isna(image_b64) or not str(image_b64).strip():
            return None
        image = Image.open(io.BytesIO(base64.b64decode(str(image_b64))))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        return image

    def retrieve(self, idx):
        row = self.ds.iloc[idx]
        question = str(row["question"])
        question += "\n"
        choices = ""
        choice_numbers = ""
        num_c = 0
        for i, letter in enumerate("ABCDE"):
            choice = row.get(letter, "")
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

        answer = str(row["answer"]).strip().upper()
        answer_letters = [c for c in answer if c in "ABCDE"]
        gt_answer = "ABCDE".index(answer_letters[0]) if answer_letters else None

        result = {
            "idx": idx,
            "img": self._decode_image(row["image"]),
            "question": question,
            "gt_answer": gt_answer,
            "num_c": num_c,
        }
        return result


if __name__ == "__main__":
    benchmark = GMAIMMBench()
    print("size:", benchmark.obtain_size())
    print(benchmark.retrieve(0))
