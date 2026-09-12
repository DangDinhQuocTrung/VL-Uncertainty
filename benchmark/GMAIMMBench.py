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
        # self.modalities = ["Histopathology"]
        self.num_questions_per_modality = 100
        # self.num_questions_per_modality = 2
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
        n_skipped_images = 0
        for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=64):
            all_modalities.update(chunk["modality"].dropna().unique().tolist())
            for modality in self.modalities:
                if counts[modality] >= self.num_questions_per_modality:
                    continue
                need = self.num_questions_per_modality - counts[modality]
                safe_indices = []
                for row_idx, row in chunk[chunk["modality"] == modality].iterrows():
                    if self._is_safe_image(row.get("image")):
                        safe_indices.append(row_idx)
                        if len(safe_indices) >= need:
                            break
                    else:
                        n_skipped_images += 1
                if not safe_indices:
                    continue
                subset = chunk.loc[safe_indices]
                kept[modality].append(subset)
                counts[modality] += len(subset)
            if all(
                counts[m] >= self.num_questions_per_modality for m in self.modalities
            ):
                break
        # print("Unique modality values:", sorted(all_modalities))
        frames = [pd.concat(parts, ignore_index=True) for parts in kept.values() if parts]
        self.ds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if n_skipped_images:
            print(f"Skipped {n_skipped_images} oversized or unreadable images")
        print(self.ds.shape)

    def obtain_size(self):
        return len(self.ds)

    def _is_safe_image(self, image_b64):
        """Reject empty, unreadable, or PIL decompression-bomb images."""
        if pd.isna(image_b64) or not str(image_b64).strip():
            return False
        try:
            with Image.open(io.BytesIO(base64.b64decode(str(image_b64)))) as image:
                max_pixels = Image.MAX_IMAGE_PIXELS
                if max_pixels and image.size[0] * image.size[1] > max_pixels:
                    return False
            return True
        except (Image.DecompressionBombError, OSError, ValueError):
            return False

    def _decode_image(self, image_b64):
        if pd.isna(image_b64) or not str(image_b64).strip():
            return None
        try:
            image = Image.open(io.BytesIO(base64.b64decode(str(image_b64))))
            max_pixels = Image.MAX_IMAGE_PIXELS
            if max_pixels and image.size[0] * image.size[1] > max_pixels:
                return None
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            return image
        except (Image.DecompressionBombError, OSError, ValueError):
            return None

    def retrieve(self, idx):
        row = self.ds.iloc[idx]
        question = str(row["question"])
        choices = ""
        choice_numbers = ""
        num_c = 0
        for i, letter in enumerate("ABCDE"):
            choice = row.get(letter, "")
            if pd.isna(choice) or not str(choice).strip():
                continue
            choices += f"({i}): {str(choice).strip()}\n"
            choice_numbers += f"({i}), "
            num_c += 1
        choice_numbers = choice_numbers[:-2]
        domain = row["modality"]

        prompt = f"You are answering a clinical question in the {domain} domain.\n"
        prompt += f"Question: {question}\n"
        prompt += f"Options:\n{choices}\n"
        prompt += (
            f"Instructions: This is a single choice question. Answer only one word with a choice number.\n"
            f"You answer must be of the format: Answer: (<choice number>).\n"
            f"Your answer must be one of: {choice_numbers}.\n"
            f"Example: Answer: (1).\n"
        )
        prompt += f"Answer: "

        answer = str(row["answer"]).strip().upper()
        answer_letters = [c for c in answer if c in "ABCDE"]
        gt_answer = "ABCDE".index(answer_letters[0]) if answer_letters else None

        result = {
            "idx": idx,
            "img": self._decode_image(row["image"]),
            "question": prompt,
            "gt_answer": str(gt_answer),
            "num_c": num_c,
        }
        return result


if __name__ == "__main__":
    benchmark = GMAIMMBench()
    print("size:", benchmark.obtain_size())
    print(benchmark.retrieve(0))
