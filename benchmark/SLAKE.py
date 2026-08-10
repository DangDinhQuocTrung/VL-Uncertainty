import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


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
        question = f"{row['question']}\nNOTE: Please give a concise answer within five words. Please use only one word to answer if possible."
        image = Image.open(self.img_root / row["img_name"])
        result = {
            "idx": idx,
            "img": image,
            "question": question,
            "gt_answer": row["answer"],
        }
        return result


if __name__ == "__main__":
    benchmark = SLAKE()
    print(benchmark.retrieve(0))
