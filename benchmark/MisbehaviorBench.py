import pandas as pd
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from huggingface_hub import snapshot_download


class MisbehaviorBench:

    def __init__(self):
        # snapshot_download(
        #     repo_id="TerryHWong/Misbehavior-Bench",
        #     repo_type="dataset",
        # )
        self.root_dir = Path("/work3/dida/cache/hub/datasets--TerryHWong--Misbehavior-Bench/snapshots/e65ed548ad57eb5b0e1c5bdf357a23f2b40cfcb0")
        self.task_names = ["hallucination", "jailbreak", "adversarial", "ood"]
        self.task_name = self.task_names[3]
        self.ds = pd.read_csv(self.root_dir / self.task_name / f"{self.task_name}.csv")

    def obtain_size(self):
        return len(self.ds)

    def retrieve(self, idx):
        row = self.ds.iloc[idx]
        question = f"{row['question']}\nNOTE: Please answer with one word."

        image_name = None
        if self.task_name == "hallucination":
            image_name = Path(row["image"]).name
        elif self.task_name == "adversarial":
            image_name = Path(row["images"]).name
        elif self.task_name == "jailbreak":
            image_name = Path(row["image"]).name
        elif self.task_name == "ood":
            image_name = row["id"] + ".png"
        image = Image.open(self.root_dir / self.task_name / "images" / image_name)
        answer = row["answer"]
        if self.task_name == "hallucination":
            answer = row[answer]

        result = {
            "idx": idx,
            "img": image,
            "question": question,
            "gt_ans": answer,
        }
        return result


if __name__ == "__main__":
    benchmark = MisbehaviorBench()
    print(benchmark.retrieve(0))
