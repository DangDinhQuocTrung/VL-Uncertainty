import os
import json
import numpy as np
import cv2
from datasets import load_dataset


class FairVLMed:

    def __init__(self):
        self.ds = load_dataset("harvardairobotics/FairVLMed")
        self.image_dir = "/work3/dida/dataset/FairVLMed/Test"
        self.qa_path = "/work3/dida/dataset/CARES/data/Harvard-FairVLMed/fundus_factuality.jsonl"
        with open(self.qa_path, "r") as f:
            self.qa_data = [json.loads(line) for line in f]
        # print(len(self.qa_data), self.qa_data[0])

    def obtain_size(self):
        return len(self.qa_data)

    def retrieve(self, idx):
        item = self.qa_data[idx]
        image_path = item["image"]
        image_number = os.path.splitext(image_path)[0].split("_")[-1]
        npz_name = f"data_{image_number}.npz"
        npz_path = os.path.join(self.image_dir, npz_name)
        row = np.load(npz_path)
        image_path = os.path.join(self.image_dir, npz_name.replace(".npz", ".png"))

        prompt = (
            "You are an expert in ophthalmology specializing in fundus image interpretation. "
            "Analyze the provided scanning laser ophthalmoscopy (SLO) fundus image and answer "
            "the clinical question, which may involve ocular conditions such as glaucoma, optic "
            "disc appearance, related findings, history, or management inferred from the fundus presentation.\n"
        )
        prompt += (
            "Instructions: Provide a clear, concise clinical answer based on the fundus image findings.\n"
        )
        prompt += f"Question: {item['text'].strip()}\n"
        prompt += "Answer: "

        result = {
            "idx": idx,
            "img": image_path,
            "question": prompt,
            "gt_answer": item["answer"],
            "gender": str(row["gender"]),
            "race": str(row["race"]),
            "age": row["age"].item(),
            "ethnicity": str(row["ethnicity"]),
            "language": str(row["language"]),
        }
        return result


def write_fundus_images():
    root_dir = "/work3/dida/dataset/FairVLMed/Test"
    for file in os.listdir(root_dir):
        if file.endswith(".npz"):
            npz_path = os.path.join(root_dir, file)
            with np.load(npz_path) as data:
                image = data["slo_fundus"]
                image_path = os.path.join(root_dir, file.replace(".npz", ".png"))
                cv2.imwrite(image_path, image)
    return


if __name__ == "__main__":
    # write_fundus_images()
    benchmark = FairVLMed()
    print(benchmark.retrieve(0))
