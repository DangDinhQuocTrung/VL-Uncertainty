import os
import json
import numpy as np
import cv2
from datasets import load_dataset


class HAM10000:

    def __init__(self):
        self.image_dir = "/work3/dida/dataset/HAM10000/images"
        self.qa_path = "/work3/dida/dataset/CARES/data/HAM10000/HAM10000_factuality.jsonl"
        with open(self.qa_path, "r") as f:
            self.qa_data = [json.loads(line) for line in f]
        print(len(self.qa_data), self.qa_data[0])

    def obtain_size(self):
        return len(self.qa_data)

    def retrieve(self, idx):
        item = self.qa_data[idx]
        image_path = item["image"]
        image_path = os.path.join(self.image_dir, os.path.basename(image_path))
        # Keep the question/options from CARES; replace the generic medical preamble.
        text = item["text"]
        prompt = (
            "You are an expert in dermatology specializing in dermoscopic image interpretation. "
            "Analyze the provided dermatoscopic (dermoscopy) image of a pigmented skin lesion "
            "and answer the multiple-choice clinical question, which may ask about lesion "
            "type/diagnosis or the anatomical site of the lesion.\n"
        )
        prompt = prompt + "Instructions: Please answer in the following format: option_character:option_text."
        prompt = prompt + "Example: If the correct answer is 'B', the answer should be B:scalp.\n"
        prompt = prompt + text + "\n"
        prompt = prompt + "Answer: "

        result = {
            "idx": idx,
            "img": image_path,
            "question": prompt,
            "gt_answer": item["fig_caption"],
            "gender": item["gender"],
            "age": item["age"],
            "num_c": 4,
        }
        return result


def plot_demographic_distribution():
    import matplotlib.pyplot as plt

    qa_path = "/work3/dida/dataset/CARES/data/HAM10000/HAM10000_factuality.jsonl"
    with open(qa_path, "r") as f:
        qa_data = [json.loads(line) for line in f]

    genders = [item["gender"] for item in qa_data]
    ages = np.array([item["age"] for item in qa_data])
    print(len(genders), len(ages))
    plt.hist(genders, bins=len(set(genders)))
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Gender Distribution")
    plt.savefig("exp/gender_distribution.png")
    plt.close()
    plt.hist(ages)
    plt.xlabel("Age")
    plt.xticks(range(0, 100, 10))
    plt.ylabel("Count")
    plt.title("Age Distribution")
    plt.savefig("exp/age_distribution.png")
    plt.close()


if __name__ == "__main__":
    plot_demographic_distribution()
