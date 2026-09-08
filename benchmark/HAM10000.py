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
        prompt = item["text"]
        prompt = prompt + "\nPlease answer in the following format: {letter}:{option text}."
        prompt = prompt + "For example, if the correct answer is 'B', the answer should be B:scalp."

        result = {
            "idx": idx,
            "img": image_path,
            "question": prompt,
            "gt_answer": item["fig_caption"],
            "gender": item["gender"],
            "age": item["age"],
        }
        return result


if __name__ == "__main__":
    benchmark = HAM10000()
    print(benchmark.retrieve(0))
