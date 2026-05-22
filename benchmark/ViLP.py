from datasets import load_dataset


class ViLP:

    def __init__(self):
        self.ds = load_dataset("ViLP/ViLP")
        self.num_images_per_question = 3

    def obtain_size(self):
        return len(self.ds["train"]) * self.num_images_per_question

    def retrieve(self, idx):
        row_idx = idx // self.num_images_per_question
        image_idx = idx % self.num_images_per_question + 1

        row = self.ds["train"][row_idx]
        question = f"{row['question']}\nNOTE: Please answer with one word."
        result = {
            "idx": idx,
            "img": row[f"image{image_idx}"],
            "question": question,
            "gt_ans": row[f"answer{image_idx}"],
        }
        return result


if __name__ == "__main__":
    benchmark = ViLP()
    print(benchmark.retrieve(0))
