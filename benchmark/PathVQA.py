from datasets import load_dataset


class PathVQA:

    def __init__(self):
        self.ds = load_dataset("flaviagiammarino/path-vqa")

    def obtain_size(self):
        return len(self.ds["test"])

    def retrieve(self, idx):
        row = self.ds["test"][idx]
        question = f"{row['question']}\nNOTE: Please give a concise answer within five words. Please use only one word to answer if possible."
        result = {
            "idx": idx,
            "img": row["image"],
            "question": question,
            "gt_answer": row["answer"],
        }
        return result


if __name__ == "__main__":
    benchmark = PathVQA()
    print(benchmark.retrieve(0))
