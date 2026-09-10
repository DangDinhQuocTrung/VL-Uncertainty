from datasets import load_dataset


class PathVQA:

    def __init__(self):
        self.ds = load_dataset("flaviagiammarino/path-vqa")

    def obtain_size(self):
        return len(self.ds["test"])

    def retrieve(self, idx):
        row = self.ds["test"][idx]
        prompt = (
            "You are an expert in pathology specializing in histopathology and cytology "
            "image interpretation. Analyze the provided pathology image and answer the "
            "visual question, which may involve tissue type, cellular morphology, staining, "
            "location, abnormal findings, or related pathology reasoning.\n"
        )
        prompt += (
            "Instructions: Please give a concise answer within five words. "
            "Please use only one word to answer if possible.\n"
        )
        prompt += f"Question: {row['question'].strip()}\n"
        prompt += "Answer: "

        result = {
            "idx": idx,
            "img": row["image"],
            "question": prompt,
            "gt_answer": row["answer"],
        }
        return result


if __name__ == "__main__":
    benchmark = PathVQA()
    print(benchmark.retrieve(0))
