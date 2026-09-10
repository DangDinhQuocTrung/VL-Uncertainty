from datasets import load_dataset


class VQARAD:

    def __init__(self):
        self.ds = load_dataset("flaviagiammarino/vqa-rad")

    def obtain_size(self):
        return len(self.ds["test"])

    def retrieve(self, idx):
        row = self.ds["test"][idx]
        prompt = (
            "You are an expert in radiology specializing in clinical medical image "
            "interpretation. Analyze the provided radiology image (e.g., head, chest, or "
            "abdomen) and answer the clinician-authored visual question, which may involve "
            "modality, plane, organ system, abnormality, presence/absence of findings, or "
            "related radiology reasoning.\n"
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
    benchmark = VQARAD()
    print(benchmark.retrieve(0))
