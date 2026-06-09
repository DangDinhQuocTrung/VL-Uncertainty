from datasets import load_dataset


def check_dataset():
    hotpot_qa = load_dataset("hotpotqa/hotpot_qa", "distractor")
    for row in hotpot_qa["validation"]:
        question = row["question"]
        answer = row["answer"]
        print(row.keys())
        print(question)
        print(answer)
        print(row["context"])
        print(row["supporting_facts"])
        print("-" * 100)
        break
    return


if __name__ == "__main__":
    check_dataset()
