from datasets import load_dataset


def check_dataset():
    hotpot_qa = load_dataset("hotpotqa/hotpot_qa", "distractor")
    print(hotpot_qa["validation"].num_rows)
    for row_index, row in enumerate(hotpot_qa["validation"]):
        if row_index != 13:
            continue
        question = row["question"]
        answer = row["answer"]
        print(row.keys())
        print(question)
        print(answer)
        context = row["context"]
        titles, texts = context["title"], context["sentences"]
        for title, text in zip(titles, texts):
            print(title)
            print(text)
            print("-" * 100)
        print(row["supporting_facts"])
        print("-" * 100)
        break
    return


if __name__ == "__main__":
    check_dataset()
