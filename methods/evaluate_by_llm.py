

def evaluate_answer_correctness_by_llm(llm, sample, model_answer):
    sample_question = sample["question"]
    sample_gt_answer = sample["gt_answer"]

    # prompt = f"Question: {sample_question}. Ground truth: {sample_gt_answer}. Model answer: {model_answer}."
    # prompt += "Please verify if the model answer matches the ground truth given the question. Respond with either 'Correct' or 'Wrong' only."

    prompt = f"We are assessing the quality of answers to the following question: {sample_question}\n"
    prompt += f"The expected answer is: {sample_gt_answer}.\n"
    prompt += f"The model answer is: {model_answer}.\n"
    prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?\n"
    prompt += "Respond with either 'yes' or 'no' only."

    llm_answer_check = llm.generate(prompt, 0.001)
    llm_answer_check = llm_answer_check.strip().lower()
    flag_answer_correct = (
        "yes" in llm_answer_check
        or "y" in llm_answer_check
        or (model_answer.strip().lower() == sample_gt_answer.strip().lower())
    )
    return flag_answer_correct, llm_answer_check
