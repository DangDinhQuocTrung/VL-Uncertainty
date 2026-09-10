import torch
import torch.nn.functional as F
from methods.evaluate_by_llm import evaluate_answer_correctness_by_llm, evaluate_multiple_choice_answer_correctness
from utils.constants import is_choice_question
from methods.nll import compute_token_nlls, compute_nll


def generate_positive_question(llm, lvlm, sample, model_answer):
    sample_question = sample["question"]
    sample_gt_answer = sample["gt_answer"]

    prompt = f"We are generating equivalent yes/no questions for the following question: {sample_question}\n"
    prompt += f"The model answer is: {model_answer}.\n"
    prompt += "Given the question and the model answer, please generate a yes/no question that is semantically equivalent to the original question.\n"
    prompt += "Respond with the generated yes/no question as the output."

    positive_question = lvlm.generate(sample["img"], prompt, 0.1)
    return positive_question


def generate_negative_question(llm, lvlm, sample, model_answer, positive_question):
    sample_question = sample["question"]
    sample_gt_answer = sample["gt_answer"]

    prompt = f"We are paraphrasing the following question: {positive_question}\n"
    prompt += "Please generate a semantically equivalent question but flips the meaning of the question, and the answer is the opposite of the original question.\n"
    prompt += f"The goal is that a 'yes' answer to the original question should be a 'no' answer to the paraphrased question, and vice versa.\n"
    prompt += "Respond with the paraphrased question as the output.\n"

    negative_question = lvlm.generate(sample["img"], prompt, 0.1)
    return negative_question


def make_prompt(question):
    prompt = question + "\n"
    prompt += "Respond with either 'yes' or 'no' only."
    return prompt


def get_yes_no_probability(tokenizer, inputs, outputs):
    yes_tokens = ["yes", "Yes", " yes", " Yes"]
    no_tokens = ["no", "No", " no", " No"]

    step0_logits = outputs["scores"][0][0]
    yes_logits = torch.tensor([step0_logits[tokenizer.encode(token, add_special_tokens=False)] for token in yes_tokens])
    no_logits = torch.tensor([step0_logits[tokenizer.encode(token, add_special_tokens=False)] for token in no_tokens])
    yes_logit = yes_logits.sum().item()
    no_logit = no_logits.sum().item()
    return yes_logit, no_logit


def estimate_uncertainty_by_nll_yes_no(args, lvlm, sample, llm, log_dict):
    # Generate answer
    answer, inputs, outputs, _answers = lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp,
        return_more=True,
        return_mode=1,
    )
    log_dict[sample["idx"]]["answer"] = answer
    flag_answer_correct = True
    if is_choice_question(args, sample):
        flag_answer_correct, llm_answer_check = evaluate_multiple_choice_answer_correctness(llm, sample, answer)
    else:
        flag_answer_correct, llm_answer_check = evaluate_answer_correctness_by_llm(llm, sample, answer)
    log_dict[sample["idx"]]["llm_answer_check"] = llm_answer_check
    log_dict[sample["idx"]]["flag_answer_correct"] = flag_answer_correct
    log_dict[sample["idx"]]["answer_sampling_list"] = [answer]

    # Compute negative log-likelihood of the answer
    nll_mode = getattr(args, "nll_mode", "avg")
    uncertainty, avg_nll, max_nll = compute_nll(inputs, outputs, mode=nll_mode)
    log_dict[sample["idx"]]["avg_nll"] = avg_nll
    log_dict[sample["idx"]]["max_nll"] = max_nll
    log_dict[sample["idx"]]["nll_mode"] = nll_mode

    # Using positive question
    positive_question = generate_positive_question(llm, lvlm, sample, answer)
    log_dict[sample["idx"]]["positive_question"] = positive_question
    positive_answer, positive_inputs, positive_outputs, _positive_answers = lvlm.generate(
        sample["img"],
        make_prompt(positive_question),
        args.inference_temp,
        return_more=True,
        return_mode=1,
    )
    log_dict[sample["idx"]]["positive_answer"] = positive_answer
    positive_uncertainty, positive_avg_nll, positive_max_nll = compute_nll(positive_inputs, positive_outputs, mode=nll_mode)
    log_dict[sample["idx"]]["positive_answer_avg_nll"] = positive_avg_nll
    log_dict[sample["idx"]]["positive_answer_max_nll"] = positive_max_nll
    positive_yes_logit, positive_no_logit = get_yes_no_probability(lvlm.processor.tokenizer, positive_inputs, positive_outputs)
    log_dict[sample["idx"]]["positive_answer_yes_logit"] = positive_yes_logit
    log_dict[sample["idx"]]["positive_answer_no_logit"] = positive_no_logit

    # Using negative question
    negative_question = generate_negative_question(llm, lvlm, sample, answer, positive_question)
    log_dict[sample["idx"]]["negative_question"] = negative_question
    negative_answer, negative_inputs, negative_outputs, _negative_answers = lvlm.generate(
        sample["img"],
        make_prompt(negative_question),
        args.inference_temp,
        return_more=True,
        return_mode=1,
    )
    log_dict[sample["idx"]]["negative_answer"] = negative_answer
    negative_uncertainty, negative_avg_nll, negative_max_nll = compute_nll(negative_inputs, negative_outputs, mode=nll_mode)
    log_dict[sample["idx"]]["negative_answer_avg_nll"] = negative_avg_nll
    log_dict[sample["idx"]]["negative_answer_max_nll"] = negative_max_nll
    negative_yes_logit, negative_no_logit = get_yes_no_probability(lvlm.processor.tokenizer, negative_inputs, negative_outputs)
    log_dict[sample["idx"]]["negative_answer_yes_logit"] = negative_yes_logit
    log_dict[sample["idx"]]["negative_answer_no_logit"] = negative_no_logit

    # Computing uncertainty
    overall_uncertainty = (positive_uncertainty + negative_uncertainty) / 2
    log_dict[sample["idx"]]["uncertainty"] = overall_uncertainty
    log_dict[sample["idx"]]["uncertainty_threshold"] = args.uncertainty_threshold
    flag_predict_hallucination = log_dict[sample["idx"]]["uncertainty"] >= args.uncertainty_threshold
    log_dict[sample["idx"]]["flag_predict_hallucination"] = flag_predict_hallucination
    flag_detection_correct = (
        log_dict[sample["idx"]]["flag_answer_correct"] and not flag_predict_hallucination
    ) or (
        not log_dict[sample["idx"]]["flag_answer_correct"] and flag_predict_hallucination
    )
    log_dict[sample["idx"]]["flag_detection_correct"] = flag_detection_correct
    return log_dict
