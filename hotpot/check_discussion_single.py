import asyncio
import re
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult, TerminationCondition
from autogen_agentchat.base._termination import TerminatedException
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseChatMessage, StopMessage, TextMessage, ThoughtEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from datasets import load_dataset

from hotpot.check_agent import create_dtu_model_client

QUESTION_INDEX = 0
CONTEXT_SET_SIZE = 2
ANSWER_HISTORY_TOKEN = "ANSWER_HISTORY:"
AGENT_ANSWER_TOKEN = "HOTPOT_ANSWER:"
FINAL_ANSWER_TOKEN = "FINAL_HOTPOT_ANSWER:"
CONFIDENCE_TOKEN = "CONFIDENCE:"
ANSWER_MAX_WORDS = 5
EVALUATOR_NAME = "evaluator"
VERDICT_TOKEN = "VERDICT:"


@dataclass(frozen=True)
class HotpotDiscussionOutput:
    question: str
    ground_truth: str
    model_answer: str


@dataclass(frozen=True)
class HotpotEvaluationResult:
    correct: bool
    explanation: str
    raw_response: str


class AgentTextMentionTermination(TerminationCondition):
    """Stop when an agent's visible reply contains the answer token.

    Unlike TextMentionTermination, this ignores ThoughtEvent messages.
    Reasoning models often mention the answer format inside their hidden
    chain-of-thought, which would end the chat before any real discussion.
    """

    def __init__(self, text: str, sources: Sequence[str] | None = None) -> None:
        self._text = text
        self._sources = sources
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def reset(self) -> None:
        self._terminated = False

    async def __call__(
        self, messages: Sequence[BaseChatMessage | ThoughtEvent]
    ) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")

        for message in messages:
            if not isinstance(message, TextMessage):
                continue
            if self._sources is not None and message.source not in self._sources:
                continue
            if self._text in message.content:
                self._terminated = True
                return StopMessage(
                    content=f"Text '{self._text}' mentioned",
                    source="AgentTextMentionTermination",
                )

        return None


def load_hotpot_dataset(split: str = "validation"):
    return load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)


def load_hotpot_row(index: int, split: str = "validation", dataset=None) -> dict:
    if dataset is None:
        dataset = load_hotpot_dataset(split)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Question index {index} out of range for split '{split}' (size {len(dataset)})")
    return dataset[index]


def build_result_record(
    output: HotpotDiscussionOutput,
    evaluation: HotpotEvaluationResult,
) -> dict[str, str | bool]:
    return {
        "question": output.question,
        "gt_answer": output.ground_truth,
        "model_answer": output.model_answer,
        "correct": evaluation.correct,
    }


def format_passage(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())


def build_context_sets(
    row: dict,
    set_size: int = CONTEXT_SET_SIZE,
) -> list[list[tuple[str, list[str]]]]:
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    passages = list(zip(titles, sentences))
    return [passages[i : i + set_size] for i in range(0, len(passages), set_size)]


def format_context_set(context_set: list[tuple[str, list[str]]]) -> str:
    return "\n\n---\n\n".join(format_passage(sents) for _, sents in context_set)


def build_context_agents(model_client, row: dict) -> list[AssistantAgent]:
    agents: list[AssistantAgent] = []

    for i, context_set in enumerate(build_context_sets(row)):
        passages = format_context_set(context_set)
        system_message = textwrap.dedent(
            f"""\
            You are research agent {i + 1}. You have read only the passages below.
            Do not invent facts outside these passages.
            Share relevant details from your passages with the other agents.
            Ask clarifying questions when you need facts another passage may contain.
            Every message must end with:
            {ANSWER_HISTORY_TOKEN} <all previous {AGENT_ANSWER_TOKEN} values in the discussion in an array, e.g. ["answer1", "answer2", "answer3"]>
            {AGENT_ANSWER_TOKEN} <your current best answer>
            {CONFIDENCE_TOKEN} <percentage from 0% to 100%, e.g. 85%>
            The answer after {AGENT_ANSWER_TOKEN} must be at most {ANSWER_MAX_WORDS} words.
            {CONFIDENCE_TOKEN} reflects how certain you are of that answer.
            When the group agrees on the final answer, also add:
            {FINAL_ANSWER_TOKEN} <the agreed short answer>
            {CONFIDENCE_TOKEN} <percentage from 0% to 100%, e.g. 85%>
            The answer after {FINAL_ANSWER_TOKEN} must also be at most {ANSWER_MAX_WORDS} words.
            The second {CONFIDENCE_TOKEN} reflects how certain you are of the final answer.
            Only use {FINAL_ANSWER_TOKEN} once the group has reached consensus.

            {passages}
            """
        )
        agents.append(
            AssistantAgent(
                f"context_{i}",
                model_client=model_client,
                system_message=system_message,
            )
        )

    return agents


def build_discussion_team(model_client, row: dict) -> RoundRobinGroupChat:
    context_agents = build_context_agents(model_client, row)
    agent_names = [agent.name for agent in context_agents]
    max_turns = len(context_agents) * 2
    termination = AgentTextMentionTermination(
        FINAL_ANSWER_TOKEN, sources=agent_names
    ) | MaxMessageTermination(max_turns)
    return RoundRobinGroupChat(context_agents, termination_condition=termination)


def print_agent_assignments(row: dict) -> None:
    print("Agent assignments:", flush=True)
    for i, context_set in enumerate(build_context_sets(row)):
        titles = ", ".join(title for title, _ in context_set)
        print(f"  context_{i}: {titles}", flush=True)
    print(flush=True)


def print_message(message: object) -> None:
    if isinstance(message, ThoughtEvent):
        preview = message.content.strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"\n({message.source} thinking) {preview}", flush=True)
        return

    if isinstance(message, TextMessage):
        print(f"\n{'=' * 80}", flush=True)
        print(f"[{message.source}]", flush=True)
        print(f"{'-' * 80}", flush=True)
        print(message.content, flush=True)
        return

    if isinstance(message, BaseChatMessage) and hasattr(message, "content") and message.content:
        print(f"\n{'=' * 80}", flush=True)
        print(f"[{message.source}] ({type(message).__name__})", flush=True)
        print(f"{'-' * 80}", flush=True)
        print(message.content, flush=True)
        return

    if isinstance(message, TaskResult):
        print(f"\n{'#' * 80}", flush=True)
        print("Discussion finished.", flush=True)
        print(f"Stop reason: {message.stop_reason}", flush=True)
        print(f"Messages exchanged: {len(message.messages)}", flush=True)
        print("Content:", message, flush=True)
        print(f"{'#' * 80}\n", flush=True)
        return

    print(f"\n[{type(message).__name__}] {message}", flush=True)


async def run_and_print_discussion(team: RoundRobinGroupChat, task: str) -> TaskResult | None:
    print("Starting discussion...\n", flush=True)
    result: TaskResult | None = None

    async for message in team.run_stream(task=task):
        print_message(message)
        if isinstance(message, TaskResult):
            result = message

    sys.stdout.flush()
    return result


def extract_answer_after_token(text: str, token: str) -> str:
    if token not in text:
        return ""

    remainder = text.split(token, 1)[1].strip()
    for line in remainder.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(CONFIDENCE_TOKEN)
        ):
            continue
        return stripped.split(CONFIDENCE_TOKEN, 1)[0].strip()

    return ""


def extract_model_answer(result: TaskResult | None) -> str:
    if result is None:
        return ""

    for message in reversed(result.messages):
        if not isinstance(message, TextMessage) or not message.source.startswith("context_"):
            continue
        answer = extract_answer_after_token(message.content, FINAL_ANSWER_TOKEN)
        if answer:
            return answer

    return ""


def build_evaluator_agent(model_client) -> AssistantAgent:
    system_message = textwrap.dedent(
        f"""\
        You are a HotpotQA answer evaluator.
        Compare the model answer against the ground-truth answer for the same question.
        Treat answers as correct when they refer to the same entity or fact, even if wording
        differs slightly (for example, extra articles, punctuation, or short aliases).
        Treat answers as incorrect when they name a different entity, give the wrong fact,
        or are too vague to match the ground truth.
        Explain your reasoning briefly, then end with exactly one line:
        {VERDICT_TOKEN} CORRECT
        or
        {VERDICT_TOKEN} INCORRECT
        """
    )
    return AssistantAgent(
        EVALUATOR_NAME,
        model_client=model_client,
        system_message=system_message,
    )


def parse_evaluation_verdict(response: str) -> bool | None:
    match = re.search(rf"{re.escape(VERDICT_TOKEN)}\s*(CORRECT|INCORRECT)\b", response, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).upper() == "CORRECT"


async def evaluate_hotpot_answer(
    output: HotpotDiscussionOutput,
    model_client=None,
) -> HotpotEvaluationResult:
    owns_client = model_client is None
    if model_client is None:
        model_client = create_dtu_model_client()

    evaluator = build_evaluator_agent(model_client)
    task = textwrap.dedent(
        f"""\
        Question: {output.question}

        Ground-truth answer: {output.ground_truth}

        Model answer: {output.model_answer or "<empty>"}

        Decide whether the model answer is correct for HotpotQA.
        """
    )

    try:
        result = await evaluator.run(task=task)
        raw_response = ""
        for message in reversed(result.messages):
            if isinstance(message, TextMessage) and message.source == EVALUATOR_NAME:
                raw_response = message.content.strip()
                break

        correct = parse_evaluation_verdict(raw_response)
        if correct is None:
            return HotpotEvaluationResult(
                correct=False,
                explanation="Evaluator did not return a parseable verdict.",
                raw_response=raw_response,
            )

        return HotpotEvaluationResult(
            correct=correct,
            explanation=raw_response,
            raw_response=raw_response,
        )
    finally:
        if owns_client:
            await model_client.close()


def build_discussion_task(row: dict) -> str:
    return textwrap.dedent(
        f"""\
        HotpotQA question: {row["question"]}

        Each context agent has been given a set of Wikipedia passages.
        Discuss the evidence together.
        Every agent must end each message with:
        {ANSWER_HISTORY_TOKEN} <all previous {AGENT_ANSWER_TOKEN} values in the discussion in an array, e.g. ["answer1", "answer2", "answer3"]>
        {AGENT_ANSWER_TOKEN} <answer>
        {CONFIDENCE_TOKEN} <percentage from 0% to 100%, e.g. 85%>
        Each {AGENT_ANSWER_TOKEN} answer must be at most {ANSWER_MAX_WORDS} words.
        {CONFIDENCE_TOKEN} reflects how certain the agent is of that answer.
        When the group reaches consensus, the speaking agent should also add:
        {FINAL_ANSWER_TOKEN} <answer>
        {CONFIDENCE_TOKEN} <percentage from 0% to 100%, e.g. 85%>
        Each {FINAL_ANSWER_TOKEN} answer must also be at most {ANSWER_MAX_WORDS} words.
        The second {CONFIDENCE_TOKEN} reflects certainty in the final agreed answer.
        """
    )


async def discuss_hotpot_question(index: int = QUESTION_INDEX) -> HotpotDiscussionOutput:
    row = load_hotpot_row(index)
    model_client = create_dtu_model_client()
    team = build_discussion_team(model_client, row)
    num_context_agents = len(build_context_sets(row))
    max_turns = num_context_agents * 2

    print(f"Question index: {index}", flush=True)
    print(f"Question: {row['question']}", flush=True)
    print(f"Context passages: {len(row['context']['title'])}", flush=True)
    print(f"Context agents: {num_context_agents} ({CONTEXT_SET_SIZE} passages each)", flush=True)
    print(f"Discussion turns (max): {max_turns}", flush=True)
    print(f"Ground-truth answer (for comparison): {row['answer']}", flush=True)
    print_agent_assignments(row)

    task = build_discussion_task(row)
    print("Task given to the team:", flush=True)
    print(task, flush=True)

    output = HotpotDiscussionOutput(
        question=row["question"],
        ground_truth=row["answer"],
        model_answer="",
    )

    try:
        result = await run_and_print_discussion(team, task)
        model_answer = extract_model_answer(result)
        output = HotpotDiscussionOutput(
            question=row["question"],
            ground_truth=row["answer"],
            model_answer=model_answer,
        )

        if result is not None:
            agent_messages = [
                message
                for message in result.messages
                if isinstance(message, TextMessage) and message.source != "user"
            ]
            if not agent_messages:
                print(
                    "No agent responses were produced. "
                    f"Stop reason: {result.stop_reason}",
                    flush=True,
                )
            print("Full message log:", flush=True)
            for i, message in enumerate(result.messages, start=1):
                if isinstance(message, TextMessage):
                    print(f"\n--- message {i} [{message.source}] ---", flush=True)
                    print(message.content, flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        raise
    finally:
        await model_client.close()

    print("\nDiscussion output:", flush=True)
    print(f"  Question: {output.question}", flush=True)
    print(f"  Ground truth: {output.ground_truth}", flush=True)
    print(f"  Model answer: {output.model_answer or '<empty>'}", flush=True)

    return output


async def discuss_hotpot_question_quiet(
    index: int,
    *,
    dataset=None,
    model_client=None,
) -> HotpotDiscussionOutput:
    row = load_hotpot_row(index, dataset=dataset)
    owns_client = model_client is None
    if model_client is None:
        model_client = create_dtu_model_client()

    team = build_discussion_team(model_client, row)
    task = build_discussion_task(row)

    try:
        result = await team.run(task=task)
        model_answer = extract_model_answer(result)
        return HotpotDiscussionOutput(
            question=row["question"],
            ground_truth=row["answer"],
            model_answer=model_answer,
        )
    finally:
        if owns_client:
            await model_client.close()


async def run_hotpot_discussion_with_evaluation(
    index: int = QUESTION_INDEX,
) -> tuple[HotpotDiscussionOutput, HotpotEvaluationResult]:
    output = await discuss_hotpot_question(index)
    evaluation = await evaluate_hotpot_answer(output)

    print("\nEvaluation:", flush=True)
    print(f"  Correct: {evaluation.correct}", flush=True)
    print(f"  Explanation:\n{evaluation.explanation}", flush=True)

    return output, evaluation


def check_discussion() -> None:
    asyncio.run(run_hotpot_discussion_with_evaluation(QUESTION_INDEX))


if __name__ == "__main__":
    check_discussion()
