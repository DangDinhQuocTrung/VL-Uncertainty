import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient


DTU_BASE_URL = "https://hackerton2026.compute.dtu.dk/v1"
DTU_API_KEY = "sk-oJU1JCZtvh5YDqy4hjIIaA"
DTU_MODEL = "alibaba/qwen-3.6-35b-a3b"
# DTU_BASE_URL = "https://openrouter.ai/api/v1"
# DTU_API_KEY = "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"
# DTU_MODEL = "qwen/qwen3.6-35b-a3b"
NO_THINK_DIRECTIVE = "/no_think"


def apply_no_think(prompt: str) -> str:
    prompt = prompt.rstrip()
    if prompt.endswith(NO_THINK_DIRECTIVE):
        return prompt
    return f"{prompt}\n{NO_THINK_DIRECTIVE}"


def create_dtu_model_client(
    model: str = DTU_MODEL,
    base_url: str = DTU_BASE_URL,
    api_key: str | None = None,
) -> OpenAIChatCompletionClient:
    key = DTU_API_KEY
    if not key:
        raise ValueError(
            "Set DTU_API_KEY (or OPENAI_API_KEY) to your Bearer token."
        )

    return OpenAIChatCompletionClient(
        model=model,
        base_url=base_url,
        api_key=key,
        seed=99,
        temperature=0.0,
        max_retries=3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )


async def chat(
    model: str = DTU_MODEL,
    base_url: str = DTU_BASE_URL,
    system_message: str = "You are a helpful AI assistant.",
) -> None:
    model_client = create_dtu_model_client(model=model, base_url=base_url)

    assistant = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message=apply_no_think(system_message),
    )

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        [assistant],
        max_turns=1,
        termination_condition=termination,
    )

    print(f"Chat with {model} at {base_url}")
    print("Set DTU_API_KEY to your Bearer token. Type 'exit' to quit.\n")
    task: str | None = input("You: ").strip()
    if not task or task.lower() == "exit":
        await model_client.close()
        return

    try:
        while True:
            await Console(team.run_stream(task=task), output_stats=True)

            task = input("\nYou: ").strip()
            if not task or task.lower() == "exit":
                break
    finally:
        await model_client.close()


def check_agent() -> None:
    model = os.environ.get("DTU_MODEL", DTU_MODEL)
    base_url = os.environ.get("DTU_BASE_URL", DTU_BASE_URL)
    asyncio.run(chat(model=model, base_url=base_url))


if __name__ == "__main__":
    check_agent()
