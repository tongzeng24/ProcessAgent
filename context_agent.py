import asyncio
import json
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
import yaml, pathlib, os


async def generate_context(model_config, agent_config, loop_n) -> None:
    def _append_suffix_to_path(original_path: str, suffix: int) -> str:
        path = pathlib.Path(original_path)
        new_name = f"{path.stem}_{suffix}{path.suffix}"
        return str(path.with_name(new_name))

    model_client = OpenAIChatCompletionClient(
        api_key=model_config["api_key"],
        model=model_config["model"],
        base_url=model_config["base_url"],
        model_info=model_config["model_info"]
    )
    prompts = yaml.safe_load(pathlib.Path(agent_config["context_agent_prompt_path"]).read_text())
    agent = AssistantAgent(name="ContextAgent", model_client=model_client)

    response = await agent.on_messages(
        [TextMessage(content=prompts["context_agent_prompt"], source="user")], CancellationToken())
    
    if response and hasattr(response.chat_message, "content"):
        try:
            payload = json.loads(response.chat_message.content)
        except json.JSONDecodeError as e:
            raise RuntimeError("LLM did not return valid JSON") from e
        
        overview_path = agent_config['llm_process_overview_save_path']
        constraint_path = _append_suffix_to_path(agent_config['llm_constraint_save_path'], loop_n)

        pathlib.Path(overview_path).parent.mkdir(parents=True, exist_ok=True)
        with open(overview_path, "w", encoding="utf-8") as f:
            f.write(payload["process_overview"].strip() + "\n")

        constraint_lines = [
            f'{c["variable"]}: [{c["range"][0]} {c["unit"]}, {c["range"][1]} {c["unit"]}]'
            for c in payload["constraints"]
        ]
        
        pathlib.Path(constraint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(constraint_path, "w", encoding="utf-8") as f:
            f.write("\n".join(constraint_lines) + "\n")

        print(f"Overview   → {agent_config['llm_process_overview_save_path']}")
        print(f"Constraints→ {constraint_path}")
    else:
        print("No content received from LLM.")

