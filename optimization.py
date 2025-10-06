import json
import asyncio
from typing import Sequence

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

from pyomo.environ import (
    Constraint,
    Var,
    ConcreteModel,
    Expression,
    Objective,
    SolverFactory,
    TransformationFactory,
    value,
)

from pyomo.network import Arc, SequentialDecomposition

from idaes.core import FlowsheetBlock

from idaes.models.unit_models import (
    PressureChanger,
    Mixer,
    Separator as Splitter,
    Heater,
    CSTR,
)

from idaes.models.unit_models import Flash

from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.core.util.model_statistics import degrees_of_freedom

import idaes.logger as idaeslog
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import InitializationError

from idaes_examples.mod.hda import hda_ideal_VLE as thermo_props
from idaes_examples.mod.hda import hda_reaction as reaction_props

import logging

from hda_objective_function import hda_objective
from agent_helper_function import calculate_params_tool, validate, add_context

import agent_helper_function

async def run_main(initial_params, metric, context, llm_config):

    model_client = OpenAIChatCompletionClient(
        api_key=llm_config["api_key"],
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        model_info=llm_config["model_info"], 
        # temperature=0.3
    )

    validator_agent = AssistantAgent(
        "ValidatorAgent",
        model_client=model_client,
        tools=[validate],
        memory = [agent_helper_function.validator_memory], 
        description="Validates whether proposed values for H101_temperature, F101_temperature, F102_temperature, and F102_deltaP fall within defined process constraints.",
        system_message=(
            """
            You are the ValidatorAgent. Your response should be a function call. You should only call the function once.
            You suppose to construct the dictionary of the constraint parameter from the memory. 
            Your job is to validate the change made to candidates are valid for values of:
            - H101_temperature: the temperature output of the heater.
            - F101_temperature: the temperature output of the first flash separation unit.
            - F102_temperature: the temperature output of the secondary flash separation unit.
            - F102_deltaP: the pressure change in the secondary flash separation unit.
            if the proposed change are valid, update the coorsponding parameters

            You obtained the amount of change directly from the previous agent.
            You MUST call the function `validate` only once per message, with the following arguments:
            - `vals`: a dictionary with keys `"H101_temperature"`, `"F101_temperature"`, `"F102_temperature"`, and `"F102_deltaP"` that holds the current candidate values.
            - `changes`: a dictionary of the numeric amount to add or subtract.
            - `constraints`: a dictionary of the operational constraints of the four variables.
            if no changes is provided, use 0 for all change.
            Only call the function once.
        """
        )
    )

    # Agents
    simulation_agent = AssistantAgent(
        "MetricCalculationAgent",
        description = """
        simulation_agent calculates objective metric and updates progress with the one set of values H101_temperature, F101_temperature, F102_temperature, and F102_deltaP.
        """,
        model_client=model_client,
        tools = [calculate_params_tool],
        system_message=(
            f"""
            You are the param agent. Your response must be a function call. You should only call the function once. 
            Your only job is to calculate the objective metric using the function provided - You do not change the condition value of H101_temperature, F101_temperature, F102_temperature, and F102_deltaP.
            You should call the  with the following parameters:
            - `conditions`: a dictionary with keys `"H101_temperature"`, `"F101_temperature"`, `"F102_temperature"`, and `"F102_deltaP"` that holds the current candidate values, obtaining from the validation agent.
            - `metric`: the {metric} metric as provided.
            make sure you include both `conditions` and 'metric` in your function call.
            """
        ),
    )
    
    suggestion_agent = AssistantAgent(
        "SuggestionAgent",
        description="Suggests parameter changes using dynamic step sizing to optimize the giving metric.",
        model_client=model_client,
        tools = [],
        memory = [agent_helper_function.constraint_memory], 
        system_message=(
            """
            You are SuggestionAgent.

            What you can see
            ────────────────
            1. constraint_memory  (chronological)
            - First lines: static constraint for the HDA problem.  
            - Subsequent lines: records that look like
                H101_temperature:<val>, F101_temperature:<val>,
                F102_temperature:<val>, F102_deltaP:<val>,
                leads to Cost=<cost>, Metric=<metric>.
                These entries exist **only for parameter sets that passed validation**.

            2. The conversation stream
            - If your previous proposal was invalid, the immediately-preceding message
                will come from ValidatorAgent and start with “Invalid, …”.
            - Use the reason in that message (e.g. which limit was exceeded) when
                adjusting your next suggestion.

            Objective (one of the following)
            ────────────────────────
            • Higher yield, or
            • Higher yield/cost, or
            • Higher revenue, (higher temperature of H101 and F101, lower temperature of F102 leads to higher revenue, try boundaries)or 
            • Lower cost

            Rules for every turn
            ────────────────────
            1. Parse the entire constraint_memory to understand long-term trends.
            2. Look at the most recent VALID parameter set (last line in memory).  
            Also check the last chat message:  
                • If it begins with “Invalid,” treat your last increments as rejected.  
                - Shrink or reverse the offending increment so the result will fall
                    inside its constraint window.  
                - Leave other increments unchanged **unless** you have evidence from
                    history that a different direction is better.

            3. Produce **one** Python dict literal (not JSON) called changes, e.g.

            {'H101_temperature': -10,
                'F101_temperature':   5,
                'F102_temperature':   0,
                'F102_deltaP':    -5000}

            • All four keys must appear.  
            • The increments are RELATIVE adjustments (can be negative, positive, or 0).  

            4. If you judge no further improvement is possible, output exactly:
                TERMINATE
            (uppercase, nothing else).

            Do NOT call any function.
            """
        )
    )


    # You are the parameter_agent, your job is to decide initial optimal values for the temperature of heater and the pressure of the feed given the context of the problem.
    # your output be H101_temperature=..., comp_ratio=...
    parameter_agent = AssistantAgent(
        "parameter_agent",
        model_client=model_client,
        description="initializes H101_temperature, F101_temperature, F102_temperature, F102_deltaP values and objective metrics.",
        system_message=(
            f""" 
            You are the parameter_agent. Your task is to propose the intial parameters and the ojective meteric to the conversation chat. 
            Your response MUST strictly follow the format below: 
            Process overview: {context},
            Initial Parameters: {initial_params},
            Objective: {metric}
            """
        )
    )

    def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
        last_content = messages[-1].content
        last_source = messages[-1].source


        if last_source == "parameter_agent":
            return validator_agent.name

        if last_source == "user":
            return parameter_agent.name

        # 3) If ValidatorAgent is the last speaker, check if error or success
        if last_source == validator_agent.name:
            if "Invalid" in last_content:
                return suggestion_agent.name
            else:
                # We got "True" or "VALID:" => success
                return simulation_agent.name
        
        # 4) cost -> suggestion
        if last_source == simulation_agent.name:
            return suggestion_agent.name
        # 5) suggestion -> validate
        if last_source == suggestion_agent.name:
            if "TERMINATE" in last_content:
                return None
            return validator_agent.name

        return None
    
    termination = TextMentionTermination("TERMINATE")

    # Set up the team with the predefined agents
    team = SelectorGroupChat(
        [parameter_agent, validator_agent, simulation_agent, suggestion_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt="You are the high-level conversation controller.",
        allow_repeated_speaker=False,
        selector_func=selector_func,
    )

    # Define the task and run the team
    task = ""
    result = await Console(team.run_stream(task=task))
    return result

def setup_and_run(
    context: str,
    constraint_text: str,
    llm_config: dict,
    optimization_config: dict
) -> dict:
    """
    Run the HDA optimization process using pre-loaded content instead of file paths.

    Args:
        context (str): The process overview string.
        constraint_text (str): The constraint description string.
        llm_config (dict): The OpenAI config (with api_key, model, etc.).
        optimization_config (dict): The HDA configuration (with initial_params, metric, etc.).

    Returns:
        dict: Optimization output with message log and stop reason.
    """
    # Extract parameters
    initial_params = optimization_config["initial_params"]
    metric = optimization_config["optimization_metric"]

    agent_helper_function.constraint_memory = ListMemory()
    agent_helper_function.validator_memory = ListMemory()
    agent_helper_function.llm_config = llm_config

    # Add constraint content to memory
    asyncio.run(add_context(agent_helper_function.constraint_memory, constraint_text))
    asyncio.run(add_context(agent_helper_function.validator_memory, constraint_text))

    # Run optimization process
    result = asyncio.run(run_main(initial_params, metric, context, llm_config))

    # Format results
    chat_log = [{
        "type": getattr(m, "type", None),
        "source": getattr(m, "source", None),
        "content": str(getattr(m, "content", "")),
        "metadata": getattr(m, "metadata", {}),
    } for m in result.messages]

    task_output = {
        "messages": chat_log,
        "stop_reason": result.stop_reason,
    }

    # Save to file
    output_path = optimization_config['optimization_save_path']
    with open(output_path, "w") as f:
        json.dump(task_output, f, indent=2)
    
    print("Optimization result saved to path ", output_path)

    return task_output
