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
import time 
import tiktoken

constraint_memory = None
validator_memory = None
llm_config = None

param_history = []

# Tool Functions
async def calculate_params_tool(conditions: dict, metric: str) -> tuple:
    """Calculates the objective metric for the HDA process based on given operating conditions.

    Args:
        conditions (dict): The current parameter values. Example:
            {
                "H101_temperature": float,  
                "F101_temperature": float, 
                "F102_temperature": float, 
                "F102_deltaP": float
            }
        metric: given objective metric

    Returns:
        - metric (str): The objective metric that the function is calculating
    """
    await asyncio.sleep(1.5)
    value = hda_objective(
        conditions['H101_temperature'], 
        conditions['F101_temperature'], 
        conditions['F102_temperature'], 
        conditions['F102_deltaP'],
        metric
    )
    if isinstance(value, str):
        value = 'Invalid Conditions'

    if conditions not in param_history:          
        param_history.append(conditions.copy())  
        await add_suggestion_memory(conditions, metric, value)
    
    return value

async def validate(
    vals: dict,
    changes: dict,
    constraints: dict
) -> dict:
    """
    Applies changes to the current parameter values and validates the result.

    This function apply a list of proposed adjustments to the current values dictionary 
    then validate the resulting updated values against the global constraints.

    Only returns updated values if the proposed changes result in a valid configuration.

    Args:
        vals (dict): The current parameter values. Example:
            {
                "H101_temperature": float,  
                "F101_temperature": float, 
                "F102_temperature": float, 
                "F102_deltaP": float
            }

        changes (dict): A dictionary of changes of each parameter, with keys being the parameter 
        and the coorsponding values being the numeric change. Example:
            {
                "H101_temperature": -5.0, 
                "F101_temperature": -10.0, 
                "F102_temperature": 15.0, 
                "F102_deltaP": -20000
            }

        constraints (dict): Dictionary defining valid ranges. Example:
            {
                "H101_temperature": [[<lower>, <upper>],
                "F101_temperature": [<lower>, <upper>],
                "F102_temperature": [<lower>, <upper>],
                "F102_deltaP": [<lower>, <upper>]
            }

    Returns:
        dict: A dictionary containing:
            - "result" (str): 
                - "All Valid" if the updated values pass validation,
                - Otherwise a description of the validation failure.
            - "conditions" (dict): 
                - The updated values if valid,
                - Otherwise the original `vals` unchanged.
    """
    updated_vals = vals.copy()
    for p in changes:
        updated_vals[p] += changes[p]

    # Validation logic
    if updated_vals in param_history:
        return {"result": "Invalid, this set of value is repeated", "values": vals}

    if updated_vals['H101_temperature'] < constraints['H101_temperature'][0] or updated_vals['H101_temperature'] > constraints['H101_temperature'][1]:
        return {
            "result": f"Invalid, H101_temperature should be within constraint {constraints['H101_temperature']}",
            "conditions": vals
        }

    if updated_vals['F101_temperature'] < constraints['F101_temperature'][0] or updated_vals['F101_temperature'] > constraints['F101_temperature'][1]:
        return {
            "result": f"Invalid, F101_temperature should be within constraint {constraints['F101_temperature']}",
            "conditions": vals
        }
    
    if updated_vals['F102_temperature'] < constraints['F102_temperature'][0] or updated_vals['F102_temperature'] > constraints['F102_temperature'][1]:
        return {
            "result": f"Invalid, F102_temperature should be within constraint {constraints['F102_temperature']}",
            "conditions": vals
        }
    
    if updated_vals['F102_deltaP'] < constraints['F102_deltaP'][0] or updated_vals['F102_deltaP'] > constraints['F102_deltaP'][1]:
        return {
            "result": f"Invalid, F102_deltaP should be within constraint {constraints['F102_deltaP']}",
            "conditions": vals
        }

    return {
        "result": "All Valid",
        "conditions": updated_vals
    }

async def add_suggestion_memory(
        conditions: dict,
        metric: str, 
        value: float
):

    current_memory = constraint_memory.content

    encoding = tiktoken.encoding_for_model(llm_config["model"])
    total_tokens = sum(len(encoding.encode(m.content)) for m in current_memory)
    if total_tokens > llm_config["model_info"]["max_tokens"]*0.9:
        current_memory.pop(1)
        
    constraint_memory.content = current_memory

    await constraint_memory.add(MemoryContent(
        content=f"H101_temperature:{conditions['H101_temperature']}, F101_temperature: {conditions['F101_temperature']}, F102_temperature: {conditions['F102_temperature']}, F102_deltaP: {conditions['F102_deltaP']}, leads to {metric} = {value}.",
        mime_type=MemoryMimeType.TEXT
    ))

async def add_context(memory, content):
    await memory.add(MemoryContent(content=content, mime_type=MemoryMimeType.TEXT))
