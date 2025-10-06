import asyncio
import re
import json
import yaml
import pandas as pd

from pathlib import Path

from context_agent import generate_context
from optimization import setup_and_run

# load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

context_config = config["ContextAgent"]
optimization_config = config["Optimization"]
model_config = config["Model"]

if __name__=="__main__":
    # Context generation
    print('Generating operating constraints and process overview')
    for i in range(context_config["context_sampling_iterations"]):
        asyncio.run(generate_context(model_config, context_config, i+1))
    
    results_dir = Path("Results")

    constraint_files = []
    overview_files = []

    for file in results_dir.glob("*.txt"):
        if file.stem.startswith("generated_constraints_"):
            constraint_files.append(file)
        else:
            overview_files.append(file)
    for file in overview_files:
        with open(file, "r") as f:
            overview_str = f.read().rstrip()

    constraint = []
    for file in constraint_files:
        with open(file, "r") as f:
            constraint_str = f.read().rstrip()
            temp = {}
            lines = constraint_str.strip().splitlines()
            for line in lines:
                matchs = re.match(r'(.+?):\s*\[([-\d\.]+)\s*\w*,\s*([-\d\.]+)\s*\w*\]', line)
                if matchs:
                    name = matchs.group(1).strip()
                    normalized_name = name.lower().replace(" ", "_")
                    low = float(matchs.group(2))
                    high = float(matchs.group(3))
                    temp[normalized_name + " min"] = low
                    temp[normalized_name + " max"] = high
            constraint.append(temp)

    constraint_df = pd.DataFrame(constraint)
    avg = dict(constraint_df.mean().round(1))
    transformed = {}
    for key, value in avg.items():
        base_key = key.rsplit(' ', 1)[0]
        tag = base_key.lower().replace(" ", "_")
        if tag not in transformed:
            transformed[tag] = []
        transformed[tag].append(float(value))
    
    avg_constraint = '\n'.join(f"{k:<25}: [{v[0]:.2f}, {v[1]:.2f}]" for k, v in transformed.items())
    with open(context_config['llm_constraint_avg_save_path'], "w", encoding="utf-8") as f:
        f.write(avg_constraint+ "\n")

    print("Avg constraint:\n" + avg_constraint)

    print('Starting optimization based on the averaged constraints')
    # optimization
    setup_and_run(
        context=overview_str, 
        constraint_text = str(transformed), 
        llm_config = model_config,
        optimization_config = optimization_config, 
    )

    # print best result
    with open(optimization_config['optimization_save_path'], 'r') as f:
        data = json.load(f)

    messages = data["messages"]

    metric_type = optimization_config['optimization_metric']
    best_value = float("inf") if metric_type == "cost" else float("-inf")
    best_conditions = None

    for i, msg in enumerate(messages):
        if msg["type"] == "ToolCallSummaryMessage" and msg["source"] == "MetricCalculationAgent":
            try:
                value = float(msg["content"])
            except ValueError:
                continue 

            if value < 0:
                continue  

            for j in range(i - 1, -1, -1):
                prev_msg = messages[j]
                if (
                    prev_msg["type"] == "ToolCallSummaryMessage"
                    and prev_msg["source"] == "ValidatorAgent"
                    and "conditions" in prev_msg["content"]
                ):
                    conditions = eval(prev_msg["content"])["conditions"]
                    break
            else:
                continue 

            if (metric_type == "cost" and value < best_value) or (metric_type != "cost" and value > best_value):
                best_value = value
                best_conditions = conditions

    print(f"Best {metric_type} value: {best_value}")
    print("Best conditions:")
    for k, v in best_conditions.items():
        print(f"  {k}: {v}")


    


