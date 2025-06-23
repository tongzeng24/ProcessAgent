# LLM-Guided Chemical Process Optimization
**A Multi-Agent Framework for Autonomous Process Constraint Generation and Optimization**

## Overview

This project presents a novel approach to chemical process optimization using large language models (LLMs) integrated within a multi-agent architecture. Each agent is assigned a specialized role—such as constraint generation, parameter suggestion, simulation, and validation—to collaboratively explore and optimize steady-state process conditions. The system is built on top of IDAES for high-fidelity process modeling.

## Features

- **Chemical Process Simulation**: Built on IDAES-PSE for robust process modeling
- **AI-Driven Optimization**: LLM-powered agents analyze and optimize process parameters
- **Multi-Agent Collaboration**: AutoGen framework enables collaborative problem-solving

## Prerequisites

- Python 3.11
- Conda package manager
- Access to OpenAI API

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tongzeng24/ProcessAgent.git
cd ProcessAgent
```

### 2. Create and Activate Environment

```bash
conda create --yes --name ProcessAgent python=3.11
conda activate ProcessAgent
```

### 3. Install All Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install IDAES Extensions

```bash
idaes get-extensions --extra petsc
```

**Or install packages individually**:
```bash
pip install idaes-pse==2.8.0
pip install autogen-agentchat==0.5.1
pip install autogen-core==0.5.1
pip install autogen-ext==0.5.1
pip install openai==1.70.0
pip install tiktoken==0.9.0
pip install pandas==2.2.3
pip install pyyaml==6.0.2
# Then run: idaes get-extensions --extra petsc
```

## Quick Start

1. **Activate the environment**:
   ```bash
   conda activate ProcessAgent
   ```

4. **Configure your LLM API keys**:
   
   **Set your environment variable**:
   ```bash
   export OPENAI_API_KEY="your-actual-api-key-here"
   ```
   
   **Then update the config.yaml file**:
   ```bash
   # Edit config.yaml and add your API key to the Model section:
   # api_key: "your-actual-api-key-here"
   ```
   
   **Note**: Make sure to add your actual OpenAI API key in the `api_key` field under the `Model` section in config.yaml before running the application.

## Usage 

### Running the Complete Pipeline

To run the entire LLM-guided chemical process optimization pipeline:

```bash
python main.py
```

### Configuration

All system settings can be customized by editing the `config.yaml` file:

```yaml
# Example configuration sections:
ContextAgent:
  context_sampling_iterations: 5
  
Optimization:
  optimization_metric: "cost"  # Options: "cost", "yield", "yield/cost"
  initial_params: {
    "H101_temperature": 600,
    "F101_temperature": 325,
    "F102_temperature": 375,
    "F102_deltaP": -240000
  }
...
```

**Key Configuration Options:**
- **Context Agent**: Adjust sampling iterations and output paths
- **Optimization**: Set objective function and initial process parameters  
- **Model**: Configure LLM model and API settings

Results will be saved to the `Results/` directory as specified in your configuration.
