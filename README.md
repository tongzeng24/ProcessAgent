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

### 1. Create and Activate Environment

```bash
conda create --yes --name ProcessAgent python=3.11
conda activate ProcessAgent
```

### 2. Install IDAES Process Simulation Framework

```bash
conda install --yes -c conda-forge idaes-pse
idaes get-extensions --extra petsc
```

### 3. Install AutoGen Multi-Agent Framework

```bash
pip install "autogen-ext"
pip install -U "autogen-agentchat"
pip install "autogen-ext[openai]"
pip install "autogen-core"
```

### 4. Install Additional Dependencies

```bash
pip install tiktoken
pip install pyyaml pandas
```

## Quick Start

 **Configure your LLM API keys**:
 ```bash
 export OPENAI_API_KEY="your-api-key-here"
 ```
 **Then update the config.yaml file**:


## Key Dependencies

| Package | Purpose |
|---------|---------|
| `idaes-pse` | Chemical process simulation and modeling |
| `autogen-*` | Multi-agent AI framework for collaborative optimization |
| `tiktoken` | Token counting for LLM interactions |
| `pyyaml` | Configuration file management |
| `pandas` | Data processing and analysis |


