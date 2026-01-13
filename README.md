# OptAgent

[![Python 3.8+](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OptAgent** is an advanced AI-powered optimization agent framework built on **langgraph** architecture that specializes in solving mathematical optimization problems. It combines the power of large language models with systematic verification and correction mechanisms to generate reliable optimization solutions.

## 🚀 Quick Start

### Prerequisites

Ensure your system meets the following requirements:

- **[Python](https://www.python.org/downloads/):** Version `3.8+`
- **[uv](https://docs.astral.sh/uv/getting-started/installation/):** For simplified Python environment and dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/AuroraLHL/OptAgent
cd OptAgent

# Install dependencies - uv will handle Python interpreter and virtual environment creation
uv sync

# Configure your LLM API keys in conf.yaml
cp conf.yaml.example conf.yaml

# Edit conf.yaml with your API credentials
# See configuration section below for details
```

### Configuration

#### Basic LLM Configuration

Edit `conf.yaml` to configure your language model:

```yaml
BASIC_MODEL:
  base_url: https://ark.cn-beijing.volces.com/api/v3
  model: "doubao-1-5-pro-32k-250115"
  api_key: your_api_key_here
  # max_retries: 3
  # verify_ssl: false  # For self-signed certificates

# Optional: Use a reasoning model for advanced planning
# REASONING_MODEL:
#   base_url: https://ark.cn-beijing.volces.com/api/v3
#   model: "doubao-1-5-thinking-pro-m-250428"
#   api_key: your_api_key_here
```

> [!NOTE]
> Please read the configuration guide carefully and update the settings to match your specific LLM provider and requirements. A restart is required every time you change the `conf.yaml` file.

## 🏗️ OptAgent Architecture

OptAgent is built upon the **deer-flow** architecture, implementing a sophisticated three-node workflow designed specifically for optimization problem solving:

```
Problem Input → Modeler → Verifier → Corrector
                   ↑         ↓         ↓
                   └─────────┴─────────┘
                   (Correction Loop)
```

### Deer-Flow Foundation

This project leverages the deer-flow framework to provide:

- **Intelligent Workflow Orchestration**: Advanced state management and decision routing
- **Adaptive Problem Solving**: Dynamic strategy selection based on problem characteristics
- **Scalable Multi-Agent Coordination**: Efficient resource allocation and parallel processing
- **Robust Error Recovery**: Self-healing mechanisms with fallback strategies

### Core Components

1. **Modeler Node**: Creates mathematical models and Python implementation code

   - Analyzes natural language problem descriptions
   - Generates mathematical formulations
   - Produces executable Python code using optimization libraries
2. **Verifier Node**: Validates models, code execution, and results comprehensively

   - Checks mathematical model correctness
   - Validates Python code syntax and logic
   - Verifies solution feasibility and optimality
   - Provides detailed feedback for corrections
3. **Corrector Node**: Fixes issues based on verification feedback

   - Addresses model formulation errors
   - Corrects code implementation issues
   - Improves solution quality based on verifier insights
4. **Reporter Node**: Generates final comprehensive reports

   - Summarizes the optimization problem and solution
   - Provides mathematical analysis and code documentation
   - Formats results for easy understanding

### Workflow Features

- **Automatic Loop Detection**: Prevents infinite correction cycles
- **Configurable Correction Limits**: Control maximum retry attempts
- **Comprehensive Error Handling**: Robust error detection and recovery
- **State Persistence**: Resume workflows using LangGraph checkpointing

## 📊 Benchmark Evaluation

OptAgent includes comprehensive evaluation tools for testing against optimization benchmarks.

### Running Evaluations

#### Single Benchmark Evaluation

```bash
# Basic evaluation using the default benchmark
uv run python scripts/run_eval.py

# Custom benchmark file
uv run python scripts/run_eval.py benchmark/your_benchmark.jsonl

# Using the shell script with custom parameters
./scripts/run_eval.sh -b benchmark/MAMO_ComplexLP.jsonl -c 10 -t 1800
```

#### Evaluation Script Parameters

The `run_eval.sh` script supports the following parameters:

```bash
# Core options
-b, --benchmark     Benchmark file path (default: benchmark/rail-data/test/problem.jsonl)
-o, --output        Output directory (default: eval_results)
-c, --concurrency   Concurrent jobs (default: 50)
-t, --timeout       Timeout in seconds (default: 1200)
-m, --max-corrections  Maximum correction attempts (default: 5)

# Performance options
--pass-n           Number of solution attempts per problem (default: 1)
--tolerance        Solution tolerance (default: 1e-1)
--batch-mode       Enable batch processing mode

# Debugging
--debug            Enable debug logging
--help             Show help message
```

#### Example Evaluation Commands

```bash
# Evaluate complex linear programming problems
./scripts/run_eval.sh -b benchmark/MAMO_ComplexLP.jsonl -c 20 -t 3600

# Quick test with debug output
./scripts/run_eval.sh -b benchmark/test_simple_data.jsonl --debug -c 5

# High-accuracy evaluation with multiple attempts
./scripts/run_eval.sh -b benchmark/OptMATH_Bench.jsonl --pass-n 3 --tolerance 1e-3
```

### Benchmark Results

Evaluation results are stored in the `eval_results/` directory with the following structure:

```
eval_results/
├── benchmark_name/
│   └── model_name/
│       └── timestamp/
│           ├── config.json       # Evaluation configuration
│           ├── results.csv       # Summary results
│           ├── traces.jsonl      # Detailed execution traces
│           ├── failed_cases.jsonl # Failed problem cases
│           └── summary.json      # Evaluation summary
```

### Supported Benchmarks

OptAgent supports evaluation on various optimization benchmark datasets:

- **MAMO_ComplexLP**: Complex linear programming problems
- **MAMO_EasyLP**: Basic linear programming problems
- **NL4OPT**: Natural language optimization problems
- **OptMATH_Bench**: Mathematical optimization benchmark
- **IndustryOR**: Industrial operations research problems

## 🛠️ Development

### Running Tests

```bash
# Run all tests
make test

# Run specific test files
pytest tests/integration/test_workflow.py

# Run with coverage
make coverage
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format
```

### Using LangGraph Studio for Debugging

OptAgent is built with LangGraph and can be debugged using LangGraph Studio:

```bash
# Install LangGraph CLI (if not already installed)
pip install -U "langgraph-cli[inmem]"

# Start LangGraph development server
langgraph dev

# Or with UV
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev
```

After starting the server, access the Studio UI at the provided URL to visualize and debug the workflow execution.

## ✨ Key Features

### Deer-Flow Powered Intelligence

- **Adaptive Learning**: Continuously improves optimization strategies through experience
- **Cross-Domain Transfer**: Leverages knowledge from different optimization domains
- **Hierarchical Planning**: Breaks down complex problems into manageable sub-problems
- **Dynamic Resource Management**: Optimally allocates computational resources

### Advanced Optimization Capabilities

- **Multi-Objective Optimization**: Handles conflicting objectives with Pareto front analysis
- **Real-Time Adaptation**: Adjusts strategies based on intermediate results
- **Robustness Guarantees**: Provides bounds on solution quality and feasibility
- **Explainable AI**: Generates detailed explanations for optimization decisions

## 🚀 Usage Examples

### Web Interface (Recommended)

OptAgent provides a modern web interface for easy interaction:

```bash
# Start the backend server (Terminal 1)
python server.py

# Start the frontend interface (Terminal 2)
python start_frontend.py
```

Then open your browser and navigate to `http://localhost:3000`

### Console Usage

```bash
# Basic optimization problem solving
uv run main.py "Minimize 2x + 3y subject to x + y >= 5, x >= 0, y >= 0"

# Interactive mode
uv run main.py --interactive
```

### API Usage

```python
from src.graph.builder import build_optag_graph

# Create the OptAgent workflow
graph = build_optag_graph()

# Define your optimization problem
initial_state = {
    "problem_statement": "Minimize cost while maximizing efficiency...",
    "max_corrections": 5
}

# Execute the workflow
result = await graph.ainvoke(initial_state)
```

## 📚 Examples

Check the `examples/` directory for detailed usage examples and sample optimization problems.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING](./CONTRIBUTING) for guidelines.

## 📜 License

This project is open source under the [MIT License](./LICENSE).

## 🙏 Acknowledgments

OptAgent is built upon the excellent work of the open source community:

- **[Deer-Flow](https://github.com/deerflow)**: For the advanced multi-agent workflow architecture that forms the foundation of this project
- **[LangChain](https://github.com/langchain-ai/langchain)**: For LLM interactions and tool integration
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: For multi-agent workflow orchestration
- **[Gurobi](https://www.gurobi.com/)**: For industry-leading optimization solver technology
- **[PuLP](https://coin-or.github.io/pulp/)**: For accessible linear programming capabilities

We are grateful to stand on the shoulders of these giants, especially the deer-flow community for providing the architectural framework that makes OptAgent possible.
