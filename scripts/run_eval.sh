#!/bin/bash

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

# OptAgent Benchmark Evaluation Script
# Usage: ./scripts/run_eval.sh [OPTIONS] BENCHMARK_FILE
#
# Note: This script supports the simplified 3-node architecture:
# - modeler: Creates mathematical model and Python code
# - verifier: Validates model, code, execution and results comprehensively
# - corrector: Fixes issues based on verification feedback

set -e  # Exit on error

# Default values
BENCHMARK_FILE="benchmark/rail-data/normal/normal-test.jsonl"
OUTPUT_DIR="eval_results"
CONCURRENCY=50
PASS_N=1
TOLERANCE=1e-1
TIMEOUT=1200
MAX_CORRECTIONS=5
DEBUG=false

# Read model name from conf.yaml
read_model_from_conf() {
    local conf_file="conf.yaml"
    if [[ -f "$conf_file" ]]; then
        # Extract model name from conf.yaml using grep and sed
        local model_name=$(grep -A 2 "BASIC_MODEL:" "$conf_file" | grep "model:" | sed 's/.*model:[[:space:]]*"\([^"]*\)".*/\1/')
        if [[ -n "$model_name" ]]; then
            echo "$model_name"
        else
            echo "DeepSeek-V3.1"  # fallback default
        fi
    else
        echo "DeepSeek-V3.1"  # fallback default
    fi
}

MODEL_NAME=$(read_model_from_conf)

# 超时配置变量
ENABLE_BATCH_MODE=false
LLM_REQUEST_TIMEOUT=""
CODE_EXECUTION_TIMEOUT=""
ENABLE_VISUALIZATION=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    cat << EOF
OptAgent Benchmark Evaluation Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -o, --output-dir DIR    Output directory for results (default: eval_results)
    -c, --concurrency NUM   Number of concurrent evaluations (default: 32)
    -n, --pass-n NUM        Number of attempts per problem for pass@n (default: 1)
    -t, --tolerance FLOAT   Relative error tolerance for correctness (default: 1e-6)
    -T, --timeout SEC       Timeout per problem in seconds (default: 1200)
    -m, --model-name NAME   Name of the model being evaluated
    -b, --benchmark FILE    Benchmark file to evaluate (default: benchmark/test.jsonl)
    --max-corrections NUM   Maximum correction attempts (default: 5)
    --debug                 Enable debug logging
    --enable-batch-mode     Enable batch evaluation mode with optimized timeouts
    --llm-request-timeout SEC       LLM request timeout in seconds (default: 120)
    --code-execution-timeout SEC    Code execution timeout in seconds (default: 60)
    --enable-visualization  Enable visualization generation after successful modeling
    -h, --help              Show this help message

BENCHMARK FILE:
    You can specify the benchmark file using -b/--benchmark option or as a positional argument.

EXAMPLES:
    # Basic evaluation
    $0

    # High concurrency evaluation with pass@3
    $0 -c 8 -n 3 -m "gpt-4"

    # Debug mode with custom output directory
    $0 --debug -o my_results

    # Custom tolerance and timeout
    $0 -t 1e-4 -T 600
    
    # Custom timeouts for the 3-node architecture
    $0 --llm-request-timeout 180 --code-execution-timeout 90
    
    # Batch mode with optimized timeouts
    $0 --enable-batch-mode --max-corrections 3
    
    # Enable visualization for better result understanding
    $0 --enable-visualization

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -n|--pass-n)
            PASS_N="$2"
            shift 2
            ;;
        -t|--tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        -T|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -m|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max-corrections)
            MAX_CORRECTIONS="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --enable-batch-mode)
            ENABLE_BATCH_MODE=true
            shift
            ;;
        --llm-request-timeout)
            LLM_REQUEST_TIMEOUT="$2"
            shift 2
            ;;
        --code-execution-timeout)
            CODE_EXECUTION_TIMEOUT="$2"
            shift 2
            ;;
        --enable-visualization)
            ENABLE_VISUALIZATION=true
            shift
            ;;
        -b|--benchmark)
            BENCHMARK_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}" >&2
            echo "Use -h or --help for usage information."
            exit 1
            ;;
        *)
            # If it's not an option, treat it as benchmark file
            BENCHMARK_FILE="$1"
            shift
            ;;
    esac
done

# Check if benchmark file exists
if [[ ! -f "$BENCHMARK_FILE" ]]; then
    echo -e "${RED}Error: Benchmark file not found: $BENCHMARK_FILE${NC}" >&2
    exit 1
fi

# Validate numeric arguments
if ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}Error: Concurrency must be a positive integer${NC}" >&2
    exit 1
fi

if ! [[ "$PASS_N" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}Error: pass-n must be a positive integer${NC}" >&2
    exit 1
fi

if ! [[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}Error: Timeout must be a positive integer${NC}" >&2
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed or not in PATH${NC}" >&2
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# Count problems in benchmark file
PROBLEM_COUNT=$(grep -c "^{" "$BENCHMARK_FILE" 2>/dev/null || echo "0")
TOTAL_TASKS=$((PROBLEM_COUNT * PASS_N))

# 简化输出 - 只显示关键信息
echo "  Problems: $PROBLEM_COUNT"
echo "  Total Tasks: $TOTAL_TASKS (${PROBLEM_COUNT} × ${PASS_N})"
echo ""

# 自动确认执行 - 直接开始评测，无需用户交互
echo "Starting evaluation..."

# Build the command
CMD_ARGS=(
    "python" "scripts/run_eval.py"
    "$BENCHMARK_FILE"
    "--output_dir" "$OUTPUT_DIR"
    "--concurrency" "$CONCURRENCY"
    "--pass_n" "$PASS_N"
    "--tolerance" "$TOLERANCE"
    "--timeout" "$TIMEOUT"
    "--max_corrections" "$MAX_CORRECTIONS"
)

if [[ -n "$MODEL_NAME" ]]; then
    CMD_ARGS+=("--model_name" "$MODEL_NAME")
fi

if [[ "$DEBUG" == "true" ]]; then
    CMD_ARGS+=("--debug")
fi

# 添加批量模式标志
if [[ "$ENABLE_BATCH_MODE" == "true" ]]; then
    CMD_ARGS+=("--enable_batch_mode")
fi

# 添加可视化标志
if [[ "$ENABLE_VISUALIZATION" == "true" ]]; then
    CMD_ARGS+=("--enable_visualization")
fi

# 添加超时配置参数
if [[ -n "$LLM_REQUEST_TIMEOUT" ]]; then
    CMD_ARGS+=("--llm_timeout" "$LLM_REQUEST_TIMEOUT")
fi

if [[ -n "$CODE_EXECUTION_TIMEOUT" ]]; then
    CMD_ARGS+=("--code_timeout" "$CODE_EXECUTION_TIMEOUT")
fi

# Record start time
START_TIME=$(date +%s)

# Create hierarchical directory structure for log file
BENCHMARK_NAME=$(basename "$BENCHMARK_FILE" | sed 's/\.[^.]*$//')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TASK_DIR="$OUTPUT_DIR/$BENCHMARK_NAME/$MODEL_NAME/$TIMESTAMP"
mkdir -p "$TASK_DIR"

# Create log file in the specific task directory
LOG_FILE="$TASK_DIR/evaluation.log"

# Function to log messages to both console and file
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Log evaluation start
log_message "================================================================================"
log_message "                        OptAgent Evaluation Started"
log_message "================================================================================"
log_message "Start Time: $(date)"
log_message "Benchmark File: $BENCHMARK_FILE"
log_message "Output Directory: $OUTPUT_DIR"
log_message "Model Name: $MODEL_NAME"
log_message "Concurrency: $CONCURRENCY"
log_message "Pass N: $PASS_N"
log_message "Tolerance: $TOLERANCE"
log_message "Timeout: $TIMEOUT"
log_message "Max Corrections: $MAX_CORRECTIONS"
log_message "Debug Mode: $DEBUG"
log_message "Batch Mode: $ENABLE_BATCH_MODE"
log_message "Visualization Enabled: $ENABLE_VISUALIZATION"
if [[ -n "$LLM_REQUEST_TIMEOUT" ]]; then
    log_message "LLM Request Timeout: $LLM_REQUEST_TIMEOUT"
fi
if [[ -n "$CODE_EXECUTION_TIMEOUT" ]]; then
    log_message "Code Execution Timeout: $CODE_EXECUTION_TIMEOUT"
fi
log_message "Log File: $LOG_FILE"
log_message "================================================================================"
log_message ""

# Run the evaluation with uv and capture all output
# Pass TASK_DIR as environment variable to Python script
export TASK_DIR="$TASK_DIR"
if uv run "${CMD_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    # Calculate elapsed time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    log_message ""
    log_message "================================================================================"
    log_message "                        Evaluation Completed Successfully!"
    log_message "================================================================================"
    log_message ""
    log_message "⏱️  Total Evaluation Time: $(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)"
    log_message "🚀 Concurrent Workers Used: ${CONCURRENCY}"
    log_message "📂 Results saved to: $OUTPUT_DIR"
    log_message "📄 Log file: $TASK_DIR/evaluation.log"
    
    # Show specific trace directory if it exists - now with hierarchical structure
    if [ -d "$OUTPUT_DIR" ]; then
        # Extract benchmark name from file path (remove extension and path)
        BENCHMARK_NAME=$(basename "$BENCHMARK_FILE" | sed 's/\.[^.]*$//')
        
        # Look for the most recent evaluation run in the hierarchical structure
        BENCHMARK_DIR="$OUTPUT_DIR/$BENCHMARK_NAME"
        if [ -d "$BENCHMARK_DIR" ]; then
            MODEL_DIR="$BENCHMARK_DIR/$MODEL_NAME"
            if [ -d "$MODEL_DIR" ]; then
                LATEST_DIR=$(ls -dt "$MODEL_DIR"/*/ 2>/dev/null | head -1)
                if [ -n "$LATEST_DIR" ]; then
                    log_message "📁 Hierarchical Results Structure:"
                    log_message "   📂 $OUTPUT_DIR/"
                    log_message "   └── 📂 $BENCHMARK_NAME/"
                    log_message "       └── 📂 $MODEL_NAME/"
                    log_message "           └── 📂 $(basename "$LATEST_DIR")/"
                    log_message ""
                    log_message "Detailed traces and logs saved to: $LATEST_DIR"
                    log_message "📄 Evaluation log: $TASK_DIR/evaluation.log"
                    log_message "Available files:"
                    ls -la "$LATEST_DIR" 2>/dev/null | grep -E "\.(json|jsonl)$" | awk -v dir="$LATEST_DIR" '{print "  - " dir $9}' | tee -a "$LOG_FILE"
                fi
            fi
        fi
    fi
    log_message ""
else
    log_message ""
    log_message "================================================================================"
    log_message "                           Evaluation Failed!"
    log_message "================================================================================"
    log_message ""
    log_message "Check the logs above for error details."
    log_message "Log file: $TASK_DIR/evaluation.log"
    exit 1
fi 