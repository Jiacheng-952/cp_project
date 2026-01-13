# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import sys
import io
import traceback
import re
import multiprocessing as mp
import time


def _executor_worker(code_str: str, out_q: mp.Queue):
    """Top-level worker function for Windows spawn compatibility."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    exec_namespace = {
        "__builtins__": __builtins__,
        "print": print,
        "len": len,
        "range": range,
        "sum": sum,
        "max": max,
        "min": min,
        "abs": abs,
    }
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code_str, exec_namespace)
        out_q.put(
            {
                "ok": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            }
        )
    except Exception as e:  # noqa: BLE001
        import traceback as _tb

        out_q.put(
            {
                "ok": False,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue() + "\n" + _tb.format_exc(),
                "error": str(e),
            }
        )


from typing import Annotated, Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
from langchain_core.tools import tool
from .decorators import log_io
from .python_repl import python_repl_tool  # Import the synchronous tool
from src.utils.robust_optimal_value_extraction import extract_optimal_value

logger = logging.getLogger(__name__)


@tool
@log_io
def execution_evaluator_tool(
    code: Annotated[str, "The Python code to execute and evaluate."],
    timeout_seconds: Annotated[int, "Maximum execution time in seconds."] = 30,
) -> Dict[str, Any]:
    """
    Execute Python code in a controlled environment and evaluate the results.

    This tool safely executes optimization code and captures:
    - Standard output and errors
    - Execution success/failure status
    - Runtime errors with details
    - Optimization results (objective values, solution status)

    Args:
        code: Python code string to execute
        timeout_seconds: Maximum execution time (default: 30 seconds) - Note: timeout is for compatibility, actual execution is not timed here.

    Returns:
        Dict containing key execution metrics.
    """
    if not isinstance(code, str) or not code.strip():
        error_msg = "Invalid or empty code provided"
        logger.error(error_msg)
        return {
            "status": "ERROR",
            "executed": False,
            "stdout": "",
            "stderr": "",
            "error_message": error_msg,
            "execution_time": 0,
            "optimal_value": None,
            "solution_status": None,
        }

    logger.info(f"Executing Python code via python_repl_tool (timeout: {timeout_seconds}s)")

    start_time = time.time()

    try:
        # Delegate execution to the safe, synchronous python_repl_tool with timeout
        result_str = python_repl_tool.invoke({"code": code, "timeout_seconds": timeout_seconds})

        execution_time = time.time() - start_time

        # Parse the result string from python_repl_tool
        stdout_content = ""
        error_content = ""

        if "Successfully executed:" in result_str:
            match = re.search(r"Stdout:\s*(.*)", result_str, re.DOTALL)
            if match:
                stdout_content = match.group(1).strip()
            status = "PASS"
            executed = True
        else:  # Assumes "Error executing code:"
            match = re.search(r"Error:\s*(.*)", result_str, re.DOTALL)
            if match:
                error_content = match.group(1).strip()
            status = "FAIL"
            executed = False

        # Extract optimization results from the stdout
        optimal_value = extract_optimal_value(stdout_content)
        solution_status = _extract_solution_status(stdout_content)

        logger.info(f"Code execution finished (took {execution_time:.2f}s)")

        return {
            "status": status,
            "executed": executed,
            "stdout": stdout_content,
            "stderr": error_content,
            "error_message": error_content if error_content else None,
            "execution_time": execution_time,
            "optimal_value": optimal_value,
            "solution_status": solution_status,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Critical error in execution_evaluator_tool: {str(e)}"
        traceback_str = traceback.format_exc()

        logger.error(f"{error_msg}\n{traceback_str}")

        return {
            "status": "ERROR",
            "executed": False,
            "stdout": "",
            "stderr": traceback_str,
            "error_message": error_msg,
            "execution_time": execution_time,
            "optimal_value": None,
            "solution_status": None,
        }


def _extract_optimal_value(output: str) -> Optional[float]:
    """
    Extract optimal value from optimization output.

    Looks for common patterns like:
    - optimal_value = 123.45
    - obj_val = 123.45
    - objective = 123.45
    - result = 123.45
    """
    if not output:
        return None

    # Common patterns for optimization results
    patterns = [
        r"optimal_value\s*=\s*([-+]?\d*\.?\d+)",
        r"obj_val\s*=\s*([-+]?\d*\.?\d+)",
        r"objective\s*=\s*([-+]?\d*\.?\d+)",
        r"result\s*=\s*([-+]?\d*\.?\d+)",
        r"answer\s*=\s*([-+]?\d*\.?\d+)",
        # OR-Tools CP-SAT specific patterns
        r"Optimal objective\s*([-+]?\d*\.?\d+)",
        r"Objective value:\s*([-+]?\d*\.?\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def _extract_solution_status(output: str) -> Optional[str]:
    """
    Extract solution status from optimization output.

    Looks for optimization solver status indicators.
    """
    if not output:
        return None

    output_lower = output.lower()

    # Check for common status indicators
    if "optimal" in output_lower:
        return "OPTIMAL"
    elif "infeasible" in output_lower:
        return "INFEASIBLE"
    elif "unbounded" in output_lower:
        return "UNBOUNDED"
    elif "time limit" in output_lower or "timeout" in output_lower:
        return "TIME_LIMIT"
    elif "error" in output_lower or "failed" in output_lower:
        return "ERROR"
    else:
        return "UNKNOWN"


def execute_and_evaluate(code: str, timeout_seconds: int = 30) -> Dict[str, Any]:
    """
    Direct function interface for code execution (without tool decorator).

    Useful for internal usage without LangChain tool overhead.
    """
    return execution_evaluator_tool.func(code, timeout_seconds)
