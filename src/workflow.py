# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
OptAgent Workflow Module

This module provides the main entry points for running OptAgent optimization
workflows. It includes both synchronous and asynchronous interfaces for
solving optimization problems using the three-phase verification system.
"""

import asyncio
import logging
import os
import sys
from functools import lru_cache
from typing import Dict, Any, Optional, Callable

from dotenv import load_dotenv

from src.config.timeout_config import TimeoutConfig
from src.graph.builder import build_optag_graph, build_optag_graph_with_memory
from src.graph.types import create_optag_state

# Load environment variables
load_dotenv()

# Add project root to Python path if not already present
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# Initialize LangSmith tracing if enabled
if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    try:
        from langsmith import Client
        from langchain_core.tracers.context import tracing_v2_enabled
        client = Client()
        # Enable tracing globally
        tracing_v2_enabled(True)
        logger.info("LangSmith tracing enabled")
    except ImportError:
        logger.warning("LangSmith not available, tracing disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith tracing: {e}")


# ===== LOGGING CONFIGURATION =====


def enable_debug_logging() -> None:
    """
    Enable debug level logging for more detailed execution information.

    This function sets the logging level for the 'src' package to DEBUG,
    providing more verbose output during OptAgent execution.
    """
    logging.getLogger("src").setLevel(logging.DEBUG)


def setup_logging(debug: bool = False) -> None:
    """可由上层入口调用的日志初始化。默认不抢占宿主配置。"""
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        root.setLevel(level)


@lru_cache(maxsize=1)
def _get_cached_graph():
    """Return cached OptAgent graph to reduce rebuild overhead."""
    return build_optag_graph()


# ===== MAIN WORKFLOW FUNCTIONS =====


def _default_render_progress(state: Dict[str, Any], step_count: int) -> None:
    verification_status = "✅" if state.get("verification_passed", False) else "⏳"
    correction_count = state.get("correction_count", 0)
    print(
        f"Step {step_count}: Verification: {verification_status} | Corrections: {correction_count}"
    )
    if state.get("verification_result") and "APPROVED" in state.get(
        "verification_result", ""
    ):
        print("  ✅ Comprehensive verification: APPROVED")
    elif state.get("verification_result") and "REJECTED" in state.get(
        "verification_result", ""
    ):
        print("  🔍 Comprehensive verification: REJECTED - correction needed")


def _default_render_final(final_state: Dict[str, Any]) -> None:
    print("\n🎯 OptAgent Workflow Results:")
    print(
        f"  ✅ Verification Passed: {'✅ YES' if final_state.get('verification_passed') else '❌ NO'}"
    )
    print(
        f"  🏆 Solution Complete: {'✅ YES' if final_state.get('solution_complete') else '❌ NO'}"
    )
    print(f"  🔄 Correction Count: {final_state.get('correction_count', 0)}")
    if final_state.get("optimal_value_extracted"):
        optimal_val = final_state.get("optimal_value", "Unknown")
        print(f"  💎 Optimal Value: {optimal_val}")
    if final_state.get("verification_passed") and final_state.get("solution_complete"):
        print("  🎉 SUCCESS: Complete verified solution generated!")
    elif final_state.get("solution_complete"):
        print("  ⚠️ PARTIAL: Solution generated but verification issues")
    else:
        print("  ❌ INCOMPLETE: Workflow did not complete successfully")
    final_solution = final_state.get("final_solution", {})
    if final_solution and final_solution.get("final_report"):
        print("\n" + "=" * 50)
        print("📊 FINAL REPORT")
        print("=" * 50)
        print(final_solution["final_report"])
        print("=" * 50)


async def run_optag_workflow_async(
    problem_statement: str,
    debug_mode: bool = False,
    max_corrections: int = 5,
    stream_output: bool = True,
    initial_state: Optional[Dict[str, Any]] = None,
    timeout_config: Optional[TimeoutConfig] = None,
    thread_id: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any], int], None]] = None,
    final_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    failure_logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the OptAgent workflow asynchronously for optimization problem solving.

    This function orchestrates the complete 3-node OptAgent workflow:
    1. Modeler: Creates mathematical model and Python code
    2. Verifier: Validates model, code, execution and results
    3. Corrector: Fixes issues based on verification feedback

    Args:
        problem_statement: The optimization problem description in natural language
        debug_mode: Enable debug mode for detailed logging
        max_corrections: Maximum number of correction attempts (default: 5)
        stream_output: Whether to stream progress output during execution (default: True)
        initial_state: Optional initial state dictionary to override defaults
        timeout_config: Timeout configuration object for controlling execution limits

    Returns:
        Final state dictionary containing the verified model, code, and solution results

    Raises:
        TimeoutError: If workflow execution exceeds configured timeout limits
        Exception: Various exceptions may be raised during workflow execution
    """
    # Initialize timeout configuration
    if timeout_config is None:
        from src.config.timeout_config import get_timeout_config

        timeout_config = get_timeout_config()

    if debug_mode:
        enable_debug_logging()

    logger.info(
        f"Starting OptAgent workflow with problem: {problem_statement[:100]}..."
    )

    # Create initial state using OptAgent state factory
    if initial_state is None:
        initial_state = create_optag_state(
            messages=[{"role": "user", "content": problem_statement}],
            problem_statement=problem_statement,
            max_corrections=max_corrections,
            debug_mode=debug_mode,
            timeout_config=timeout_config.to_dict(),
        )
    else:
        # If initial state is provided, ensure key parameters are correctly set
        initial_state.update(
            {
                "max_corrections": max_corrections,
                "debug_mode": debug_mode,
                "timeout_config": timeout_config.to_dict(),
            }
        )

    # Configuration for OptAgent execution
    config = {
        "configurable": {
            "thread_id": thread_id or "optag-session",
            "timeout_config": timeout_config,
        },
        "recursion_limit": timeout_config.langgraph_recursion_limit,
    }

    if failure_logger:
        config["configurable"]["failure_logger"] = failure_logger

    # Build OptAgent graph (cached)
    optag_graph = _get_cached_graph()

    logger.info("OptAgent workflow started - processing optimization problem")

    # Stream execution with overall timeout budget
    final_state = None
    step_count = 0
    overall_timeout = int(getattr(timeout_config, "single_problem_timeout", 300) or 300)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + overall_timeout
    agen = optag_graph.astream(
        input=initial_state,
        config=config,
        stream_mode="values",
    )
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError("Workflow internal timeout")
        try:
            state = await asyncio.wait_for(agen.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            break

        final_state = state
        step_count += 1

        if stream_output:
            (progress_callback or _default_render_progress)(state, step_count)

    if stream_output and final_state:
        (final_callback or _default_render_final)(final_state)

    logger.info("OptAgent workflow completed")
    return final_state


def run_optimization_problem(problem_statement: str, **kwargs) -> Dict[str, Any]:
    """
    Synchronous convenience function to run optimization problem solving.

    This function provides a synchronous interface to the async OptAgent workflow.
    It's designed for use in scripts and interactive environments where async/await
    syntax is not convenient.

    Args:
        problem_statement: Natural language description of the optimization problem
        **kwargs: Additional arguments passed to run_optag_workflow_async (see that
                 function's documentation for available options)

    Returns:
        Dictionary containing the key results from the optimization workflow:
        - model_verified: Whether the mathematical model passed verification
        - code_verified: Whether the code passed all validation tiers
        - solution_complete: Whether the workflow completed successfully
        - verified_model: The final verified mathematical model
        - verified_code: The final verified Python code
        - verified_optimal_value: The extracted optimal value (if found)
        - optimal_value_extracted: Whether optimal value extraction succeeded
        - final_solution: Complete solution package
        - workflow_phase: Final workflow phase reached
        - correction_history: History of corrections made during execution

    Raises:
        Exception: Any exception raised during workflow execution
    """
    # Run the async workflow in a new event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        result_state = asyncio.run(
            run_optag_workflow_async(problem_statement, **kwargs)
        )
    else:
        result_state = loop.run_until_complete(
            run_optag_workflow_async(problem_statement, **kwargs)
        )

    # Extract and return key results
    return {
        "verification_passed": result_state.get("verification_passed", False),
        "solution_complete": result_state.get("solution_complete", False),
        "current_model": result_state.get("current_model", ""),
        "current_code": result_state.get("current_code", ""),
        "optimal_value": result_state.get("optimal_value", 0.0),
        "optimal_value_extracted": result_state.get("optimal_value_extracted", False),
        "final_solution": result_state.get("final_solution", ""),
        "correction_count": result_state.get("correction_count", 0),
        "correction_history": result_state.get("correction_history", []),
    }


# ===== GRAPH INSTANCES =====


def get_graph():
    return build_optag_graph()


def get_graph_with_memory(use_checkpointer: bool = True):
    return build_optag_graph_with_memory(use_checkpointer=use_checkpointer)


# 保持向后兼容的别名（惰性获取）
optag = get_graph()
optag_with_memory = get_graph_with_memory(use_checkpointer=False)
graph = optag


# ===== COMMAND LINE INTERFACE =====

if __name__ == "__main__":
    print("=== OptAgent Graph Structure ===")
    optag_graph = build_optag_graph()
    mermaid_diagram = optag_graph.get_graph(xray=True).draw_mermaid()
    print(mermaid_diagram)
