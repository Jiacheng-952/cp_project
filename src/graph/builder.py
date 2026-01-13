# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
OptAgent Graph Builder Module

This module provides functions for building and configuring the
OptAgent workflow state graph with 3 nodes: modeler, verifier, and corrector.
"""

import logging
import os
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .types import OptAgentState
from .main_nodes import (
    modeler_node,
    verifier_node,
    corrector_node,
    reporter_node,
    visualizer_node,
)
from .csp_cop_classifier_node import csp_cop_classifier_node

logger = logging.getLogger(__name__)


def problem_input_node(state: OptAgentState, config) -> Dict[str, Any]:
    """
    Extract problem statement from messages or existing state.
    """
    logger.info("Processing problem input for OptAgent")

    problem_statement = state.get("problem_statement", "")

    if not problem_statement:
        messages = state.get("messages", [])
        if messages:
            first_message = messages[0]
            if hasattr(first_message, "content"):
                problem_statement = first_message.content
            elif isinstance(first_message, dict):
                problem_statement = first_message.get("content", "")

    return {"problem_statement": problem_statement}


# ===== ROUTING CONDITION FUNCTIONS =====


def problem_input_condition(state: OptAgentState) -> str:
    """
    Problem input routing condition for workflow initialization.

    Args:
        state: Current OptAgent state

    Returns:
        Next node name: "csp_cop_classifier" or "reporter"
    """
    problem_statement = state.get("problem_statement", "")

    if problem_statement:
        return "csp_cop_classifier"
    else:
        # Try to extract problem from messages
        messages = state.get("messages", [])
        if messages:
            return "csp_cop_classifier"
        else:
            return "reporter"


def verifier_condition(state: OptAgentState) -> str:
    """
    Verifier routing condition for determining next step.

    Args:
        state: Current OptAgent state

    Returns:
        Next node name: "visualizer", "reporter" or "corrector"
    """
    verification_passed = state.get("verification_passed", False)

    if verification_passed:
        enable_visualization = state.get("enable_visualization", False)
        if enable_visualization:
            return "visualizer"
        else:
            return "reporter"
    else:
        correction_count = state.get("correction_count", 0)
        max_corrections = state.get("max_corrections", 3)

        if correction_count < max_corrections:
            return "corrector"
        else:
            return "reporter"


def classifier_condition(state: OptAgentState) -> str:
    """
    Classifier routing condition - always goes to modeler with appropriate template.

    Args:
        state: Current OptAgent state

    Returns:
        Next node name: "modeler"
    """
    return "modeler"


def corrector_condition(state: OptAgentState) -> str:
    """
    Corrector routing condition - always goes back to verifier.

    Args:
        state: Current OptAgent state

    Returns:
        Next node name: "verifier"
    """
    return "verifier"


# ===== GRAPH CONSTRUCTION FUNCTIONS =====


def _build_optag_graph() -> StateGraph:
    """
    Internal function to build the OptAgent workflow state graph.

    Constructs the enhanced OptAgent workflow graph with CSP/COP classifier, modeler, 
    verifier, and corrector nodes, plus routing logic.

    Returns:
        StateGraph: Uncompiled LangGraph state graph ready for compilation
    """
    builder = StateGraph(OptAgentState)

    # ===== Add all nodes =====
    builder.add_node("problem_input", problem_input_node)
    builder.add_node("csp_cop_classifier", csp_cop_classifier_node)
    builder.add_node("modeler", modeler_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("corrector", corrector_node)
    builder.add_node("visualizer", visualizer_node)
    builder.add_node("reporter", reporter_node)

    # ===== Define graph flow =====

    # Entry point
    builder.add_edge(START, "problem_input")

    # Problem input routing
    builder.add_conditional_edges(
        "problem_input", problem_input_condition, ["csp_cop_classifier", "reporter"]
    )
    
    # CSP/COP classifier goes to modeler
    builder.add_edge("csp_cop_classifier", "modeler")

    # Modeler always goes to verifier
    builder.add_edge("modeler", "verifier")

    # Verifier routing - to visualizer (success + viz enabled), reporter (success), or corrector (failure)
    builder.add_conditional_edges(
        "verifier", verifier_condition, ["visualizer", "reporter", "corrector"]
    )

    # Corrector always goes back to verifier
    builder.add_edge("corrector", "verifier")

    # Visualizer always goes to reporter
    builder.add_edge("visualizer", "reporter")

    # Final output
    builder.add_edge("reporter", END)

    return builder


# ===== PUBLIC API FUNCTIONS =====


def build_optag_graph():
    """
    Build and compile the OptAgent workflow graph without memory persistence.

    This is the primary function for creating OptAgent workflow graphs
    for stateless execution environments such as API servers.

    Returns:
        Compiled LangGraph instance ready for execution

    Example:
        >>> graph = build_optag_graph()
        >>> result = await graph.ainvoke(initial_state)
    """
    builder = _build_optag_graph()
    
    # LangSmith tracing is handled globally via environment variables
    # The tracing_v2_enabled context manager in workflow.py handles the setup
    return builder.compile()


def build_optag_graph_with_memory(use_checkpointer: bool = True):
    """
    Build and compile the OptAgent workflow graph with optional memory persistence.

    This function creates OptAgent workflow graphs with optional state
    persistence using LangGraph's checkpointer functionality for resumable workflows.

    Args:
        use_checkpointer: Whether to enable memory persistence using MemorySaver (default: True)

    Returns:
        Compiled LangGraph instance with optional memory persistence

    Example:
        >>> graph = build_optag_graph_with_memory(use_checkpointer=True)
        >>> result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": "session-1"}})
    """
    builder = _build_optag_graph()
    if use_checkpointer:
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)
    return builder.compile()
