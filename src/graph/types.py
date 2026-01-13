# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
OptAgent State Type Definitions

This module defines the state types for the 3-node OptAgent
optimization workflow system: modeler, verifier, and corrector.
"""

from typing import Literal, Dict, Any, List
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import MessagesState


class OptAgentState(MessagesState):
    """
    State definition for the 3-node OptAgent optimization workflow.

    This class extends LangGraph's MessagesState to include fields necessary
    for the three-node process: modeler, verifier, and corrector.
    All fields are optional (NotRequired) to provide flexibility in state updates.

    Attributes:
        problem_statement: The optimization problem description

        # Current solution
        current_model: Current mathematical model text
        current_code: Current Python code implementation
        solution_version: Version number of the current solution

        # Verification results
        verification_result: Result from comprehensive verification
        verification_passed: Whether the solution passed verification
        optimal_value: Extracted optimal value if verification passed
        optimal_value_extracted: Whether optimal value extraction succeeded
        has_feasible_solution: Whether a feasible solution was found (especially important for CSP problems)
        variable_values: Dictionary of variable names and their values from the solution

        # Correction tracking
        correction_count: Number of correction attempts made
        max_corrections: Maximum allowed correction attempts

        # Final solution
        final_solution: Complete solution package when successful
        solution_complete: Whether the entire workflow completed successfully

        # Control and debug
        debug_mode: Whether debug mode is enabled
        correction_history: History of corrections made during workflow

        # Evaluation metadata
        problem_id: Problem identifier for evaluation
        current_attempt: Current attempt number
    """

    # ===== CORE FIELDS =====

    # Problem and workflow control
    problem_statement: NotRequired[str]

    # Current solution state
    current_model: NotRequired[str]
    current_code: NotRequired[str]
    solution_version: NotRequired[int]
    execution_result: NotRequired[str]

    # Verification state
    verification_result: NotRequired[str]
    verification_passed: NotRequired[bool]
    verification_failed: NotRequired[bool]
    optimal_value: NotRequired[float]
    optimal_value_extracted: NotRequired[bool]
    has_feasible_solution: NotRequired[bool]
    variable_values: NotRequired[Dict[str, Any]]

    # Correction tracking
    correction_count: NotRequired[int]
    max_corrections: NotRequired[int]
    correction_needed: NotRequired[bool]

    # Loop detection and prevention
    verification_history: NotRequired[List[str]]
    error_patterns: NotRequired[Dict[str, int]]

    # Final solution state
    final_solution: NotRequired[Dict[str, Any]]
    solution_complete: NotRequired[bool]

    # Control parameters
    debug_mode: NotRequired[bool]
    correction_history: NotRequired[List[str]]

    # Evaluation metadata
    problem_id: NotRequired[str]
    current_attempt: NotRequired[int]

    # Data files for problems with separated data
    data_files: NotRequired[List[str]]

    # Timeout configuration
    timeout_config: NotRequired[Dict[str, Any]]

    # Visualization control and results
    enable_visualization: NotRequired[bool]
    visualization_code: NotRequired[str]
    visualization_status: NotRequired[str]  # "success", "failed", "skipped"
    visualization_error: NotRequired[str]
    plot_files: NotRequired[List[str]]
    output_dir: NotRequired[str]

    # Problem classification and optimization tracking
    problem_type: NotRequired[str]  # "CSP", "COP", "UNKNOWN"
    csp_cop_classification: NotRequired[str]  # "CSP", "COP", "UNKNOWN"
    csp_cop_confidence: NotRequired[str]  # "HIGH", "MEDIUM", "LOW"
    csp_cop_reasoning: NotRequired[str]
    optimization_iterations: NotRequired[int]
    max_optimization_iterations: NotRequired[int]
    current_best_optimal_value: NotRequired[float]
    current_best_solution: NotRequired[Dict[str, Any]]


def create_optag_state(**kwargs) -> OptAgentState:
    """
    Create an OptAgentState with default values and validation.

    This factory function ensures all fields have proper default values
    and validates the state consistency. It provides a safe way to initialize
    the OptAgent state with sensible defaults while allowing
    customization through keyword arguments.

    Args:
        **kwargs: Optional field values to override defaults

    Returns:
        OptAgentState: Fully initialized state dictionary

    Raises:
        Warning: If unknown fields are provided (they will be ignored)
    """
    import time
    import logging

    logger = logging.getLogger(__name__)

    # Default values for all OptAgent state fields
    defaults = {
        # Core workflow fields
        "problem_statement": "",
        # Current solution
        "current_model": "",
        "current_code": "",
        "solution_version": 0,
        # Verification state
        "verification_result": "",
        "verification_passed": False,
        "verification_failed": False,
        "optimal_value": 0.0,
        "optimal_value_extracted": False,
        "has_feasible_solution": False,
        "variable_values": {},
        # Correction tracking
        "correction_count": 0,
        "max_corrections": 5,
        "correction_needed": False,
        # Loop detection and prevention
        "verification_history": [],
        "error_patterns": {},
        # Final solution
        "final_solution": {},
        "solution_complete": False,
        # Control parameters
        "debug_mode": False,
        "correction_history": [],
        # Evaluation metadata
        "problem_id": "unknown",
        "current_attempt": 0,
        # Data files for problems with separated data
        "data_files": [],
        # LangGraph MessagesState fields
        "messages": [],
        # Timeout configuration
        "timeout_config": {},
        # Visualization control and results
        "enable_visualization": False,
        "visualization_code": "",
        "visualization_status": "skipped",
        "visualization_error": "",
        "plot_files": [],
        "output_dir": "",
        # Problem classification and optimization tracking
        "problem_type": "UNKNOWN",
        "csp_cop_classification": "UNKNOWN",
        "csp_cop_confidence": "LOW",
        "csp_cop_reasoning": "",
        "optimization_iterations": 0,
        "max_optimization_iterations": 10,
        "current_best_optimal_value": 0.0,
        "current_best_solution": {},
        # State management metadata
        "state_created_at": time.time(),
        "state_version": 1,
    }

    # Safely update defaults with provided kwargs
    for key, value in kwargs.items():
        if key in defaults or key.startswith("_"):  # Allow private fields
            defaults[key] = value
        else:
            logger.warning(f"Unknown state field '{key}' ignored during state creation")

    # Validate state consistency if error handling module is available
    try:
        from src.utils.error_handling import SafeStateManager

        if not SafeStateManager.validate_state_consistency(defaults):
            logger.warning("State consistency validation failed during creation")
    except ImportError:
        # Gracefully handle if error_handling module is not available
        pass

    return defaults


# ===== LEGACY COMPATIBILITY =====

# Aliases for backward compatibility
State = OptAgentState


def create_state(**kwargs) -> OptAgentState:
    """
    Factory function for creating OptAgent state.

    Args:
        **kwargs: Optional field values to override defaults

    Returns:
        OptAgentState: Fully initialized state dictionary
    """
    return create_optag_state(**kwargs)
