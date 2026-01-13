# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
OptAgent Graph Package

This package contains the core components for building and executing OptAgent
optimization workflows using LangGraph state machines.

Components:
- builder: Graph construction and node composition functions
- nodes: Individual workflow nodes for validation and processing
- types: State type definitions and factory functions
"""

from .builder import (
    build_optag_graph,
    build_optag_graph_with_memory,
)
from .main_nodes import (
    modeler_node,
    verifier_node,
    corrector_node,
    visualizer_node,
    reporter_node,
)
from .types import (
    OptAgentState,
    create_optag_state,
    create_state,
    State,
)

__all__ = [
    # Graph builders
    "build_optag_graph",
    "build_optag_graph_with_memory",
    # Workflow nodes
    "modeler_node",
    "verifier_node",
    "corrector_node",
    "visualizer_node",
    "reporter_node",
    # State types and factories
    "OptAgentState",
    "create_optag_state",
    "create_state",
    "State",
]
