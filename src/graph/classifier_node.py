# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Problem Classifier Node Module

This module provides intelligent problem classification using LLM
for determining whether a problem is best suited for Constraint Programming (CP)
or Mathematical Programming (MP).
"""

import logging
import asyncio
from typing import Dict, Any

from .types import OptAgentState

logger = logging.getLogger(__name__)


async def classifier_node(state: OptAgentState, config) -> Dict[str, Any]:
    """
    Intelligent problem classifier node for determining problem type.
    
    This node uses LLM to analyze the problem statement and determine
    whether it's best suited for Constraint Programming (CP) or 
    Mathematical Programming (MP).
    
    Args:
        state: Current OptAgent state
        config: LangGraph node configuration
        
    Returns:
        State update dictionary with problem classification
    """
    logger.info("Classifier node analyzing problem type")
    
    # Extract problem statement
    problem_statement = state.get("problem_statement", "")
    
    # Determine if this is CSP or COP based on keywords
    problem_lower = problem_statement.lower()
    
    # COP indicators: optimization keywords
    cop_keywords = [
        'minimize', 'maximize', '最小化', '最大化', '最优', '最佳',
        '目标函数', 'objective', 'optimize', 'optimization',
        '最小成本', '最大利润', '最小时间', '最大效率'
    ]
    
    # CSP indicators: satisfaction keywords (no optimization)
    csp_keywords = [
        '可行解', '满足约束', '约束满足', 'feasible', 'satisfy',
        '找到解', '求解', 'solution', 'find a solution'
    ]
    
    cop_count = sum(1 for keyword in cop_keywords if keyword in problem_lower)
    csp_count = sum(1 for keyword in csp_keywords if keyword in problem_lower)
    
    # Determine problem type
    if cop_count > csp_count and cop_count > 0:
        problem_type = "COP"
        logger.info(f"Classifier determined: COP problem (optimization keywords: {cop_count})")
    elif csp_count > 0:
        problem_type = "CSP"
        logger.info(f"Classifier determined: CSP problem (satisfaction keywords: {csp_count})")
    else:
        # Default to COP if no clear indicators (most optimization problems are COP)
        problem_type = "COP"
        logger.info("Classifier determined: COP problem (default)")
    
    return {
        "problem_type": problem_type,
        "solver_type": "cp_solver",
        "classification_result": f"{problem_type}_SPECIALIZED"
    }


def fallback_classification(problem_statement: str) -> tuple[str, str]:
    """
    Fallback classification for specialized CP agent.
    
    For this specialized CP agent, we always classify problems as CP
    regardless of their actual nature to ensure consistent CP processing.
    
    Args:
        problem_statement: The problem statement to classify
        
    Returns:
        Tuple of (problem_type, solver_type) - always returns CP classification
    """
    # Always return CP classification for specialized CP agent
    return "cp", "cp_solver"