# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Constraint Programming (CP) Solvers Configuration

This module provides configuration for various CP solvers and their integration
with the OptAgent framework.
"""

from enum import Enum
from typing import Dict, Any


class CPSolver(Enum):
    """Supported CP solvers enumeration."""
    ORTOOLS_CP_SAT = "ortools_cp_sat"  # Google OR-Tools CP-SAT
    ORTOOLS_CP = "ortools_cp"  # Google OR-Tools CP (legacy)
    CPLEX_CP = "cplex_cp"  # IBM CPLEX CP Optimizer
    MINIZINC = "minizinc"  # MiniZinc modeling language
    GECODE = "gecode"  # Gecode solver
    CHUFFED = "chuffed"  # Chuffed solver


class CPProblemType(Enum):
    """CP problem classification types."""
    SCHEDULING = "scheduling"
    ROUTING = "routing"
    ASSIGNMENT = "assignment"
    PACKING = "packing"
    SEQUENCING = "sequencing"
    CSP = "csp"  # Constraint Satisfaction Problem
    COP = "cop"  # Constraint Optimization Problem


class CPSearchStrategy(Enum):
    """CP search strategies."""
    DEFAULT = "default"
    FIRST_SOLUTION = "first_solution"
    BEST_BOUND = "best_bound"
    PARALLEL = "parallel"


# CP Solver Configuration
CP_SOLVER_CONFIG: Dict[str, Dict[str, Any]] = {
    "ortools_cp_sat": {
        "name": "OR-Tools CP-SAT",
        "package": "ortools",
        "import": "from ortools.sat.python import cp_model",
        "model_class": "cp_model.CpModel",
        "solver_class": "cp_model.CpSolver",
        "variable_types": ["IntVar", "BoolVar", "IntervalVar"],
        "supported_constraints": [
            "AllDifferent", "Linear", "Element", "Circuit",
            "Cumulative", "NoOverlap", "Inverse"
        ]
    },
    "minizinc": {
        "name": "MiniZinc",
        "package": "minizinc",
        "model_language": "minizinc",
        "supported_constraints": [
            "alldifferent", "cumulative", "disjunctive",
            "element", "table", "regular"
        ]
    }
}


# CP Problem Type Keywords for Automatic Detection
CP_PROBLEM_KEYWORDS: Dict[CPProblemType, list] = {
    CPProblemType.SCHEDULING: [
        'schedule', 'timetable', 'calendar', 'shift', 'roster',
        'deadline', 'precedence', 'interval', 'duration',
        'start time', 'end time', 'makespan', 'tardiness'
    ],
    CPProblemType.ROUTING: [
        'route', 'path', 'tour', 'traveling salesman', 'tsp',
        'vehicle routing', 'vrp', 'circuit', 'hamiltonian',
        'delivery', 'pickup', 'depot'
    ],
    CPProblemType.ASSIGNMENT: [
        'assign', 'allocation', 'matching', 'bipartite',
        'personnel assignment', 'task assignment', 'room assignment',
        'resource assignment'
    ],
    CPProblemType.PACKING: [
        'pack', 'bin packing', 'container loading', 'knapsack',
        'cutting stock', 'pallet loading', '2d packing', '3d packing'
    ],
    CPProblemType.SEQUENCING: [
        'sequence', 'permutation', 'ordering', 'precedence',
        'successor', 'predecessor', 'job shop', 'flow shop'
    ]
}


def detect_cp_problem_type(problem_statement: str) -> CPProblemType:
    """
    Detect the most appropriate CP problem type based on keywords in the problem statement.
    
    Args:
        problem_statement: The natural language description of the optimization problem
        
    Returns:
        The detected CP problem type, or None if no clear match
    """
    problem_lower = problem_statement.lower()
    
    # Count keyword matches for each problem type
    match_counts = {}
    for problem_type, keywords in CP_PROBLEM_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in problem_lower)
        if count > 0:
            match_counts[problem_type] = count
    
    if match_counts:
        # Return the problem type with the highest match count
        return max(match_counts.items(), key=lambda x: x[1])[0]
    
    return None


def should_use_cp_solver(problem_statement: str) -> bool:
    """
    Determine whether a CP solver is appropriate for the given problem.
    
    Args:
        problem_statement: The natural language description of the optimization problem
        
    Returns:
        True if CP solver is recommended, False otherwise
    """
    # Check for CP-specific keywords
    cp_keywords = [
        'schedule', 'timetable', 'assignment', 'routing', 'sequence',
        'permutation', 'alldifferent', 'cumulative', 'interval',
        'precedence', 'no-overlap', 'circuit', 'element', 'all different'
    ]
    
    problem_lower = problem_statement.lower()
    
    # Check for explicit CP indicators
    if any(keyword in problem_lower for keyword in cp_keywords):
        return True
    
    # Check for discrete combinatorial nature
    discrete_keywords = ['integer', 'binary', 'count', 'number of', 'how many']
    if any(keyword in problem_lower for keyword in discrete_keywords):
        return True
    
    return False


def get_recommended_cp_solver(problem_type: CPProblemType = None) -> CPSolver:
    """
    Get the recommended CP solver for a given problem type.
    
    Args:
        problem_type: The detected CP problem type
        
    Returns:
        Recommended CP solver
    """
    # Default recommendation
    if problem_type is None:
        return CPSolver.ORTOOLS_CP_SAT
    
    # Problem-specific recommendations
    solver_recommendations = {
        CPProblemType.SCHEDULING: CPSolver.ORTOOLS_CP_SAT,
        CPProblemType.ROUTING: CPSolver.ORTOOLS_CP_SAT,
        CPProblemType.ASSIGNMENT: CPSolver.ORTOOLS_CP_SAT,
        CPProblemType.PACKING: CPSolver.ORTOOLS_CP_SAT,
        CPProblemType.SEQUENCING: CPSolver.ORTOOLS_CP_SAT,
        CPProblemType.CSP: CPSolver.ORTOOLS_CP_SAT,
        CPProblemType.COP: CPSolver.ORTOOLS_CP_SAT
    }
    
    return solver_recommendations.get(problem_type, CPSolver.ORTOOLS_CP_SAT)