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
    CPMPY = "cpmpy"  # CPMpy constraint programming library


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
    "cpmpy": {
        "name": "CPMpy",
        "package": "cpmpy",
        "import": "import cpmpy as cp",
        "model_class": "cp.Model",
        "solver_class": "cp.Model.solve",
        "variable_types": ["intvar", "boolvar"],
        "supported_constraints": [
            "AllDifferent", "Cumulative", "NoOverlap", "Circuit",
            "Inverse", "Table", "Count", "Element", "Minimum",
            "Maximum", "Abs", "GlobalCardinalityCount"
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
    if not problem_statement:
        return False
    
    problem_lower = problem_statement.lower()
    
    # MP-specific keywords that should NOT trigger CP template
    mp_keywords = [
        '线性规划', '非线性规划', '二次规划', '凸规划', '整数规划',
        '线性回归', '非线性回归', '逻辑回归', '回归分析',
        '投资组合', '组合优化', '风险收益', '资产配置',
        '线性', '非线性', '二次', '凸', '回归',
        '最大化利润', '最小化成本', '资源约束', '利润最大化', '成本最小化'
    ]
    
    # First check for MP keywords (exclude these from CP)
    if any(keyword in problem_lower for keyword in mp_keywords):
        return False
    
    # Check for clear MIP indicators - if these are present, it's likely an MIP problem
    mip_indicators = [
        'maximize', 'minimize', 'subject to', '<=', '>=', '==', '=',
        'constraints:', '约束条件', '目标函数', 'objective function'
    ]
    
    # Count MIP indicators
    mip_indicator_count = sum(1 for indicator in mip_indicators if indicator in problem_lower)
    
    # If there are strong MIP indicators, it's probably not a CP problem
    if mip_indicator_count >= 3:
        # Additional check for mathematical expressions
        import re
        # Look for mathematical expressions with variables and operators
        math_expr_pattern = r'[a-zA-Z]\s*[+\-*\/<>]=?\s*[a-zA-Z\d]'
        math_expressions = len(re.findall(math_expr_pattern, problem_lower))
        
        # If we have both MIP indicators and mathematical expressions, it's likely MIP
        if math_expressions >= 2:
            return False
    
    # CP-specific keywords (English and Chinese)
    cp_keywords = [
        # English keywords
        'schedule', 'timetable', 'assignment', 'routing', 'sequence',
        'permutation', 'alldifferent', 'cumulative', 'interval',
        'precedence', 'no-overlap', 'circuit', 'element', 'all different',
        'timetabling', 'shift', 'roster', 'tour', 'vehicle routing',
        'bin packing', 'container loading', 'cutting stock', 'job shop',
        'flow shop', 'scheduling', 'sequencing', 'tournament', 'round-robin',
        'sports', 'league', 'fixture', 'match', 'game', 'team', 'player',
        'delivery route', 'truck', 'visit customer', 'task allocation',
        'worker constraint', 'employee shift', 'team arrangement',
        'assign task', 'allocate resource', 'assign worker', 'task to worker',
        'tasks to workers', 'skill constraints', 'worker skills', 'task assignment',
        'personnel assignment', 'job assignment', 'staff scheduling',
        # Special terms that strongly indicate CP problems
        'exactly one', 'one to one', 'each worker', 'each task',
        # Chinese keywords (more specific to avoid false positives)
        '排班', '排班表', '调度', '路径规划', '分配', '序列',
        '排列', '约束', '约束规划', '约束编程', '装箱', '装箱问题',
        '切割', '切割问题', '作业', '作业调度', '车间', '车间调度',
        '车辆', '车辆路径', '旅行商', '旅行商问题', '资源分配',
        '班次', '班表', '时间表', '时间安排', '任务分配', '任务调度',
        '赛事', '比赛', '球队', '主场', '客场', '时段', '单循环赛',
        '体育', '联赛', '赛程', '对阵', '轮次', '循环赛', '淘汰赛',
        '配送路线', '卡车', '访问客户', '任务分配', '工人约束', '员工班次', '队伍安排',
        '分配任务', '资源配置', '分配工人', '任务给工人', '技能约束', '工人技能'
    ]
    
    # Then check for CP indicators
    if any(keyword in problem_lower for keyword in cp_keywords):
        return True
    
    # Check for discrete combinatorial nature
    discrete_keywords = ['integer', 'binary', 'count', 'number of', 'how many', 
                         '整数', '二进制', '计数', '数量', '多少个']
    if any(keyword in problem_lower for keyword in discrete_keywords):
        # Additional check for assignment-like language
        assignment_keywords = ['assign', '分配', '匹配', 'match', 'allocate', 'allocate']
        if any(keyword in problem_lower for keyword in assignment_keywords):
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
        return CPSolver.CPMPY
    
    # Problem-specific recommendations - now using CPMpy for all problem types
    solver_recommendations = {
        CPProblemType.SCHEDULING: CPSolver.CPMPY,
        CPProblemType.ROUTING: CPSolver.CPMPY,
        CPProblemType.ASSIGNMENT: CPSolver.CPMPY,
        CPProblemType.PACKING: CPSolver.CPMPY,
        CPProblemType.SEQUENCING: CPSolver.CPMPY,
        CPProblemType.CSP: CPSolver.CPMPY,
        CPProblemType.COP: CPSolver.CPMPY
    }
    
    return solver_recommendations.get(problem_type, CPSolver.CPMPY)