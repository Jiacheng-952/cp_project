# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import re
from typing import Dict, Any, Optional, List
from contextlib import redirect_stdout, redirect_stderr
import io

from langchain_core.tools import tool
from .decorators import log_io

logger = logging.getLogger(__name__)


def extract_cp_solution_status(stdout_content: str) -> Optional[str]:
    """
    Extract CP solver status from execution output.
    
    Args:
        stdout_content: Standard output from CP solver execution
        
    Returns:
        Solution status string or None if not found
    """
    if not stdout_content:
        return None
    
    # 首先检查明确的INFEASIBLE状态（最高优先级）
    if re.search(r'optimal_value\s*=\s*INFEASIBLE', stdout_content, re.IGNORECASE):
        return 'INFEASIBLE'
    if re.search(r'objective\s*=\s*INFEASIBLE', stdout_content, re.IGNORECASE):
        return 'INFEASIBLE'
    if re.search(r'result\s*=\s*INFEASIBLE', stdout_content, re.IGNORECASE):
        return 'INFEASIBLE'
    if re.search(r'PROBLEM IS INFEASIBLE', stdout_content, re.IGNORECASE):
        return 'INFEASIBLE'
    if re.search(r'问题无可行解', stdout_content):
        return 'INFEASIBLE'
    
    # CP solver status patterns
    cp_status_patterns = [
        (r'optimal_value\s*=\s*(OPTIMAL|FEASIBLE|MODEL_INVALID|ERROR)', re.IGNORECASE),
        (r'OPTIMAL SOLUTION FOUND', re.IGNORECASE),
        (r'FEASIBLE SOLUTION FOUND', re.IGNORECASE),
        (r'MODEL IS INVALID', re.IGNORECASE),
        (r'最优解已找到', re.IGNORECASE),
        (r'可行解已找到', re.IGNORECASE),
        (r'模型无效', re.IGNORECASE),
        (r'求解失败', re.IGNORECASE),
    ]
    
    for pattern, flags in cp_status_patterns:
        match = re.search(pattern, stdout_content, flags)
        if match:
            if len(match.groups()) > 0:
                status = match.group(1).upper()
                # Map Chinese statuses to English equivalents
                status_mapping = {
                    '最优解已找到': 'OPTIMAL',
                    '可行解已找到': 'FEASIBLE',
                    '模型无效': 'MODEL_INVALID',
                    '求解失败': 'ERROR'
                }
                return status_mapping.get(status, status)
            else:
                return match.group(0).upper()
    
    # 最后检查数值最优值（最低优先级）
    if re.search(r'optimal_value\s*=\s*[\d.-]+', stdout_content, re.IGNORECASE):
        return 'FEASIBLE'  # 有数值最优值，说明是可行解
    
    return None


def extract_cp_optimal_value(stdout_content: str) -> Optional[float]:
    """
    Extract optimal value from CP solver execution output.
    
    Args:
        stdout_content: Standard output from CP solver execution
        
    Returns:
        Optimal value as float, or None if not found
    """
    if not stdout_content:
        return None
    
    # Patterns for extracting optimal value from CP solver output
    cp_value_patterns = [
        r'optimal_value\s*=\s*([\d.-]+)',
        r'objective\s*=\s*([\d.-]+)',
        r'result\s*=\s*([\d.-]+)',
        r'ObjectiveValue\(\)\s*:\s*([\d.-]+)',
        r'最优值\s*=\s*([\d.-]+)',
        r'当前目标值\s*=\s*([\d.-]+)',
    ]
    
    for pattern in cp_value_patterns:
        match = re.search(pattern, stdout_content)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                continue
    
    return None


def has_feasible_solution(stdout_content: str) -> bool:
    """
    Check if a feasible solution was found in CP solver output.
    This is especially important for CSP problems that may not have an objective function.
    This function helps distinguish between problems with feasible solutions and infeasible problems.
    
    Args:
        stdout_content: Standard output from CP solver execution
        
    Returns:
        True if a feasible solution was found, False otherwise
    """
    if not stdout_content:
        return False
    
    # Try to parse JSON first - CPMpy outputs JSON format
    try:
        import json
        data = json.loads(stdout_content.strip())
        if isinstance(data, dict):
            optimal_value = data.get('optimal_value')
            # Check if optimal_value exists and is not INFEASIBLE
            if optimal_value is not None and optimal_value != "INFEASIBLE":
                return True
            # Also check for any solution data
            if data.get('result') or any(k not in ['optimal_value', 'objective', 'result'] for k in data.keys()):
                return True
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Split stdout into lines for more precise processing
    lines = stdout_content.split('\n')
    
    # First check for explicit INFEASIBLE status - if found, return False immediately
    for line in lines:
        if re.search(r'optimal_value\s*=\s*INFEASIBLE', line, re.IGNORECASE) or \
           re.search(r'objective\s*=\s*INFEASIBLE', line, re.IGNORECASE) or \
           re.search(r'result\s*=\s*INFEASIBLE', line, re.IGNORECASE):
            return False
    
    # Patterns indicating a feasible solution was found
    feasible_patterns = [
        # Specific patterns for feasible/optimal solutions
        r'optimal_value\s*=\s*(OPTIMAL|FEASIBLE)',
        r'objective\s*=\s*(OPTIMAL|FEASIBLE)',
        r'result\s*=\s*(OPTIMAL|FEASIBLE)',
        r'optimal_value\s*=\s*[\d.-]+',  # Numeric optimal value indicates a solution
        r'objective\s*=\s*[\d.-]+',
        r'result\s*=\s*[\d.-]+',
        r'OPTIMAL SOLUTION FOUND',
        r'FEASIBLE SOLUTION FOUND',
        r'最优解已找到',
        r'可行解已找到',
        r'feasible\s*=\s*是',
        r'可行解\s*=\s*是',
        # More specific patterns to detect variable value output and solution messages
        r'^\s*[\w\[\]]+\s*=\s*[\d\w\.\-]+\s*$',  # Variable assignment patterns like "x = 5" or "q[0] = 3" at line start/end
        r'变量取值:',  # Chinese "Variable values:"
        r'Solution found!',  # Generic solution found message
        r'解已找到',  # Chinese "Solution found"
        r'Solution time:',  # Solution time indicator
    ]
    
    for line in lines:
        # Skip lines that clearly indicate infeasibility
        if 'INFEASIBLE' in line.upper() or '不可行' in line:
            continue
            
        for pattern in feasible_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
    
    return False


def extract_cp_variable_values(stdout_content: str) -> Dict[str, Any]:
    """
    Extract CP variable values from execution output.
    Supports both JSON format (CPMpy) and line-by-line variable assignments.
    
    Args:
        stdout_content: Standard output from CP solver execution
        
    Returns:
        Dictionary of variable names and their values
    """
    variables = {}
    
    if not stdout_content:
        return variables
    
    # First try to parse as JSON (CPMpy format)
    try:
        import json
        # Try to extract JSON from the content
        data = json.loads(stdout_content.strip())
        
        if isinstance(data, dict):
            # Extract all keys except known metadata fields
            metadata_fields = ['optimal_value', 'objective', 'result', 'status']
            for key, value in data.items():
                if key not in metadata_fields:
                    variables[key] = value
            
            # If we found variables in JSON, return them
            if variables:
                return variables
    except (json.JSONDecodeError, ValueError):
        # Not JSON format, fall back to line-based parsing
        pass
    
    # Fallback: Split stdout into lines for more precise processing
    lines = stdout_content.split('\n')
    
    # Pattern to match variable assignments like "x = 5" or "variable_name = value"
    # But be more restrictive to avoid matching text explanations
    variable_pattern = r'^\s*(\w+(?:\[\w+\])?)\s*=\s*([\d.-]+)\s*$'
    
    # Keywords that indicate this is not a variable assignment
    exclusion_keywords = ['optimal_value', 'objective', 'result']
    
    for line in lines:
        match = re.match(variable_pattern, line)
        if match:
            var_name, var_value = match.groups()
            
            # Skip lines with exclusion keywords
            if var_name.lower() in exclusion_keywords:
                continue
                
            try:
                # Try to convert to int first, then float
                if '.' in var_value:
                    variables[var_name] = float(var_value)
                else:
                    variables[var_name] = int(var_value)
            except (ValueError, TypeError):
                variables[var_name] = var_value
    
    return variables


@tool
@log_io
def cp_solver_executor_tool(
    code: str,
    timeout_seconds: int = 300  # 增加到5分钟，适应复杂问题
) -> Dict[str, Any]:
    """
    Execute CP (Constraint Programming) solver code using various CP solvers including CPMpy.
    
    This tool is specifically designed for CP problems and provides:
    - Safe execution of CP solver code
    - Extraction of CP-specific solution metrics
    - Detailed error reporting for CP models
    
    Args:
        code: Python code containing CP model using CPMpy or OR-Tools CP-SAT
        timeout_seconds: Maximum execution time (default: 60 seconds)
        
    Returns:
        Dict containing CP execution results
    """
    if not isinstance(code, str) or not code.strip():
        error_msg = "Invalid or empty CP code provided"
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
            "variable_values": {},
            "cp_specific": {
                "solver_used": "OR-Tools CP-SAT",
                "problem_type": "Unknown",
                "search_time": 0
            }
        }
    
    logger.info("Executing CP solver code")
    
    # Import the standard execution evaluator
    from .execution_evaluator import execute_and_evaluate
    
    try:
        # Execute the code using the standard evaluator with timeout
        result = execute_and_evaluate(code, timeout_seconds)
        
        # Extract CP-specific information
        stdout_content = result.get("stdout", "")
        
        # Extract CP solution status
        cp_status = extract_cp_solution_status(stdout_content)
        
        # Extract CP optimal value
        cp_optimal_value = extract_cp_optimal_value(stdout_content)
        
        # Extract variable values
        variable_values = extract_cp_variable_values(stdout_content)
        
        # Determine CP problem type from code
        problem_type = "Unknown"
        cp_keywords = {
            "scheduling": ["schedule", "timetable", "interval", "precedence"],
            "routing": ["route", "tour", "circuit", "vehicle"],
            "assignment": ["assign", "matching", "allocation"],
            "packing": ["pack", "bin", "container", "cutting"],
            "sequencing": ["sequence", "permutation", "order"]
        }
        
        code_lower = code.lower()
        for ptype, keywords in cp_keywords.items():
            if any(keyword in code_lower for keyword in keywords):
                problem_type = ptype
                break
        
        # Extract search time if available
        search_time = 0
        time_match = re.search(r'Solution time:\s*([\d.]+)s', stdout_content)
        if time_match:
            try:
                search_time = float(time_match.group(1))
            except (ValueError, TypeError):
                pass
        
        # Check if we have a feasible solution (important for CSP problems)
        has_feasible = has_feasible_solution(stdout_content)
        
        # For CSP problems, we might not have an optimal value but still have a feasible solution
        final_optimal_value = None
        if cp_optimal_value is not None:
            final_optimal_value = cp_optimal_value
        elif result.get("optimal_value") is not None:
            final_optimal_value = result.get("optimal_value")
        elif has_feasible and cp_status in ["OPTIMAL", "FEASIBLE"]:
            # For CSP problems with feasible solutions, set a default value to indicate success
            # This ensures that CSP problems are properly recognized as having solutions
            final_optimal_value = "FEASIBLE_SOLUTION_FOUND"
        
        # Determine which solver was used
        solver_used = "Unknown"
        if "cpmpy" in code.lower():
            solver_used = "CPMpy"
        elif "ortools" in code.lower():
            solver_used = "OR-Tools CP-SAT"
        
        # Enhance result with CP-specific information
        result.update({
            "optimal_value": final_optimal_value,
            "solution_status": cp_status if cp_status else result.get("solution_status"),
            "variable_values": variable_values,
            "has_feasible_solution": has_feasible,
            "cp_specific": {
                "solver_used": solver_used,
                "problem_type": problem_type,
                "search_time": search_time,
                "status_mapping": {
                    "OPTIMAL": "Optimal solution found",
                    "FEASIBLE": "Feasible solution found (may not be optimal)",
                    "INFEASIBLE": "Problem is infeasible",
                    "MODEL_INVALID": "Model is invalid",
                    "ERROR": "Solver error occurred"
                }
            }
        })
        
        logger.info(f"CP solver execution completed: {result['status']}")
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"CP solver execution failed: {str(e)}"
        traceback_str = traceback.format_exc()
        
        logger.error(f"{error_msg}\n{traceback_str}")
        
        return {
            "status": "ERROR",
            "executed": False,
            "stdout": "",
            "stderr": traceback_str,
            "error_message": error_msg,
            "execution_time": 0,
            "optimal_value": None,
            "solution_status": None,
            "variable_values": {},
            "cp_specific": {
                "solver_used": "OR-Tools CP-SAT",
                "problem_type": "Unknown",
                "search_time": 0,
                "error_details": str(e)
            }
        }


def execute_cp_model(code: str, timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Direct function interface for CP model execution.
    
    Useful for internal usage without LangChain tool overhead.
    
    Args:
        code: Python code containing CP model
        timeout_seconds: Maximum execution time
        
    Returns:
        CP execution results
    """
    return cp_solver_executor_tool.func(code, timeout_seconds)


# Test function for CP solver execution
def test_cp_solver():
    """Test the CP solver executor with a simple example."""
    # Test with CPMpy
    test_code_cpmpy = """
import cpmpy as cp
import json

# Simple CP model example using CPMpy
model = cp.Model()

# Decision variables
x = cp.intvar(0, 10, name='x')
y = cp.intvar(0, 10, name='y')

# Constraints
model += (x + y == 10)
model += (x <= y)

# Objective
model.maximize(x + 2 * y)

# Solve
if model.solve():
    solution = {'optimal_value': int(model.objective_value()), 'x': int(x.value()), 'y': int(y.value())}
    print(json.dumps(solution))
else:
    print("optimal_value = INFEASIBLE")
"""
    
    result = execute_cp_model(test_code_cpmpy)
    print("CP Solver Test Result (CPMpy):")
    print(f"Status: {result['status']}")
    print(f"Optimal Value: {result['optimal_value']}")
    print(f"Solution Status: {result['solution_status']}")
    print(f"Variable Values: {result['variable_values']}")
    
    return result


if __name__ == "__main__":
    # Run test if executed directly
    test_cp_solver()