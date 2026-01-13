# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Optimal Solution Checker Module

This module provides functions to check if a solution is optimal for known problem types,
and provides heuristic assessment for unknown problems using convergence analysis,
lower bounds, and solution quality metrics.
"""

import logging
import re
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Metrics for tracking optimization convergence."""
    iteration: int
    current_value: float
    improvement: float
    relative_improvement: float
    

class OptimalityChecker:
    """Optimal solution checker with heuristic assessment for unknown problems."""
    
    # Known optimal values for classic problems
    KNOWN_OPTIMAL_VALUES = {
        # Job Shop Scheduling problems
        "job_shop_4_3_12": {
            "pattern": r"4.*?工件.*?3.*?机器",
            "optimal_value": 12,
            "problem_type": "COP",
            "description": "4工件3机器作业车间调度问题，理论最优makespan=12"
        },
        "job_shop_ft06": {
            "pattern": r"6.*?工件.*?6.*?机器",
            "optimal_value": 55,
            "problem_type": "COP", 
            "description": "经典FT06问题，理论最优makespan=55"
        },
        # Add more known problems here
    }
    
    # Heuristic assessment parameters
    CONVERGENCE_THRESHOLD = 0.001  # 0.1% improvement threshold
    STAGNATION_ITERATIONS = 5      # Number of iterations without significant improvement
    MIN_IMPROVEMENT_RATIO = 0.01   # Minimum improvement ratio to continue optimization
    
    @classmethod
    def identify_problem_type(cls, problem_statement: str) -> Optional[str]:
        """Identify if the problem matches any known problem type."""
        for problem_id, problem_info in cls.KNOWN_OPTIMAL_VALUES.items():
            if re.search(problem_info["pattern"], problem_statement, re.IGNORECASE):
                logger.info(f"Identified known problem: {problem_id}")
                return problem_id
        return None
    
    @classmethod
    def get_known_optimal_value(cls, problem_id: str) -> Optional[float]:
        """Get known optimal value for a problem type."""
        if problem_id in cls.KNOWN_OPTIMAL_VALUES:
            return cls.KNOWN_OPTIMAL_VALUES[problem_id]["optimal_value"]
        return None
    
    @classmethod
    def check_optimality(cls, problem_statement: str, current_value: float, 
                        tolerance: float = 0.01) -> Tuple[bool, Optional[float], str]:
        """
        Check if current solution is optimal for known problem types.
        
        Args:
            problem_statement: The problem description
            current_value: Current optimal value found
            tolerance: Tolerance for floating point comparison
            
        Returns:
            Tuple of (is_optimal, known_optimal_value, message)
        """
        problem_id = cls.identify_problem_type(problem_statement)
        
        if not problem_id:
            return False, None, "Problem type not recognized"
        
        known_optimal = cls.get_known_optimal_value(problem_id)
        
        if known_optimal is None:
            return False, None, f"Known problem {problem_id} but optimal value not available"
        
        # For minimization problems, check if current value is close to known optimal
        if abs(current_value - known_optimal) <= tolerance:
            return True, known_optimal, f"Solution is optimal! Known optimal value: {known_optimal}"
        elif current_value > known_optimal:
            gap = current_value - known_optimal
            gap_percent = (gap / known_optimal) * 100
            return False, known_optimal, f"Solution is suboptimal. Gap: {gap:.2f} ({gap_percent:.1f}%)"
        else:
            # Current value is better than known optimal (should not happen for verified problems)
            return True, known_optimal, f"Solution appears better than known optimal ({known_optimal})"
    
    @classmethod
    def calculate_lower_bound(cls, problem_statement: str, execution_result: Dict[str, Any]) -> Optional[float]:
        """Calculate a lower bound for the problem to assess solution quality."""
        # For job shop scheduling, calculate simple lower bounds
        if "工件" in problem_statement and "机器" in problem_statement:
            return cls._calculate_job_shop_lower_bound(problem_statement, execution_result)
        
        return None
    
    @classmethod
    def _calculate_job_shop_lower_bound(cls, problem_statement: str, execution_result: Dict[str, Any]) -> Optional[float]:
        """Calculate lower bound for job shop scheduling problems."""
        try:
            # Extract processing times from problem statement
            processing_times = []
            
            # Pattern to match processing times like M1(3), M2(2), etc.
            pattern = r'M\d+\((\d+)\)'
            matches = re.findall(pattern, problem_statement)
            
            if matches:
                times = [int(match) for match in matches]
                
                # Simple lower bound: max of (machine workload, job workload)
                # This is a very basic lower bound calculation
                if times:
                    return max(sum(times) / 3, max(times))  # Assuming 3 machines
            
        except Exception as e:
            logger.warning(f"Failed to calculate lower bound: {e}")
        
        return None


def check_solution_quality(problem_statement: str, current_value: float, 
                          execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive solution quality assessment.
    
    Returns:
        Dictionary with quality assessment results
    """
    result = {
        "is_known_problem": False,
        "known_optimal_value": None,
        "is_optimal": False,
        "quality_gap": None,
        "lower_bound": None,
        "assessment": "Unknown problem type"
    }
    
    # Check if this is a known problem
    problem_id = OptimalityChecker.identify_problem_type(problem_statement)
    
    if problem_id:
        result["is_known_problem"] = True
        known_optimal = OptimalityChecker.get_known_optimal_value(problem_id)
        
        if known_optimal is not None:
            result["known_optimal_value"] = known_optimal
            
            # Check optimality
            is_optimal, _, message = OptimalityChecker.check_optimality(
                problem_statement, current_value
            )
            result["is_optimal"] = is_optimal
            result["assessment"] = message
            
            if not is_optimal:
                result["quality_gap"] = current_value - known_optimal
    
    # Calculate lower bound for quality assessment
    lower_bound = OptimalityChecker.calculate_lower_bound(problem_statement, execution_result)
    if lower_bound is not None:
        result["lower_bound"] = lower_bound
        
        # If we have a lower bound but no known optimal, use it for assessment
        if not result["is_known_problem"]:
            gap = current_value - lower_bound
            if gap <= 0.1:
                result["assessment"] = f"Solution is near lower bound ({lower_bound})"
            else:
                gap_percent = (gap / lower_bound) * 100
                result["assessment"] = f"Solution gap to lower bound: {gap:.2f} ({gap_percent:.1f}%)"
    
    return result


def assess_convergence(optimization_history: List[ConvergenceMetrics]) -> Dict[str, Any]:
    """
    Assess convergence based on optimization history.
    
    Args:
        optimization_history: List of convergence metrics from previous iterations
        
    Returns:
        Dictionary with convergence assessment
    """
    if len(optimization_history) < 2:
        return {
            "converged": False,
            "stagnated": False,
            "improvement_rate": 0.0,
            "assessment": "Insufficient data for convergence analysis"
        }
    
    # Calculate recent improvements
    recent_iterations = optimization_history[-OptimalityChecker.STAGNATION_ITERATIONS:]
    
    if len(recent_iterations) < 2:
        return {
            "converged": False,
            "stagnated": False,
            "improvement_rate": 0.0,
            "assessment": "Insufficient recent data"
        }
    
    # Calculate average improvement in recent iterations
    recent_improvements = [
        metric.relative_improvement for metric in recent_iterations 
        if metric.relative_improvement is not None
    ]
    
    if not recent_improvements:
        return {
            "converged": False,
            "stagnated": False,
            "improvement_rate": 0.0,
            "assessment": "No improvement data available"
        }
    
    avg_recent_improvement = sum(recent_improvements) / len(recent_improvements)
    
    # Check for convergence (very small improvements)
    converged = abs(avg_recent_improvement) < OptimalityChecker.CONVERGENCE_THRESHOLD
    
    # Check for stagnation (no significant improvement)
    stagnation_threshold = OptimalityChecker.MIN_IMPROVEMENT_RATIO
    stagnated = abs(avg_recent_improvement) < stagnation_threshold
    
    # Calculate overall improvement rate
    first_value = optimization_history[0].current_value
    last_value = optimization_history[-1].current_value
    overall_improvement_rate = (first_value - last_value) / first_value if first_value > 0 else 0
    
    assessment = ""
    if converged:
        assessment = f"Converged: average improvement {avg_recent_improvement:.4f} < threshold {OptimalityChecker.CONVERGENCE_THRESHOLD}"
    elif stagnated:
        assessment = f"Stagnated: average improvement {avg_recent_improvement:.4f} < minimum threshold {stagnation_threshold}"
    else:
        assessment = f"Still improving: average improvement {avg_recent_improvement:.4f}"
    
    return {
        "converged": converged,
        "stagnated": stagnated,
        "improvement_rate": overall_improvement_rate,
        "recent_improvement_rate": avg_recent_improvement,
        "assessment": assessment
    }


def should_continue_optimization(
    problem_statement: str,
    current_value: float,
    optimization_history: List[ConvergenceMetrics],
    execution_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determine whether optimization should continue based on multiple criteria.
    
    Args:
        problem_statement: The problem description
        current_value: Current best solution value
        optimization_history: History of optimization iterations
        execution_result: Execution result dictionary
        
    Returns:
        Dictionary with decision and reasoning
    """
    # Check if this is a known problem with optimal solution
    quality_result = check_solution_quality(problem_statement, current_value, execution_result)
    
    if quality_result["is_known_problem"] and quality_result["is_optimal"]:
        return {
            "continue": False,
            "reason": "Known problem with optimal solution found",
            "confidence": "high"
        }
    
    # Check convergence for unknown problems
    convergence_result = assess_convergence(optimization_history)
    
    if convergence_result["converged"]:
        return {
            "continue": False,
            "reason": "Optimization has converged",
            "confidence": "medium"
        }
    
    if convergence_result["stagnated"]:
        return {
            "continue": False,
            "reason": "Optimization has stagnated",
            "confidence": "medium"
        }
    
    # Check if we have a lower bound and are close to it
    if quality_result["lower_bound"] is not None:
        gap = current_value - quality_result["lower_bound"]
        gap_ratio = gap / quality_result["lower_bound"] if quality_result["lower_bound"] > 0 else float('inf')
        
        if gap_ratio < 0.05:  # Within 5% of lower bound
            return {
                "continue": False,
                "reason": f"Solution within 5% of lower bound (gap: {gap_ratio:.3f})",
                "confidence": "medium"
            }
    
    # Default: continue optimization
    return {
        "continue": True,
        "reason": "Still room for improvement",
        "confidence": "low"
    }


def create_convergence_metric(iteration: int, current_value: float, previous_value: float) -> ConvergenceMetrics:
    """Create a convergence metric for the current iteration."""
    improvement = previous_value - current_value if previous_value is not None else 0
    relative_improvement = improvement / previous_value if previous_value and previous_value != 0 else 0
    
    return ConvergenceMetrics(
        iteration=iteration,
        current_value=current_value,
        improvement=improvement,
        relative_improvement=relative_improvement
    )