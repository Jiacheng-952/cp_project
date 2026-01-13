# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Performance Optimizer Module for OptAgent

This module provides performance optimization utilities for the OptAgent system,
including caching mechanisms, timeout control, and parallel processing.
"""

import hashlib
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from functools import wraps
from src.utils.performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)


from src.config.performance_config import PerformanceOptimizationConfig, get_performance_config


class CodeExecutionCache:
    """Cache for code execution results to avoid redundant computations."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self.performance_monitor = get_performance_monitor()
    
    def _generate_key(self, code: str, timeout: int) -> str:
        """Generate a unique cache key for code and timeout combination."""
        content = f"{code}:{timeout}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self._access_times.items()
            if current_time - access_time > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used entry if cache is full."""
        if len(self._cache) >= self.max_size:
            lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            self._cache.pop(lru_key, None)
            self._access_times.pop(lru_key, None)
    
    def get(self, code: str, timeout: int) -> Optional[Dict[str, Any]]:
        """Get cached execution result if available."""
        key = self._generate_key(code, timeout)
        
        if key in self._cache:
            self._access_times[key] = time.time()
            logger.debug(f"Cache hit for code execution: {key[:16]}...")
            self.performance_monitor.record_cache_hit("code_execution")
            return self._cache[key]
        
        logger.debug(f"Cache miss for code execution: {key[:16]}...")
        self.performance_monitor.record_cache_miss("code_execution")
        return None
    
    def set(self, code: str, timeout: int, result: Dict[str, Any]):
        """Store execution result in cache."""
        self._cleanup_expired()
        self._evict_lru()
        
        key = self._generate_key(code, timeout)
        self._cache[key] = result
        self._access_times[key] = time.time()
        logger.debug(f"Cached code execution result: {key[:16]}...")
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Code execution cache cleared")


class TimeoutController:
    """Dynamic timeout control based on problem complexity."""
    
    def __init__(self, config: PerformanceOptimizationConfig):
        self.config = config
    
    def estimate_complexity(self, problem_statement: str) -> str:
        """Estimate problem complexity based on keywords and structure."""
        statement_lower = problem_statement.lower()
        
        # Simple problems: basic constraints, small search space
        simple_indicators = [
            'simple', 'basic', 'small', 'few', 'minimal', 'elementary',
            'assignment', 'allocation', 'matching', 'selection'
        ]
        
        # Complex problems: multiple constraints, large search space
        complex_indicators = [
            'complex', 'difficult', 'large', 'many', 'multiple', 'advanced',
            'scheduling', 'routing', 'timetabling', 'optimization',
            'cumulative', 'nooverlap', 'precedence', 'resource'
        ]
        
        simple_count = sum(1 for indicator in simple_indicators if indicator in statement_lower)
        complex_count = sum(1 for indicator in complex_indicators if indicator in statement_lower)
        
        if complex_count > simple_count:
            return "complex"
        elif simple_count > 0:
            return "simple"
        else:
            return "medium"
    
    def get_llm_timeout(self, problem_statement: str) -> int:
        """Get appropriate LLM timeout based on problem complexity."""
        if not self.config.timeout.enable_dynamic_timeout:
            return self.config.timeout.default_llm_timeout
        
        complexity = self.estimate_complexity(problem_statement)
        
        if complexity == "simple":
            return self.config.timeout.simple_problem_timeout
        elif complexity == "complex":
            return self.config.timeout.complex_problem_timeout
        else:
            return self.config.timeout.default_llm_timeout


class QuickVerificationEngine:
    """Fast verification engine using rule-based checks."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, Callable]:
        """Initialize verification rules."""
        return {
            "basic_solution_check": self._basic_solution_check,
            "constraint_satisfaction": self._constraint_satisfaction_check,
            "objective_validation": self._objective_validation_check,
        }
    
    def _basic_solution_check(self, execution_result: Dict[str, Any]) -> bool:
        """Basic check for solution existence and validity."""
        if execution_result.get("status") != "PASS":
            return False
        
        if not execution_result.get("executed", False):
            return False
        
        # Check if we have any solution indicators
        has_solution = (
            execution_result.get("optimal_value") is not None or
            execution_result.get("has_feasible_solution", False) or
            execution_result.get("variable_values", {})
        )
        
        return has_solution
    
    def _constraint_satisfaction_check(self, execution_result: Dict[str, Any], 
                                     problem_statement: str) -> bool:
        """Quick constraint satisfaction check."""
        # For now, rely on the solver's internal constraint checking
        # This can be enhanced with domain-specific constraint validation
        return execution_result.get("status") == "PASS"
    
    def _objective_validation_check(self, execution_result: Dict[str, Any], 
                                  problem_statement: str) -> bool:
        """Quick objective function validation."""
        optimal_value = execution_result.get("optimal_value")
        
        # For CSP problems, "FEASIBLE_SOLUTION_FOUND" is valid
        if optimal_value == "FEASIBLE_SOLUTION_FOUND":
            return True
        
        # Basic sanity checks
        if optimal_value is None:
            return False
        
        # Check if optimal value makes sense
        if isinstance(optimal_value, (int, float)):
            # Check for obviously wrong values
            if optimal_value < 0 and "minimize" not in problem_statement.lower():
                return False
            if optimal_value > 1e9:  # Unreasonably large value
                return False
        
        return True
    
    def quick_verify(self, execution_result: Dict[str, Any], 
                    problem_statement: str) -> Dict[str, Any]:
        """Perform quick verification using rule-based checks."""
        start_time = time.time()
        
        # Apply all rules
        results = {}
        for rule_name, rule_func in self.rules.items():
            try:
                if rule_name == "basic_solution_check":
                    results[rule_name] = rule_func(execution_result)
                else:
                    results[rule_name] = rule_func(execution_result, problem_statement)
            except Exception as e:
                logger.warning(f"Rule {rule_name} failed: {e}")
                results[rule_name] = False
        
        # Overall verification result
        all_passed = all(results.values())
        
        verification_result = {
            "quick_verification_passed": all_passed,
            "rule_results": results,
            "verification_time_ms": int((time.time() - start_time) * 1000),
            "verification_method": "quick_rule_based"
        }
        
        logger.info(f"Quick verification completed in {verification_result['verification_time_ms']}ms: {all_passed}")
        return verification_result


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, config: Optional[PerformanceOptimizationConfig] = None):
        self.config = config or get_performance_config()
        self.code_cache = CodeExecutionCache(
            max_size=self.config.cache.cache_max_size,
            ttl_seconds=self.config.cache.cache_ttl_seconds
        )
        self.timeout_controller = TimeoutController(self.config)
        self.quick_verifier = QuickVerificationEngine()
    
    def get_cached_execution_result(self, code: str, timeout: int) -> Optional[Dict[str, Any]]:
        """Get cached execution result if available."""
        if not self.config.cache.enable_code_cache:
            return None
        
        return self.code_cache.get(code, timeout)
    
    def cache_execution_result(self, code: str, timeout: int, result: Dict[str, Any]):
        """Cache execution result."""
        if self.config.cache.enable_code_cache:
            self.code_cache.set(code, timeout, result)
    
    def get_optimized_timeout(self, problem_statement: str) -> int:
        """Get optimized timeout based on problem complexity."""
        return self.timeout_controller.get_llm_timeout(problem_statement)
    
    async def parallel_verification(self, code: str, problem_statement: str, 
                                  timeout: int) -> Dict[str, Any]:
        """Perform parallel verification tasks."""
        if not self.config.verification.enable_parallel_verification:
            # Fallback to sequential execution
            from src.tools import cp_solver_executor_tool
            execution_result = cp_solver_executor_tool.invoke({
                "code": code, 
                "timeout_seconds": timeout
            })
            return self.quick_verifier.quick_verify(execution_result, problem_statement)
        
        # Parallel execution
        from src.tools import cp_solver_executor_tool
        
        async def execute_code():
            return cp_solver_executor_tool.invoke({
                "code": code, 
                "timeout_seconds": timeout
            })
        
        async def quick_verify_wrapper():
            # This would run in parallel with code execution
            # For now, we'll run it after execution
            return {}
        
        # Execute code and quick verification in parallel
        execution_result, _ = await asyncio.gather(
            execute_code(),
            quick_verify_wrapper()
        )
        
        # Perform quick verification on the result
        return self.quick_verifier.quick_verify(execution_result, problem_statement)


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create the global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def performance_timed(func: Callable):
    """Decorator to measure and log function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = int((time.time() - start_time) * 1000)
        logger.info(f"{func.__name__} executed in {execution_time}ms")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = int((time.time() - start_time) * 1000)
        logger.info(f"{func.__name__} executed in {execution_time}ms")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper