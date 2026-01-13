# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Evaluation system for testing agent framework on benchmarks.

This module provides a comprehensive evaluation system for testing the 
performance of the agent framework on various optimization benchmarks.

Key Components:
- BenchmarkEvaluator: Main evaluator class
- EvalConfig: Configuration management
- BenchmarkRunner: Concurrent test execution
- ResultExtractor: Extract numerical results from agent outputs  
- MetricsCalculator: Calculate evaluation metrics
- TraceManager: Manage evaluation traces and results

Example Usage:
    from src.eval import BenchmarkEvaluator
    
    evaluator = BenchmarkEvaluator(
        benchmark_file="benchmark.jsonl",
        concurrency=4,
        pass_n=3,
        model_name="gpt-4"
    )
    
    results = evaluator.run()
    print(f"Accuracy: {results['overview']['accuracy']:.3f}")
"""

from .evaluator import BenchmarkEvaluator
from .config import EvalConfig

# BenchmarkRunner functionality is now integrated into BenchmarkEvaluator
from .result_extractor import ResultExtractor
from .metrics_calculator import MetricsCalculator
from .trace_manager import TraceManager

__all__ = [
    "BenchmarkEvaluator",
    "EvalConfig",
    "ResultExtractor",
    "MetricsCalculator",
    "TraceManager",
]

__version__ = "1.0.0"
