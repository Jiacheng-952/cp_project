# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Metrics Calculation Module

This module provides comprehensive metrics calculation for benchmark evaluation
results. It computes accuracy, error statistics, pass@k metrics, and timing
analysis for OptAgent evaluation runs.

The MetricsCalculator class handles:
- Relative error calculations with configurable tolerance
- Pass@k metric computation for multiple k values  
- Statistical analysis of error distributions
- Timing and performance metrics
- Summary report generation

Key metrics include overall accuracy, error statistics (mean, median, min, max),
and pass@k scores that measure the probability of success within k attempts.
"""

import math
import statistics
from typing import List, Dict, Any, Optional
from collections import defaultdict


class MetricsCalculator:
    """Calculate various evaluation metrics."""

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize metrics calculator.

        Args:
            tolerance: Relative error tolerance for correctness
        """
        self.tolerance = tolerance

    def calculate_relative_error(self, predicted, expected) -> Optional[float]:
        """
        Calculate relative error between predicted and expected values.

        Args:
            predicted: Predicted value (float or "infeasible")
            expected: Expected (ground truth) value (float or string)

        Returns:
            Relative error as a float, or None for infeasible problems
        """
        # Handle infeasible problems
        if (isinstance(predicted, str) and predicted == "infeasible") or (
            isinstance(expected, str)
            and expected.lower()
            in [
                "no best solution",
                "no solution",
                "infeasible",
                "n/a",
                "na",
                "none",
                "",
            ]
        ):
            return None  # No error calculation for infeasible problems

        # Convert to float if needed
        try:
            predicted_val = float(predicted)
            expected_val = float(expected)
        except (ValueError, TypeError):
            return float("inf")  # Invalid conversion

        if expected_val == 0:
            return abs(predicted_val - expected_val)
        return abs(predicted_val - expected_val) / abs(expected_val)

    def is_correct(self, predicted, expected) -> bool:
        """
        Check if prediction is correct within tolerance.

        Args:
            predicted: Predicted value (float or "infeasible")
            expected: Expected value (float or string)

        Returns:
            True if correct, False otherwise
        """
        # Handle infeasible problems - both must indicate infeasible
        if isinstance(predicted, str) and predicted == "infeasible":
            if isinstance(expected, str) and expected.lower() in [
                "no best solution",
                "no solution",
                "infeasible",
                "n/a",
                "na",
                "none",
                "",
            ]:
                return True  # Correctly identified as infeasible
            else:
                return False  # Incorrectly identified as infeasible

        # If expected is infeasible but predicted is not
        if isinstance(expected, str) and expected.lower() in [
            "no best solution",
            "no solution",
            "infeasible",
            "n/a",
            "na",
            "none",
            "",
        ]:
            return False  # Should have identified as infeasible

        # Normal numerical comparison
        relative_error = self.calculate_relative_error(predicted, expected)
        if relative_error is None:
            return False  # Error in calculation
        return relative_error <= self.tolerance

    def calculate_pass_at_k(self, results: List[Dict[str, Any]], k: int = 1) -> float:
        """
        Calculate pass@k metric.

        Args:
            results: List of evaluation results, each containing 'correct' field
            k: Number of attempts to consider

        Returns:
            Pass@k score as a float between 0 and 1
        """
        if not results:
            return 0.0

        # Group results by problem_id
        problems = defaultdict(list)
        for result in results:
            problem_id = result.get("problem_id", "default")
            problems[problem_id].append(result)

        total_problems = len(problems)
        passed_problems = 0

        for problem_id, problem_results in problems.items():
            # Take first k attempts for this problem
            attempts = problem_results[:k]
            # Problem passes if any attempt is correct
            if any(attempt.get("correct", False) for attempt in attempts):
                passed_problems += 1

        return passed_problems / total_problems if total_problems > 0 else 0.0

    def calculate_accuracy(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall accuracy.

        Args:
            results: List of evaluation results

        Returns:
            Accuracy as a float between 0 and 1
        """
        if not results:
            return 0.0

        correct_count = sum(1 for result in results if result.get("correct", False))
        return correct_count / len(results)

    def calculate_error_statistics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate error statistics for all results.

        Args:
            results: List of evaluation results with 'relative_error' field

        Returns:
            Dictionary containing error statistics
        """
        errors = [
            result.get("relative_error", float("inf"))
            for result in results
            if result.get("relative_error") is not None
        ]

        if not errors:
            return {
                "mean_error": float("inf"),
                "median_error": float("inf"),
                "std_error": float("inf"),
                "min_error": float("inf"),
                "max_error": float("inf"),
            }

        return {
            "mean_error": statistics.mean(errors),
            "median_error": statistics.median(errors),
            "std_error": statistics.stdev(errors) if len(errors) > 1 else 0.0,
            "min_error": min(errors),
            "max_error": max(errors),
        }

    def calculate_success_rate_by_tolerance(
        self, results: List[Dict[str, Any]], tolerances: List[float]
    ) -> Dict[str, float]:
        """
        Calculate success rates at different tolerance levels.

        Args:
            results: List of evaluation results
            tolerances: List of tolerance values to test

        Returns:
            Dictionary mapping tolerance to success rate
        """
        success_rates = {}

        for tolerance in tolerances:
            correct_count = 0
            total_count = 0

            for result in results:
                relative_error = result.get("relative_error")
                if relative_error is not None:
                    total_count += 1
                    if relative_error <= tolerance:
                        correct_count += 1

            success_rates[str(tolerance)] = (
                correct_count / total_count if total_count > 0 else 0.0
            )

        return success_rates

    def generate_summary_report(
        self, results: List[Dict[str, Any]], pass_k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.

        Args:
            results: List of all evaluation results
            pass_k_values: List of k values for pass@k calculation

        Returns:
            Comprehensive summary dictionary
        """
        if pass_k_values is None:
            pass_k_values = [1, 3, 5]

        # Basic statistics
        total_problems = len(
            set(result.get("problem_id", i) for i, result in enumerate(results))
        )
        total_attempts = len(results)
        successful_attempts = sum(
            1 for result in results if result.get("correct", False)
        )

        # Calculate metrics
        accuracy = self.calculate_accuracy(results)
        error_stats = self.calculate_error_statistics(results)

        # Pass@k metrics
        pass_at_k = {}
        for k in pass_k_values:
            if k <= total_attempts:
                pass_at_k[f"pass@{k}"] = self.calculate_pass_at_k(results, k)

        # Tolerance analysis
        tolerances = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
        tolerance_analysis = self.calculate_success_rate_by_tolerance(
            results, tolerances
        )

        # Retry statistics
        retry_counts = [
            result.get("retry_count", 0)
            for result in results
            if result.get("retry_count") is not None
        ]
        retry_stats = {}
        if retry_counts:
            retry_stats = {
                "mean_retries": statistics.mean(retry_counts),
                "median_retries": statistics.median(retry_counts),
                "min_retries": min(retry_counts),
                "max_retries": max(retry_counts),
                "total_retries": sum(retry_counts),
                "problems_with_retries": sum(1 for count in retry_counts if count > 0),
                "problems_without_retries": sum(
                    1 for count in retry_counts if count == 0
                ),
                "retry_distribution": self._analyze_retry_distribution(retry_counts),
            }

        # Timing statistics
        execution_times = [
            result.get("execution_time", 0)
            for result in results
            if result.get("execution_time") is not None
        ]
        timing_stats = {}
        if execution_times:
            cumulative_time = sum(execution_times)
            timing_stats = {
                "mean_time": statistics.mean(execution_times),
                "median_time": statistics.median(execution_times),
                "cumulative_execution_time": cumulative_time,  # Sum of all individual execution times
                "total_time": cumulative_time,  # Legacy field name for backwards compatibility
                "min_time": min(execution_times),
                "max_time": max(execution_times),
            }

            # Add wall clock time and speedup metrics if available
            # These will be added by the evaluator if concurrent execution is used
            if hasattr(self, "_wall_clock_time"):
                timing_stats["wall_clock_time"] = self._wall_clock_time
                if self._wall_clock_time > 0:
                    timing_stats["actual_speedup"] = (
                        cumulative_time / self._wall_clock_time
                    )
                    if hasattr(self, "_concurrency"):
                        timing_stats["parallel_efficiency"] = (
                            timing_stats["actual_speedup"] / self._concurrency
                        ) * 100

        return {
            "overview": {
                "total_problems": total_problems,
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "accuracy": accuracy,
                "tolerance_used": self.tolerance,
            },
            "pass_at_k": pass_at_k,
            "error_statistics": error_stats,
            "tolerance_analysis": tolerance_analysis,
            "retry_statistics": retry_stats,
            "timing_statistics": timing_stats,
            "problem_distribution": self._analyze_problem_distribution(results),
        }

    def _analyze_problem_distribution(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the distribution of results across problems."""
        problems = defaultdict(list)
        for result in results:
            problem_id = result.get("problem_id", "unknown")
            problems[problem_id].append(result)

        problem_stats = {}
        for problem_id, problem_results in problems.items():
            correct_count = sum(1 for r in problem_results if r.get("correct", False))
            problem_stats[problem_id] = {
                "attempts": len(problem_results),
                "correct": correct_count,
                "success_rate": (
                    correct_count / len(problem_results) if problem_results else 0.0
                ),
            }

        return {
            "total_problems": len(problems),
            "problems_with_success": sum(
                1 for stats in problem_stats.values() if stats["success_rate"] > 0
            ),
            "average_attempts_per_problem": (
                statistics.mean([stats["attempts"] for stats in problem_stats.values()])
                if problem_stats
                else 0
            ),
            "per_problem_stats": problem_stats,
        }

    def _analyze_retry_distribution(self, retry_counts: List[int]) -> Dict[str, Any]:
        """Analyze the distribution of retry counts."""
        from collections import Counter

        if not retry_counts:
            return {}

        counter = Counter(retry_counts)
        total = len(retry_counts)

        distribution = {}
        for retry_count in sorted(counter.keys()):
            count = counter[retry_count]
            percentage = (count / total) * 100
            distribution[f"{retry_count}_retries"] = {
                "count": count,
                "percentage": percentage,
            }

        return {
            "distribution": distribution,
            "most_common_retry_count": counter.most_common(1)[0] if counter else None,
            "unique_retry_counts": len(counter),
        }
