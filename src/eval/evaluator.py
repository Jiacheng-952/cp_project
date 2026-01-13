# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
OptAgent Benchmark Evaluator Module

This module provides the evaluation interface for the 3-node OptAgent
framework. It coordinates the evaluation process using the new graph
structure and provides optimal value extraction from verifier outputs.
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING

from .config import EvalConfig

# BenchmarkRunner functionality is now integrated into the evaluator

if TYPE_CHECKING:
    from src.config.timeout_config import TimeoutConfig

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Main evaluator class for running OptAgent benchmark tests.

    This class provides a high-level interface for configuring and executing
    benchmark evaluations using the 3-node framework.

    Attributes:
        config: EvalConfig instance containing all evaluation parameters

    Example Usage:
        # Basic evaluation
        evaluator = BenchmarkEvaluator("benchmark.jsonl")
        results = evaluator.run()

        # Advanced configuration
        evaluator = BenchmarkEvaluator(
            benchmark_file="complex_benchmark.jsonl",
            concurrency=8,
            pass_n=3,
            timeout=600,
            enable_batch_mode=True
        )
        results = evaluator.run()
    """

    def __init__(
        self,
        benchmark_file: str,
        output_dir: str = "eval_results",
        concurrency: int = 4,
        pass_n: int = 1,
        tolerance: float = 1e-6,
        timeout: int = 300,
        model_name: Optional[str] = None,
        timeout_config: Optional["TimeoutConfig"] = None,
        enable_batch_mode: bool = False,
        enable_visualization: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize benchmark evaluator.

        Args:
            benchmark_file: Path to JSONL benchmark file containing problems
            output_dir: Directory to save evaluation results (default: "eval_results")
            concurrency: Number of concurrent problem evaluations (default: 4)
            pass_n: Number of attempts per problem for pass@n evaluation (default: 1)
            tolerance: Relative error tolerance for answer correctness (default: 1e-6)
            timeout: Timeout per problem in seconds (default: 300)
            model_name: Identifier of the model being evaluated (optional)
            timeout_config: Custom timeout configuration object (optional)
            enable_batch_mode: Enable optimized batch evaluation mode (default: False)
            enable_visualization: Enable visualization generation after successful modeling (default: False)
            **kwargs: Additional configuration parameters passed to EvalConfig
        """
        # Create or configure timeout settings
        if timeout_config is None:
            from src.config.timeout_config import TimeoutConfig

            timeout_config = TimeoutConfig(
                single_problem_timeout=timeout,
                llm_request_timeout=120,
                code_execution_timeout=60,
                agent_recursion_limit=25,
                max_retries=2,
                retry_delay=1.0,
            )

        # Create evaluation configuration
        self.config = EvalConfig(
            benchmark_file=benchmark_file,
            output_dir=output_dir,
            concurrency=concurrency,
            pass_n=pass_n,
            tolerance=tolerance,
            timeout=timeout,
            model_name=model_name,
            timeout_config=timeout_config,
            enable_batch_mode=enable_batch_mode,
            enable_visualization=enable_visualization,
            **kwargs,
        )

        logger.info(f"Initialized BenchmarkEvaluator for {benchmark_file}")
        logger.info(
            f"Configuration: concurrency={concurrency}, pass_n={pass_n}, timeout={timeout}s"
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute the benchmark evaluation.

        This method coordinates the complete evaluation process including:
        1. Loading benchmark problems
        2. Running concurrent evaluations using the graph
        3. Computing metrics and generating reports
        4. Saving results to the output directory

        Returns:
            Dict containing comprehensive evaluation results including:
            - summary: Overall performance metrics
            - problems: Individual problem results
            - config: Evaluation configuration used
            - timing: Execution timing information

        Raises:
            FileNotFoundError: If benchmark file doesn't exist
            RuntimeError: If evaluation encounters critical errors
        """
        logger.info("Starting benchmark evaluation")

        try:
            # Load and run benchmark directly
            from .result_extractor import ResultExtractor
            from .metrics_calculator import MetricsCalculator
            from .trace_manager import TraceManager
            import json
            import time

            start_time = time.time()
            result_extractor = ResultExtractor()
            metrics_calculator = MetricsCalculator(tolerance=self.config.tolerance)

            # Initialize trace manager
            from pathlib import Path

            benchmark_name = Path(self.config.benchmark_file).stem

            # Use the run directory from environment if available (set by sh script)
            import os

            run_dir = os.environ.get("TASK_DIR")
            if run_dir:
                # Use the pre-created directory from sh script
                trace_manager = TraceManager(
                    run_dir=run_dir,
                    benchmark_name=benchmark_name,
                    model_name=self.config.model_name or "unknown",
                )
            else:
                # Fallback: create directory ourselves (for direct Python execution)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fallback_run_dir = (
                    Path(self.config.output_dir)
                    / benchmark_name
                    / (self.config.model_name or "unknown")
                    / timestamp
                )
                trace_manager = TraceManager(
                    run_dir=str(fallback_run_dir),
                    benchmark_name=benchmark_name,
                    model_name=self.config.model_name or "unknown",
                )

            # Save config
            trace_manager.save_config(self._config_to_dict())

            # Load benchmark problems
            problems = self._load_benchmark_data()

            if not problems:
                raise ValueError("No valid problems found in benchmark file")

            # Run evaluations
            results = self._run_evaluations(problems, result_extractor, trace_manager)

            # Pass timing information to metrics calculator for better statistics
            metrics_calculator._wall_clock_time = getattr(
                self, "_last_wall_clock_time", 0
            )
            metrics_calculator._concurrency = self.config.concurrency

            # Calculate metrics
            metrics = metrics_calculator.generate_summary_report(
                results, [self.config.pass_n]
            )

            # Save summary
            output_data = {
                "summary": metrics,
                "problems": results,
                "config": self._config_to_dict(),
                "timing": {
                    "total_time": time.time() - start_time,
                    "problems_count": len(problems),
                },
            }

            trace_manager.save_summary(output_data)

            logger.info("Benchmark evaluation completed successfully")
            return output_data

        except Exception as e:
            logger.error(f"Benchmark evaluation failed: {e}")
            raise RuntimeError(f"Evaluation failed: {e}") from e

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BenchmarkEvaluator":
        """
        Create evaluator instance from command line arguments.

        This class method provides a convenient way to initialize the evaluator
        from parsed command line arguments, handling type conversions and
        default value assignments.

        Args:
            args: Parsed command line arguments from argparse

        Returns:
            BenchmarkEvaluator: Configured evaluator instance

        Example:
            parser = argparse.ArgumentParser()
            parser.add_argument("--benchmark_file", required=True)
            # ... add other arguments
            args = parser.parse_args()
            evaluator = BenchmarkEvaluator.from_args(args)
        """
        # Extract standard parameters
        standard_params = {
            "benchmark_file": args.benchmark_file,
            "output_dir": getattr(args, "output_dir", "eval_results"),
            "concurrency": getattr(args, "concurrency", 4),
            "pass_n": getattr(args, "pass_n", 1),
            "tolerance": getattr(args, "tolerance", 1e-6),
            "timeout": getattr(args, "timeout", 300),
            "model_name": getattr(args, "model_name", None),
            "enable_batch_mode": getattr(args, "enable_batch_mode", False),
        }

        # Create timeout configuration if parameters are provided
        timeout_config = None
        if hasattr(args, "llm_timeout") or hasattr(args, "code_timeout"):
            from src.config.timeout_config import TimeoutConfig

            timeout_config = TimeoutConfig(
                single_problem_timeout=getattr(args, "timeout", 300),
                llm_request_timeout=getattr(args, "llm_timeout", 120),
                code_execution_timeout=getattr(args, "code_timeout", 60),
                agent_recursion_limit=getattr(args, "recursion_limit", 25),
                max_retries=getattr(args, "max_retries", 2),
                retry_delay=getattr(args, "retry_delay", 1.0),
            )
            standard_params["timeout_config"] = timeout_config

        # Only add specific additional parameters that EvalConfig supports
        if hasattr(args, "max_corrections"):
            standard_params["max_corrections"] = getattr(args, "max_corrections", 5)
        if hasattr(args, "debug"):
            standard_params["debug"] = getattr(args, "debug", False)
        if hasattr(args, "enable_visualization"):
            standard_params["enable_visualization"] = getattr(
                args, "enable_visualization", False
            )

        return cls(**standard_params)

    def _config_to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "benchmark_file": self.config.benchmark_file,
            "concurrency": self.config.concurrency,
            "pass_n": self.config.pass_n,
            "tolerance": self.config.tolerance,
            "timeout": self.config.timeout,
            "max_corrections": getattr(self.config, "max_corrections", 5),
            "debug": self.config.debug,
            "model_name": self.config.model_name,
            "enable_visualization": getattr(self.config, "enable_visualization", False),
            "framework": "3_node",
        }

    def _load_benchmark_data(self) -> list:
        """Load benchmark data from JSONL file."""
        problems = []

        try:
            import json

            with open(self.config.benchmark_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        problem = json.loads(line)

                        # Handle different field name formats
                        question_field = None
                        answer_field = None

                        # Check for question field variants
                        if "question" in problem:
                            question_field = "question"
                        elif "en_question" in problem:
                            question_field = "en_question"
                        elif "problem" in problem:
                            question_field = "problem"
                        elif "statement" in problem:
                            question_field = "statement"

                        # Check for answer field variants
                        if "answer" in problem:
                            answer_field = "answer"
                        elif "en_answer" in problem:
                            answer_field = "en_answer"
                        elif "optimal_value" in problem:
                            answer_field = "optimal_value"
                        elif "result" in problem:
                            answer_field = "result"

                        if not question_field or not answer_field:
                            logger.warning(
                                f"Line {line_num}: Missing required fields. Available: {list(problem.keys())}"
                            )
                            continue

                        # Handle data files (for separated problem description and data)
                        data_files = problem.get("data_files", [])
                        if isinstance(data_files, str):
                            data_files = [data_files]  # Convert single file to list

                        # Convert relative paths to absolute paths based on benchmark file location
                        resolved_data_files = []
                        if data_files:
                            import os

                            benchmark_dir = os.path.dirname(
                                os.path.abspath(self.config.benchmark_file)
                            )
                            for data_file in data_files:
                                if os.path.isabs(data_file):
                                    resolved_data_files.append(data_file)
                                else:
                                    resolved_data_files.append(
                                        os.path.join(benchmark_dir, data_file)
                                    )

                        # Normalize the problem structure
                        normalized_problem = {
                            "id": problem.get("id", f"problem_{line_num}"),
                            "question": problem[question_field],
                            "answer": problem[answer_field],
                            "data_files": resolved_data_files,  # Add data files
                            "original": problem,  # Keep original for reference
                        }

                        problems.append(normalized_problem)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue

        except FileNotFoundError:
            logger.error(f"Benchmark file not found: {self.config.benchmark_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading benchmark file: {e}")
            raise

        logger.info(
            f"Loaded {len(problems)} problems from {self.config.benchmark_file}"
        )
        return problems

    def _run_evaluations(self, problems: list, result_extractor, trace_manager) -> list:
        """Run evaluations on all problems with concurrent execution."""
        import asyncio
        import time

        start_time = time.time()
        total_problems = len(problems)
        total_attempts = total_problems * self.config.pass_n

        logger.info(
            f"🚀 Starting concurrent evaluation of {total_problems} problems with {self.config.concurrency} concurrent workers"
        )
        logger.info(
            f"📊 Total attempts: {total_attempts} (pass_n={self.config.pass_n})"
        )

        # Run concurrent evaluation
        results = asyncio.run(
            self._run_concurrent_evaluations(problems, result_extractor, trace_manager)
        )

        # Calculate timing statistics
        wall_clock_time = time.time() - start_time  # Real elapsed time (wall clock)

        # Calculate cumulative execution time from individual results
        cumulative_execution_time = sum(r.get("execution_time", 0) for r in results)

        avg_time_per_problem = (
            wall_clock_time / total_problems if total_problems > 0 else 0
        )
        avg_time_per_attempt = (
            wall_clock_time / total_attempts if total_attempts > 0 else 0
        )

        # Calculate actual speedup achieved
        actual_speedup = (
            cumulative_execution_time / wall_clock_time if wall_clock_time > 0 else 1
        )

        logger.info(f"⏱️  Wall Clock Time: {wall_clock_time:.2f} seconds")
        logger.info(
            f"⏰ Cumulative Execution Time: {cumulative_execution_time:.2f} seconds"
        )
        logger.info(
            f"📈 Average wall clock time per problem: {avg_time_per_problem:.2f}s"
        )
        logger.info(
            f"📈 Average wall clock time per attempt: {avg_time_per_attempt:.2f}s"
        )
        logger.info(
            f"🚀 Actual concurrency speedup: {actual_speedup:.1f}x (vs theoretical {self.config.concurrency}x)"
        )
        logger.info(
            f"⚡ Parallel efficiency: {(actual_speedup/self.config.concurrency)*100:.1f}%"
        )

        # Store wall clock time for metrics calculation
        self._last_wall_clock_time = wall_clock_time

        return results

    async def _run_concurrent_evaluations(
        self, problems: list, result_extractor, trace_manager
    ) -> list:
        """Run concurrent evaluations using asyncio with semaphore for rate limiting."""
        import asyncio

        semaphore = asyncio.Semaphore(self.config.concurrency)
        results = []

        async def process_problem_with_attempts(problem_idx: int, problem: dict):
            """Process a single problem with all its attempts."""
            problem_id = problem.get("id", f"problem_{problem_idx+1}")
            problem_results = []

            # Log problem start
            logger.info(f"📝 [{problem_idx+1}/{len(problems)}] Starting: {problem_id}")
            logger.debug(
                f"Problem statement: {problem.get('en_question', problem.get('question', 'No question found'))[:100]}..."
            )
            logger.debug(
                f"Expected answer: {problem.get('en_answer', problem.get('answer', 'No answer found'))}"
            )

            try:
                # Process all attempts for this problem
                for attempt in range(self.config.pass_n):
                    async with semaphore:  # Rate limiting
                        try:
                            result = await self._evaluate_single_problem(
                                problem, attempt, result_extractor, trace_manager
                            )
                            problem_results.append(result)
                            logger.debug(
                                f"✅ [{problem_idx+1}/{len(problems)}] Attempt {attempt+1} completed: {result.get('status', 'unknown')}"
                            )
                        except Exception as e:
                            logger.error(
                                f"❌ [{problem_idx+1}/{len(problems)}] Attempt {attempt+1} failed: {e}"
                            )
                            error_result = {
                                "problem_id": problem.get(
                                    "id", f"problem_{problem_idx+1}"
                                ),
                                "attempt": attempt,
                                "status": "error",
                                "correct": False,
                                "error": str(e),
                                "execution_time": 0.0,
                                "optimal_solution_variables": None,
                            }
                            problem_results.append(error_result)
                            # Save trace for error case
                            trace_manager.save_trace(error_result)

                logger.info(
                    f"🎯 [{problem_idx+1}/{len(problems)}] Completed: {problem_id} ({len(problem_results)} attempts)"
                )

            except Exception as e:
                logger.error(
                    f"💥 [{problem_idx+1}/{len(problems)}] Problem {problem_id} failed completely: {e}"
                )
                error_result = {
                    "problem_id": problem.get("id", f"problem_{problem_idx+1}"),
                    "attempt": 0,
                    "status": "error",
                    "correct": False,
                    "error": str(e),
                    "execution_time": 0.0,
                    "optimal_solution_variables": None,
                }
                problem_results.append(error_result)
                trace_manager.save_trace(error_result)

            return problem_results

        # Create tasks for all problems
        tasks = [
            process_problem_with_attempts(i, problem)
            for i, problem in enumerate(problems)
        ]

        # Execute all tasks concurrently
        logger.info(
            f"🔄 Executing {len(tasks)} problem tasks with max {self.config.concurrency} concurrent workers..."
        )
        all_problem_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        for i, problem_results in enumerate(all_problem_results):
            if isinstance(problem_results, Exception):
                logger.error(
                    f"💥 Problem {i+1} failed with exception: {problem_results}"
                )
                # Create error result for failed problem
                error_result = {
                    "problem_id": problems[i].get("id", f"problem_{i+1}"),
                    "attempt": 0,
                    "status": "error",
                    "correct": False,
                    "error": str(problem_results),
                    "execution_time": 0.0,
                    "optimal_solution_variables": None,
                }
                results.append(error_result)
            else:
                # Add all attempts for this problem
                results.extend(problem_results)

        return results

    async def _evaluate_single_problem(
        self, problem: dict, attempt: int, result_extractor, trace_manager
    ):
        """Evaluate a single problem."""
        import time
        import asyncio

        start_time = time.time()

        # Track retry attempts for this problem
        retry_count = 0

        try:
            # Import graph components
            from src.graph.builder import build_optag_graph
            from src.graph.types import create_optag_state

            # Create initial state
            initial_state = create_optag_state(
                problem_statement=problem["question"],
                problem_id=problem["id"],
                current_attempt=attempt,
                max_corrections=getattr(self.config, "max_corrections", 5),
                data_files=problem.get("data_files", []),  # Add data files to state
                enable_visualization=getattr(
                    self.config, "enable_visualization", False
                ),
                output_dir=(
                    trace_manager.run_dir
                    if hasattr(trace_manager, "run_dir")
                    else self.config.output_dir
                ),
            )

            # Build and run graph
            graph = build_optag_graph()

            # Execute the workflow
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state), timeout=self.config.timeout
            )

            # Extract results
            extracted_result = result_extractor.extract_optimal_value_from_state(
                final_state
            )

            # Extract optimal solution variables
            optimal_solution_variables = (
                result_extractor.extract_optimal_solution_from_state(final_state)
            )

            # Determine success
            is_correct = False
            if extracted_result is not None:
                try:
                    expected_answer_str = problem.get(
                        "en_answer", problem.get("answer", "")
                    )
                    expected_answer = float(expected_answer_str)
                    from .metrics_calculator import MetricsCalculator

                    metrics_calc = MetricsCalculator(tolerance=self.config.tolerance)
                    is_correct = metrics_calc.is_correct(
                        extracted_result, expected_answer
                    )
                    logger.info(
                        f"Problem {problem.get('id', 'unknown')}: extracted={extracted_result}, expected={expected_answer}, correct={is_correct}, tolerance={self.config.tolerance}"
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Could not convert expected answer to float: {expected_answer_str}, error: {e}"
                    )

            execution_time = time.time() - start_time

            # Get retry count from final state
            retry_count = final_state.get("correction_count", 0)

            # Build result
            result = {
                "problem_id": problem["id"],
                "attempt": attempt,
                "status": (
                    "success"
                    if final_state.get("solution_complete", False)
                    else "failed"
                ),
                "correct": is_correct,
                "extracted_value": extracted_result,
                "expected_value": problem.get("en_answer", problem.get("answer", "")),
                "execution_time": execution_time,
                "solution_complete": final_state.get("solution_complete", False),
                "verification_passed": final_state.get("verification_passed", False),
                "correction_count": retry_count,
                "retry_count": retry_count,  # Add explicit retry_count field
                "optimal_solution_variables": optimal_solution_variables,  # Add decision variables
            }

            # Save trace with detailed execution information (JSON serializable)
            try:
                # Convert AIMessage objects to strings for JSON serialization
                messages = final_state.get("messages", [])
                serializable_messages = []
                for msg in messages:
                    if hasattr(msg, "content"):
                        serializable_messages.append(
                            {"content": str(msg.content), "type": type(msg).__name__}
                        )
                    elif isinstance(msg, dict):
                        serializable_messages.append(msg)
                    else:
                        serializable_messages.append(str(msg))

                trace_data = {
                    **result,
                    "problem_statement": problem.get(
                        "en_question", problem.get("question", "")
                    ),
                    "current_model": final_state.get("current_model", ""),
                    "current_code": final_state.get("current_code", ""),
                    "verification_result": final_state.get("verification_result", ""),
                    "optimal_value": final_state.get("optimal_value"),
                    "messages": serializable_messages,
                    "verification_history": final_state.get("verification_history", []),
                    "correction_history": final_state.get("correction_history", []),
                    # optimal_solution_variables is already included via **result spread
                }
                trace_manager.save_trace(trace_data)
            except Exception as trace_error:
                logger.warning(f"Failed to save detailed trace: {trace_error}")
                # Save minimal trace
                minimal_trace = {
                    **result,
                    "problem_statement": problem.get(
                        "en_question", problem.get("question", "")
                    ),
                    # optimal_solution_variables is already included via **result spread
                }
                trace_manager.save_trace(minimal_trace)

            return result

        except asyncio.TimeoutError:
            return {
                "problem_id": problem["id"],
                "attempt": attempt,
                "status": "timeout",
                "correct": False,
                "execution_time": time.time() - start_time,
                "error": "Evaluation timeout",
                "optimal_solution_variables": None,
            }
        except Exception as e:
            return {
                "problem_id": problem["id"],
                "attempt": attempt,
                "status": "error",
                "correct": False,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "optimal_solution_variables": None,
            }


def main():
    """
    Command line interface for benchmark evaluation.

    This function provides a comprehensive CLI for running benchmark evaluations
    with the 3-node OptAgent framework. It handles argument parsing,
    logging setup, and result reporting.

    Usage:
        python -m src.eval.evaluator benchmark.jsonl
        python -m src.eval.evaluator benchmark.jsonl --concurrency 8 --timeout 600
    """
    parser = argparse.ArgumentParser(
        description="Run OptAgent benchmark evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("benchmark_file", help="Path to JSONL benchmark file")

    # Basic configuration
    parser.add_argument(
        "--output_dir", default="eval_results", help="Output directory for results"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Number of concurrent evaluations"
    )
    parser.add_argument(
        "--pass_n", type=int, default=1, help="Number of attempts per problem"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Relative error tolerance for correctness",
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Total timeout per problem (seconds)"
    )

    # Optional configuration
    parser.add_argument("--model_name", help="Model identifier for results")
    parser.add_argument(
        "--max_corrections",
        type=int,
        default=5,
        help="Maximum correction attempts per problem",
    )
    parser.add_argument(
        "--enable_batch_mode",
        action="store_true",
        help="Enable batch evaluation optimizations",
    )
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable visualization generation after successful modeling",
    )

    # Timeout configuration
    parser.add_argument(
        "--llm_timeout", type=int, default=120, help="LLM request timeout (seconds)"
    )
    parser.add_argument(
        "--code_timeout", type=int, default=60, help="Code execution timeout (seconds)"
    )
    parser.add_argument(
        "--recursion_limit", type=int, default=25, help="Agent recursion limit"
    )
    parser.add_argument(
        "--max_retries", type=int, default=2, help="Maximum retry attempts"
    )
    parser.add_argument(
        "--retry_delay", type=float, default=1.0, help="Delay between retries (seconds)"
    )

    # Debug and logging
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate benchmark file exists
    if not os.path.exists(args.benchmark_file):
        logger.error(f"Benchmark file not found: {args.benchmark_file}")
        sys.exit(1)

    try:
        # Create and run evaluator
        evaluator = BenchmarkEvaluator.from_args(args)
        results = evaluator.run()

        # Report results
        summary_data = results.get("summary", {})
        pass_at_k = summary_data.get("pass_at_k", {})
        pass_at_1 = pass_at_k.get("pass@1", 0)
        accuracy = summary_data.get("overview", {}).get("accuracy", 0)
        logger.info(
            f"Evaluation completed: {accuracy:.1%} pass rate (pass@1: {pass_at_1:.1%})"
        )
        logger.info(f"Results saved to: {evaluator.config.output_dir}")

        print("\n=== EVALUATION SUMMARY ===")
        print(f"📊 Pass Rate: {accuracy:.1%}")
        print(f"🎯 Pass@1: {pass_at_1:.1%}")
        overview = summary_data.get("overview", {})
        successful_attempts = overview.get("successful_attempts", 0)
        total_attempts = overview.get("total_attempts", 0)
        print(f"✅ Problems Solved: {successful_attempts}/{total_attempts}")
        timing_stats = summary_data.get("timing_statistics", {})
        avg_time = timing_stats.get("mean_time", 0)
        wall_clock_time = timing_stats.get(
            "wall_clock_time", timing_stats.get("total_time", 0)
        )
        cumulative_time = timing_stats.get("cumulative_execution_time", 0)
        actual_speedup = timing_stats.get("actual_speedup", 1)
        parallel_efficiency = timing_stats.get("parallel_efficiency", 0)

        print(f"⏱️  Wall Clock Time: {wall_clock_time:.1f}s")
        print(f"⏰ Cumulative Execution Time: {cumulative_time:.1f}s")
        print(f"📈 Average Time per Problem: {avg_time:.1f}s")
        print(f"🚀 Concurrency: {evaluator.config.concurrency} workers")
        print(
            f"⚡ Speedup: {actual_speedup:.1f}x (efficiency: {parallel_efficiency:.1f}%)"
        )

        # Display retry statistics
        retry_stats = summary_data.get("retry_statistics", {})
        if retry_stats:
            mean_retries = retry_stats.get("mean_retries", 0)
            max_retries = retry_stats.get("max_retries", 0)
            min_retries = retry_stats.get("min_retries", 0)
            total_retries = retry_stats.get("total_retries", 0)
            problems_with_retries = retry_stats.get("problems_with_retries", 0)
            problems_without_retries = retry_stats.get("problems_without_retries", 0)

            print(f"🔄 Retry Statistics:")
            print(f"   📊 Average Retries: {mean_retries:.2f}")
            print(f"   📈 Max Retries: {max_retries}")
            print(f"   📉 Min Retries: {min_retries}")
            print(f"   🔢 Total Retries: {total_retries}")
            print(f"   ✅ Problems with Retries: {problems_with_retries}")
            print(f"   🎯 Problems without Retries: {problems_without_retries}")

        print(f"📂 Results Directory: {evaluator.config.output_dir}")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
