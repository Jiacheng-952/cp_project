# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Evaluation Configuration Module

This module defines configuration classes and settings for the OptAgent
evaluation system. It provides comprehensive configuration management for
benchmark testing, including concurrency, timeouts, and optimization parameters.

The main configuration class EvalConfig encapsulates all settings needed
to run benchmark evaluations with proper validation and defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from src.config.timeout_config import get_timeout_config, TimeoutConfig


@dataclass
class EvalConfig:
    """Configuration for the evaluation process."""

    benchmark_file: str
    output_dir: str = "eval_results"
    concurrency: int = 4
    pass_n: int = 1
    tolerance: float = 1e-6
    timeout: int = 300
    model_name: Optional[str] = None
    max_corrections: int = 5
    debug: bool = False
    enable_batch_mode: bool = False
    enable_visualization: bool = False
    failure_log: Optional[str] = None
    timeout_config: TimeoutConfig = field(default_factory=get_timeout_config)

    def __post_init__(self):
        """Validate and create directories after initialization."""
        # Validate input file exists
        if not os.path.exists(self.benchmark_file):
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_file}")

        # Validate numeric parameters
        if self.concurrency < 1:
            raise ValueError("Concurrency must be at least 1")

        if self.pass_n < 1:
            raise ValueError("pass_n must be at least 1")

        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")

        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")

        if self.max_corrections < 0:
            raise ValueError("max_corrections must be non-negative")

        # Initialize timeout configuration if not provided
        if self.timeout_config is None:
            self.timeout_config = TimeoutConfig()
            # Use legacy timeout value to update single_problem_timeout
            self.timeout_config.single_problem_timeout = self.timeout

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
