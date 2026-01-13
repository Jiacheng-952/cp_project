# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Trace Management Module

This module provides comprehensive trace management for evaluation runs,
including saving detailed execution traces, managing result files, and
exporting data in various formats.

The TraceManager class handles:
- Detailed trace creation with execution metadata
- File system organization for evaluation results
- JSON and CSV export functionality  
- Configuration and summary management
- Result archival and retrieval

Traces include complete information about each problem evaluation including
input parameters, execution results, timing data, and any errors encountered.
This enables detailed analysis and debugging of evaluation runs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TraceManager:
    """Manages evaluation traces and results."""

    def __init__(self, run_dir: str, benchmark_name: str, model_name: str = "unknown"):
        """
        Initialize trace manager.

        Args:
            run_dir: Pre-created run directory path
            benchmark_name: Name of the benchmark
            model_name: Name of the model being evaluated
        """
        self.benchmark_name = benchmark_name
        self.model_name = model_name

        # Use the pre-created run directory
        self.run_dir = Path(run_dir)
        # Ensure the directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize file paths
        self.traces_file = self.run_dir / "traces.json"
        self.results_file = self.run_dir / "results.json"
        self.failed_cases_file = self.run_dir / "failed_cases.json"
        self.config_file = self.run_dir / "config.json"
        self.summary_file = self.run_dir / "summary.json"

        # Initialize trace counter
        self.trace_count = 0

        logger.info(f"Created evaluation run directory: {self.run_dir}")

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save evaluation configuration.

        Args:
            config: Configuration dictionary
        """
        try:
            config_data = {
                "benchmark_name": self.benchmark_name,
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "config": config,
            }

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved configuration to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def save_trace(self, trace_data: Dict[str, Any]) -> None:
        """
        Save a single evaluation trace.

        Args:
            trace_data: Trace data dictionary
        """
        try:
            # Add trace metadata
            trace_data.update(
                {
                    "trace_id": self.trace_count,
                    "timestamp": datetime.now().isoformat(),
                    "benchmark": self.benchmark_name,
                    "model": self.model_name,
                }
            )

            # Load existing traces or create new list
            traces = []
            if self.traces_file.exists():
                try:
                    with open(self.traces_file, "r", encoding="utf-8") as f:
                        traces = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    traces = []

            # Add new trace
            traces.append(trace_data)

            # Save updated traces file
            with open(self.traces_file, "w", encoding="utf-8") as f:
                json.dump(traces, f, indent=2, ensure_ascii=False)

            # If this is a failed case, also save to failed cases file
            if not trace_data.get("correct", False):
                failed_cases = []
                if self.failed_cases_file.exists():
                    try:
                        with open(self.failed_cases_file, "r", encoding="utf-8") as f:
                            failed_cases = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        failed_cases = []

                failed_cases.append(trace_data)

                with open(self.failed_cases_file, "w", encoding="utf-8") as f:
                    json.dump(failed_cases, f, indent=2, ensure_ascii=False)

            self.trace_count += 1

        except Exception as e:
            logger.error(f"Failed to save trace: {e}")

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """
        Save evaluation summary.

        Args:
            summary: Summary data dictionary
        """
        try:
            summary_data = {
                "benchmark_name": self.benchmark_name,
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "total_traces": self.trace_count,
                "run_directory": str(self.run_dir),
                "summary": summary,
            }

            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved summary to {self.summary_file}")

        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
