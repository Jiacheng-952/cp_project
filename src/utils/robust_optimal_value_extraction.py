# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Robust Optimal Value Extraction Module

This module provides utilities for extracting optimal values from various text
formats, specifically designed for the simplified OptAgent framework.
"""

import re
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def extract_optimal_value(text: str) -> Optional[float]:
    """
    Extract optimal value from text using prioritized pattern matching.

    This function is specifically designed for extracting optimal values from
    verifier outputs in the simplified framework.

    Args:
        text: Text content to analyze

    Returns:
        Extracted optimal value or None if not found
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Priority-ordered patterns for extracting optimization results
    patterns = [
        # HIGHEST PRIORITY: Explicit optimal value markers in verification reports
        r"\*\*OPTIMAL VALUE:\*\*\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        r"OPTIMAL VALUE:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        r"\*\*Optimal Value:\*\*\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        r"Optimal Value:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        # HIGH PRIORITY: Direct optimal value assignments from execution results
        r"optimal[_\s]*value[_\s]*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        r"objective[_\s]*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        r"result[_\s]*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        # MEDIUM PRIORITY: Object attribute patterns from code execution
        r"model\.ObjVal\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        r"Best\s+objective\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        r"Optimal\s+objective\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        # LOWER PRIORITY: Generic patterns
        r"Final\s+Result:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        r"Solution:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
    ]

    # Patterns to EXCLUDE (avoid extracting parameter assignments)
    exclusion_patterns = [
        r"min_\w+\s*=\s*([+-]?\d*\.?\d+)",  # min_full_time = 10
        r"max_\w+\s*=\s*([+-]?\d*\.?\d+)",  # max_points = 200
        r"\w+_per_\w+\s*=\s*([+-]?\d*\.?\d+)",  # points_per_seasonal = 2
        r"\w+_ratio\s*=\s*([+-]?\d*\.?\d+)",  # max_seasonal_ratio = 0.3
        r"#.*=.*([+-]?\d*\.?\d+)",  # Comments with assignments
    ]

    # Check for infeasible solutions first
    infeasible_patterns = [
        r"Model is infeasible",
        r"Status.*INFEASIBLE",
        r"No feasible solution",
        r"Problem is infeasible",
        r"INFEASIBLE",
        r"Status.*UNBOUNDED",
        r"Problem is unbounded",
        r"UNBOUNDED",
        r"optimal_value = INFEASIBLE",
        r"optimal_value = UNBOUNDED",
    ]

    for pattern in infeasible_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.info(
                f"Detected infeasible/unbounded solution with pattern: {pattern}"
            )
            return None

    # Try each pattern in priority order
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)

        if matches:
            # Check if any matches should be excluded
            valid_matches = []
            for match in matches:
                # Convert match to context for exclusion checking
                match_pos = text.find(str(match))
                match_context = text[max(0, match_pos - 50) : match_pos + 50]

                # Check exclusion patterns
                should_exclude = False
                for exclusion_pattern in exclusion_patterns:
                    if re.search(exclusion_pattern, match_context, re.IGNORECASE):
                        should_exclude = True
                        break

                if not should_exclude:
                    valid_matches.append(match)

            if valid_matches:
                try:
                    # Use the first valid match
                    value = float(valid_matches[0])
                    logger.debug(
                        f"Extracted optimal value {value} using pattern: {pattern}"
                    )
                    return value
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"Failed to convert match '{valid_matches[0]}' to float: {e}"
                    )
                    continue

    logger.debug("No optimal value found in text")
    return None


def extract_optimal_value_from_execution_result(
    execution_result: dict,
) -> Optional[float]:
    """
    Extract optimal value from execution result dictionary.

    Args:
        execution_result: Dictionary containing execution results

    Returns:
        Extracted optimal value or None if not found
    """
    if not isinstance(execution_result, dict):
        return None

    # Method 1: Direct extraction from execution result fields
    if execution_result.get("optimal_value") is not None:
        try:
            value = float(execution_result["optimal_value"])
            logger.debug(
                f"Extracted optimal value from execution result field: {value}"
            )
            return value
        except (ValueError, TypeError):
            logger.warning(
                f"Could not convert execution optimal_value to float: {execution_result['optimal_value']}"
            )

    # Method 2: Extract from stdout
    stdout = execution_result.get("stdout", "")
    if isinstance(stdout, str) and stdout:
        extracted_value = extract_optimal_value(stdout)
        if extracted_value is not None:
            logger.debug(
                f"Extracted optimal value from execution stdout: {extracted_value}"
            )
            return extracted_value

    # Method 3: Extract from stderr (sometimes contains solver output)
    stderr = execution_result.get("stderr", "")
    if isinstance(stderr, str) and stderr:
        extracted_value = extract_optimal_value(stderr)
        if extracted_value is not None:
            logger.debug(
                f"Extracted optimal value from execution stderr: {extracted_value}"
            )
            return extracted_value

    logger.debug("No optimal value found in execution result")
    return None


def extract_optimal_value_from_mixed_content(
    content: Union[str, dict, list]
) -> Optional[float]:
    """
    Extract optimal value from mixed content types.

    This function handles various content types that might contain optimal values,
    including strings, dictionaries, and lists.

    Args:
        content: Content to analyze (string, dict, or list)

    Returns:
        Extracted optimal value or None if not found
    """
    if content is None:
        return None

    # Handle string content
    if isinstance(content, str):
        return extract_optimal_value(content)

    # Handle dictionary content
    if isinstance(content, dict):
        # Try direct optimal_value field first
        if "optimal_value" in content:
            try:
                return float(content["optimal_value"])
            except (ValueError, TypeError):
                pass

        # Try execution result format
        if "stdout" in content or "stderr" in content:
            return extract_optimal_value_from_execution_result(content)

        # Try extracting from all string values in the dict
        for key, value in content.items():
            if isinstance(value, str):
                extracted_value = extract_optimal_value(value)
                if extracted_value is not None:
                    logger.debug(
                        f"Extracted optimal value from dict field '{key}': {extracted_value}"
                    )
                    return extracted_value

    # Handle list content
    if isinstance(content, list):
        for item in content:
            extracted_value = extract_optimal_value_from_mixed_content(item)
            if extracted_value is not None:
                return extracted_value

    # Try converting to string as last resort
    try:
        text_content = str(content)
        return extract_optimal_value(text_content)
    except:
        pass

    return None
