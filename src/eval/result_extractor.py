# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Result Extraction Module

This module provides utilities for extracting numerical results and optimal
values from the OptAgent workflow outputs. It handles the new
verifier output format and extracts values from the comprehensive verification
reports.
"""

import re
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ResultExtractor:
    """Extracts numerical results from agent outputs."""

    def __init__(self):
        # Priority-ordered patterns for extracting optimization results from verifier output
        self.result_patterns = [
            # HIGHEST PRIORITY: Explicit optimal value in verification reports
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
        self.exclusion_patterns = [
            r"min_\w+\s*=\s*([+-]?\d*\.?\d+)",  # min_full_time = 10
            r"max_\w+\s*=\s*([+-]?\d*\.?\d+)",  # max_points = 200
            r"\w+_per_\w+\s*=\s*([+-]?\d*\.?\d+)",  # points_per_seasonal = 2
            r"\w+_ratio\s*=\s*([+-]?\d*\.?\d+)",  # max_seasonal_ratio = 0.3
            r"#.*=.*([+-]?\d*\.?\d+)",  # Comments with assignments
        ]

        # Patterns that indicate infeasible/unbounded solutions
        # These patterns should only match actual solver output, not code constants
        self.infeasible_patterns = [
            r"Model is infeasible",
            r"No feasible solution",
            r"Problem is infeasible",
            r"Problem is unbounded",
            r"optimal_value = INFEASIBLE",
            r"optimal_value = UNBOUNDED",
            # Match solver status outputs but not code constants
            r"Optimization status:\s*\d+.*INFEASIBLE",
            r"Optimization status:\s*\d+.*UNBOUNDED",
            # Only match standalone INFEASIBLE/UNBOUNDED outputs, not code constants
            r"^\s*INFEASIBLE\s*$",
            r"^\s*UNBOUNDED\s*$",
        ]

    def detect_infeasible_solution(self, text: str) -> bool:
        """
        Detect if the solution indicates an infeasible or unbounded problem.

        This method tries to distinguish between code containing infeasible patterns
        and actual solver output indicating infeasible solutions.

        Args:
            text: Text content to analyze

        Returns:
            True if the solution is infeasible/unbounded, False otherwise
        """
        if not isinstance(text, str):
            return False

        # If the text looks like code (contains common code patterns), be more careful
        code_indicators = [
            "import ",
            "def ",
            "if ",
            "for ",
            "print(",
            "model.",
            ".addVar",
            ".optimize()",
        ]
        looks_like_code = any(indicator in text for indicator in code_indicators)

        if looks_like_code:
            # For code text, only check very specific output patterns that are unlikely to be in code
            code_safe_patterns = [
                r"^\s*INFEASIBLE\s*$",
                r"^\s*UNBOUNDED\s*$",
                r"Model is infeasible",
                r"No feasible solution",
                r"Problem is infeasible",
                r"Problem is unbounded",
                # Only match actual printed output, not string literals
                r"^optimal_value = INFEASIBLE$",
                r"^optimal_value = UNBOUNDED$",
            ]

            for pattern in code_safe_patterns:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    logger.info(
                        f"Detected infeasible/unbounded solution in code with pattern: {pattern}"
                    )
                    return True
        else:
            # For non-code text (like solver output), use all patterns
            for pattern in self.infeasible_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.info(
                        f"Detected infeasible/unbounded solution with pattern: {pattern}"
                    )
                    return True

        return False

    def extract_optimal_value_from_text(self, text: str) -> Optional[float]:
        """
        Extract optimal value from text using prioritized pattern matching.

        Args:
            text: Text content to analyze

        Returns:
            Extracted optimal value or None if not found
        """
        if not isinstance(text, str) or not text.strip():
            return None

        # Check for infeasible solutions first
        if self.detect_infeasible_solution(text):
            logger.info("Solution is infeasible/unbounded, returning None")
            return None

        # Try each pattern in priority order
        for pattern in self.result_patterns:
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
                    for exclusion_pattern in self.exclusion_patterns:
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

    def extract_optimal_value_from_state(self, state: dict) -> Optional[float]:
        """
        Extract optimal value from OptAgent state.

        This method handles the state structure and extracts
        the optimal value from various possible locations.

        Args:
            state: Final state from OptAgent workflow

        Returns:
            Extracted optimal value or None if not found
        """
        if not isinstance(state, dict):
            logger.warning("State is not a dictionary")
            return None

        # Method 1: Direct extraction from state fields
        if state.get("optimal_value_extracted", False):
            optimal_value = state.get("optimal_value")
            if optimal_value is not None:
                try:
                    value = float(optimal_value)
                    logger.debug(f"Extracted optimal value from state field: {value}")
                    return value
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert state optimal_value to float: {optimal_value}"
                    )

        # Method 2: Extract from final solution
        final_solution = state.get("final_solution", {})
        if isinstance(final_solution, dict):
            # Check if optimal value is in final solution
            if final_solution.get("optimal_value_extracted", False):
                optimal_value = final_solution.get("optimal_value")
                if optimal_value is not None:
                    try:
                        value = float(optimal_value)
                        logger.debug(
                            f"Extracted optimal value from final solution: {value}"
                        )
                        return value
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Could not convert final solution optimal_value to float: {optimal_value}"
                        )

        # Method 3: Extract from verification result text
        verification_result = state.get("verification_result", "")
        if isinstance(verification_result, str) and verification_result:
            extracted_value = self.extract_optimal_value_from_text(verification_result)
            if extracted_value is not None:
                logger.debug(
                    f"Extracted optimal value from verification result: {extracted_value}"
                )
                return extracted_value

        # Method 4: Extract from final solution report text
        if isinstance(final_solution, dict):
            final_report = final_solution.get("final_report", "")
            if isinstance(final_report, str) and final_report:
                extracted_value = self.extract_optimal_value_from_text(final_report)
                if extracted_value is not None:
                    logger.debug(
                        f"Extracted optimal value from final report: {extracted_value}"
                    )
                    return extracted_value

            # Check execution stdout in final solution
            execution_stdout = final_solution.get("execution_stdout", "")
            if isinstance(execution_stdout, str) and execution_stdout:
                extracted_value = self.extract_optimal_value_from_text(execution_stdout)
                if extracted_value is not None:
                    logger.debug(
                        f"Extracted optimal value from execution stdout: {extracted_value}"
                    )
                    return extracted_value

        # Method 5: Extract from messages (last resort)
        messages = state.get("messages", [])
        if isinstance(messages, list):
            # Process messages in reverse order (most recent first)
            for message in reversed(messages):
                message_content = ""
                if isinstance(message, dict):
                    message_content = message.get("content", "")
                elif hasattr(message, "content"):
                    message_content = message.content

                if isinstance(message_content, str) and message_content:
                    extracted_value = self.extract_optimal_value_from_text(
                        message_content
                    )
                    if extracted_value is not None:
                        logger.debug(
                            f"Extracted optimal value from message: {extracted_value}"
                        )
                        return extracted_value

        logger.debug("No optimal value found in state")
        return None

    def extract_optimal_solution_variables(self, text: str) -> Optional[dict]:
        """
        Extract decision variable values from optimization code or output text.

        This method extracts decision variable assignments from various formats commonly
        found in optimization solver outputs and code.

        Args:
            text: Text content to analyze (typically current_code or execution output)

        Returns:
            Dictionary mapping variable names to their optimal values, or None if not found
        """
        if not isinstance(text, str) or not text.strip():
            return None

        # Check for infeasible solutions first
        if self.detect_infeasible_solution(text):
            logger.debug("Solution is infeasible, no variable values to extract")
            return {"status": "infeasible"}

        variables = {}

        # Pattern 1: Direct variable assignments (e.g., x1 = 5.0, y[0] = 2.5)
        variable_patterns = [
            # OR-Tools CP-SAT variable output format: VarName = Value
            r"([A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])*)\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            # Standard variable assignments
            r"(\w+(?:\[\d+\])?(?:\[[\d,\s]+\])?)\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            # OR-Tools CP-SAT variable access patterns
            r"(\w+)\.X\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            r"(\w+)\.get\(GRB\.Attr\.X\)\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        ]

        for pattern in variable_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for var_name, value in matches:
                # Skip common non-variable assignments
                if var_name.lower() in [
                    "optimal_value",
                    "objective",
                    "obj_val",
                    "result",
                    "answer",
                ]:
                    continue
                # Skip time/status related variables
                if any(
                    keyword in var_name.lower()
                    for keyword in ["time", "status", "iter", "count", "step"]
                ):
                    continue
                try:
                    variables[var_name] = float(value)
                except ValueError:
                    continue

        # Pattern 2: Variable printing patterns
        print_patterns = [
            # Standard print patterns
            r"print\s*\(\s*f?[\"']([^\"']*?)\s*\{(\w+(?:\[\d+\])?)\}[\"']\s*\)",
            r"print\s*\(\s*f?[\"'](\w+(?:\[\d+\])?)\s*=\s*\{([^}]+)\}[\"']\s*\)",
            # Variable display patterns
            r"(\w+(?:\[\d+\])?)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        ]

        for pattern in print_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    var_name, value = match
                    # Clean up variable name if it's in the description
                    if "=" in var_name or ":" in var_name:
                        var_name = var_name.split("=")[0].split(":")[0].strip()

                    # Skip non-variable patterns
                    if var_name.lower() in [
                        "optimal_value",
                        "objective",
                        "obj_val",
                        "result",
                        "answer",
                    ]:
                        continue

                    try:
                        # Handle case where value might be a variable reference
                        if (
                            isinstance(value, str)
                            and value.replace(".", "")
                            .replace("-", "")
                            .replace("+", "")
                            .isdigit()
                        ):
                            variables[var_name] = float(value)
                        elif isinstance(value, str):
                            # Try to extract numeric value from expression
                            num_match = re.search(
                                r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", value
                            )
                            if num_match:
                                variables[var_name] = float(num_match.group(1))
                    except (ValueError, AttributeError):
                        continue

        # Pattern 3: OR-Tools CP-SAT-style variable output
        gurobi_patterns = [
            # Variable output in format: Variable_name    value
            r"(\w+(?:\[\d+,?\d*\])?)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(?:\n|$)",
            # For loop variable printing
            r"for\s+(\w+)\s+in\s+\w+\.getVars\(\):\s*\n\s*(?:if\s+\w+\.X\s*>\s*[\d.]+:)?\s*print\s*\([^)]*(\w+)\.X[^)]*\)",
        ]

        for pattern in gurobi_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                if len(match) == 2:
                    var_name, value = match
                    try:
                        variables[var_name] = float(value)
                    except ValueError:
                        continue

        # Pattern 4: Structured output parsing (for arrays/matrices)
        array_patterns = [
            # Array-like output: x[0] = 1.0, x[1] = 2.0
            r"(\w+)\[(\d+)\]\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            # Matrix-like output: y[0,1] = 3.5
            r"(\w+)\[(\d+,\d+)\]\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        ]

        for pattern in array_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    var_base, index, value = match
                    var_name = f"{var_base}[{index}]"
                    try:
                        variables[var_name] = float(value)
                    except ValueError:
                        continue

        if variables:
            logger.debug(f"Extracted {len(variables)} decision variables")
            return variables
        else:
            logger.debug("No decision variables found in text")
            return None

    def extract_optimal_solution_from_state(self, state: dict) -> Optional[dict]:
        """
        Extract optimal solution variables from OptAgent state.

        This method extracts decision variable values from various locations in the state,
        primarily from the current_code field and execution outputs.

        Args:
            state: Final state from OptAgent workflow

        Returns:
            Dictionary mapping variable names to their optimal values, or None if not found
        """
        if not isinstance(state, dict):
            logger.warning("State is not a dictionary")
            return None

        # Method 1: Extract from current_code (most reliable source)
        current_code = state.get("current_code", "")
        if isinstance(current_code, str) and current_code:
            variables = self.extract_optimal_solution_variables(current_code)
            if variables is not None:
                logger.debug(
                    f"Extracted optimal solution from current_code: {len(variables)} variables"
                )
                return variables

        # Method 2: Extract from verification result text
        verification_result = state.get("verification_result", "")
        if isinstance(verification_result, str) and verification_result:
            variables = self.extract_optimal_solution_variables(verification_result)
            if variables is not None:
                logger.debug(
                    f"Extracted optimal solution from verification result: {len(variables)} variables"
                )
                return variables

        # Method 3: Extract from final solution execution stdout
        final_solution = state.get("final_solution", {})
        if isinstance(final_solution, dict):
            execution_stdout = final_solution.get("execution_stdout", "")
            if isinstance(execution_stdout, str) and execution_stdout:
                variables = self.extract_optimal_solution_variables(execution_stdout)
                if variables is not None:
                    logger.debug(
                        f"Extracted optimal solution from execution stdout: {len(variables)} variables"
                    )
                    return variables

        # Method 4: Extract from messages (last resort)
        messages = state.get("messages", [])
        if isinstance(messages, list):
            # Process messages in reverse order (most recent first)
            for message in reversed(messages):
                message_content = ""
                if isinstance(message, dict):
                    message_content = message.get("content", "")
                elif hasattr(message, "content"):
                    message_content = message.content

                if isinstance(message_content, str) and message_content:
                    variables = self.extract_optimal_solution_variables(message_content)
                    if variables is not None:
                        logger.debug(
                            f"Extracted optimal solution from message: {len(variables)} variables"
                        )
                        return variables

        logger.debug("No optimal solution variables found in state")
        return None
