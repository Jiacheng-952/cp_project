# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import ast
import logging
from typing import Annotated, Dict, Any
from langchain_core.tools import tool
from .decorators import log_io

logger = logging.getLogger(__name__)


@tool
@log_io
def syntax_validator_tool(
    code: Annotated[str, "The Python code to validate for syntax correctness."],
) -> Dict[str, Any]:
    """
    Validate Python code syntax using AST parsing.

    This is a deterministic tool that checks if the provided Python code
    has valid syntax without executing it.

    Args:
        code: Python code string to validate

    Returns:
        Dict containing:
        - status: "PASS" or "FAIL"
        - valid: boolean indicating if syntax is valid
        - error_message: description of syntax error if any
        - error_line: line number where error occurred (if any)
        - error_column: column number where error occurred (if any)
    """
    if not isinstance(code, str):
        error_msg = f"Invalid input: code must be a string, got {type(code)}"
        logger.error(error_msg)
        return {
            "status": "FAIL",
            "valid": False,
            "error_message": error_msg,
            "error_line": None,
            "error_column": None,
        }

    if not code.strip():
        return {
            "status": "FAIL",
            "valid": False,
            "error_message": "Empty code provided",
            "error_line": None,
            "error_column": None,
        }

    logger.info("Validating Python code syntax")

    try:
        # Parse the code using AST
        ast.parse(code.encode("utf-8"))
        logger.info("Code syntax validation successful")
        return {
            "status": "PASS",
            "valid": True,
            "error_message": None,
            "error_line": None,
            "error_column": None,
        }

    except SyntaxError as e:
        error_msg = f"Syntax error: {e.msg}"
        logger.error(
            f"Syntax validation failed: {error_msg} at line {e.lineno}, column {e.offset}"
        )
        return {
            "status": "FAIL",
            "valid": False,
            "error_message": error_msg,
            "error_line": e.lineno,
            "error_column": e.offset,
        }

    except Exception as e:
        error_msg = f"Unexpected error during syntax validation: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "FAIL",
            "valid": False,
            "error_message": error_msg,
            "error_line": None,
            "error_column": None,
        }


def validate_python_syntax(code: str) -> Dict[str, Any]:
    """
    Direct function interface for syntax validation (without tool decorator).

    Useful for internal usage without LangChain tool overhead.
    """
    return syntax_validator_tool.func(code)
