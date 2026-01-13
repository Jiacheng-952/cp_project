"""
Author: LHL
Date: 2025-08-19 19:29:06
LastEditTime: 2025-09-17 17:41:39
FilePath: /OptAgent-langgraph/src/tools/python_repl.py
"""

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from .decorators import log_io

# Initialize logger
logger = logging.getLogger(__name__)


# @tool
# @log_io
# def python_repl_tool(
#     code: Annotated[
#         str, "The python code to execute to do further analysis or calculation."
#     ],
# ):
#     """Use this to execute python code and do data analysis or calculation. If you want to see the output of a value,
#     you should print it out with `print(...)`. This is visible to the user."""
#     if not isinstance(code, str):
#         error_msg = f"Invalid input: code must be a string, got {type(code)}"
#         logger.error(error_msg)
#         return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"

#     logger.info("Executing Python code via PythonAstREPLTool")
#     try:
#         # Create a new, lightweight REPL instance for each call.
#         repl_tool = PythonAstREPLTool()

#         # Invoke the tool synchronously. LangGraph will manage the threading.
#         result = repl_tool.invoke({"query": code})

#         # The tool's output is a string, which is either the stdout or a traceback.
#         if "Traceback (most recent call last):" in result:
#             logger.error(result)
#             return f"Error executing code:\n```python\n{code}\n```\nError: {result}"

#         logger.info("Code execution successful")
#         return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

#     except BaseException as e:
#         error_msg = repr(e)
#         logger.error(error_msg)
#         return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"


@tool
@log_io
def python_repl_tool(
    code: Annotated[
        str, "The python code to execute to do further analysis or calculation."
    ],
    timeout_seconds: Annotated[int, "Maximum execution time in seconds."] = 30,
):
    """Use this to execute python code and do data analysis or calculation. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    if not isinstance(code, str):
        error_msg = f"Invalid input: code must be a string, got {type(code)}"
        logger.error(error_msg)
        return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"

    logger.info(f"Executing Python code via temporary file (timeout: {timeout_seconds}s)")
    
    # Step 1: Syntax check before execution
    try:
        compile(code, '<string>', 'exec')
        logger.info("Code syntax check passed")
    except SyntaxError as e:
        error_msg = f"Syntax error in Python code at line {e.lineno}: {e.msg}"
        logger.error(error_msg)
        # Provide helpful error message with context
        if e.text:
            error_msg += f"\nProblematic line: {e.text.strip()}"
        return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"
    
    try:
        import tempfile
        import subprocess
        import os
        import sys

        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            # Try to execute using uv environment first (preferred)
            # This ensures we use the project's virtual environment with all dependencies
            try:
                # Find project root directory (where uv.lock exists)
                project_root = os.getcwd()
                while project_root != "/" and not os.path.exists(
                    os.path.join(project_root, "uv.lock")
                ):
                    project_root = os.path.dirname(project_root)

                # If we found uv.lock, use that directory; otherwise use current directory
                if os.path.exists(os.path.join(project_root, "uv.lock")):
                    working_dir = project_root
                    logger.info(f"Found uv project at: {working_dir}")
                else:
                    working_dir = os.getcwd()
                    logger.warning("No uv.lock found, using current directory")

                result = subprocess.run(
                    ["uv", "run", "python", temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=working_dir,
                )
                logger.info("Code executed using uv environment")
            except (FileNotFoundError, subprocess.SubprocessError):
                # Fallback to system Python if uv is not available
                logger.warning("uv not available, falling back to system Python")
                result = subprocess.run(
                    [sys.executable, temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=working_dir,  # Use same working directory
                )

            stdout_content = result.stdout
            stderr_content = result.stderr

            # Check if execution was successful
            if result.returncode == 0:
                if stderr_content:
                    logger.warning(f"Code execution produced stderr: {stderr_content}")

                logger.info("Code execution successful")
                return f"Successfully executed:\n```python\n{code}\n```\nStdout: {stdout_content}"
            else:
                # Execution failed
                error_msg = (
                    stderr_content
                    if stderr_content
                    else f"Process returned code {result.returncode}"
                )
                logger.error(f"Code execution failed: {error_msg}")
                return (
                    f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"
                )

        except subprocess.TimeoutExpired:
            error_msg = f"Code execution timed out after {timeout_seconds} seconds"
            logger.error(error_msg)
            return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"

        except Exception as e:
            error_msg = f"Subprocess execution failed: {str(e)}"
            logger.error(error_msg)
            return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}"
                )

    except Exception as e:
        error_msg = f"Failed to create temporary file: {str(e)}"
        logger.error(error_msg)
        return f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"
