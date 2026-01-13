# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Robust Python code extraction from LLM-generated text, optimized for scenarios
where the code block is expected at the beginning of the response.
"""
import re
import logging

logger = logging.getLogger(__name__)


def _remove_markdown_formatting(code: str) -> str:
    """
    Removes markdown formatting from extracted code to ensure it's valid Python.
    
    This function removes:
    - Markdown bullet points (like "- **Problem Type**: ...")
    - Markdown headers (like "## Title")
    - Bold/italic markers
    - Other markdown-specific formatting that would cause syntax errors
    
    Args:
        code: Potentially contaminated Python code with markdown formatting
        
    Returns:
        Cleaned Python code with markdown formatting removed
    """
    lines = code.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip lines that are clearly markdown formatting
        # Check for markdown bullet points with bold text
        if re.match(r'^-\s*\*\*.*?\*\*:', stripped):
            logger.debug(f"Removing markdown bullet point: {stripped}")
            continue
        
        # Skip markdown headers
        if stripped.startswith('#') and not stripped.startswith('# '):
            # Keep Python comments (# ) but remove markdown headers (##, ###, etc.)
            if stripped.startswith('##') or stripped.startswith('###'):
                logger.debug(f"Removing markdown header: {stripped}")
                continue
        
        # Skip lines that are just markdown formatting
        if re.match(r'^\*\*.*?\*\*$', stripped):
            logger.debug(f"Removing markdown bold line: {stripped}")
            continue
        
        # Keep the line if it's not markdown formatting
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)


def _final_cleanup(code: str) -> str:
    """
    Strips leading/trailing whitespace and language identifiers from a raw code block.
    Example: 'python\nimport os' -> 'import os'
    """
    lines = code.strip().split("\n")
    # Check if the first line is a short, single-word language identifier
    first_line = lines[0].strip()
    if len(lines) > 1 and len(first_line.split()) == 1 and len(first_line) < 15:
        # It's likely a language tag like 'python', 'py', etc. Remove it.
        return "\n".join(lines[1:]).strip()
    else:
        # The first line is actual code.
        return "\n".join(lines).strip()


def extract_python_code(text: str) -> str:
    """
    Robustly extracts a Python code block from the beginning of an LLM response.

    This extractor is designed with the assumption that the primary code block
    is the first significant content in the text. It uses a prioritized, top-down
    approach:

    1.  **Top-Down Fenced Block Search**: It first looks for the first occurrence
        of a markdown code fence (```) and extracts its content. This is the
        preferred and most reliable method.

    2.  **Heuristic Fallback Scan**: If and only if NO fenced code blocks are found
        anywhere in the text, it falls back to a line-by-line heuristic scan.
        This scan starts at the first line that looks like code (e.g., 'import', '#')
        and stops when it encounters multiple consecutive lines of prose, making it
        resilient to earlier bugs.

    Args:
        text: Raw text from an LLM response.

    Returns:
        The cleaned, extracted Python code as a string, or an empty string if no
        code is found.
    """
    if not isinstance(text, str) or not text.strip():
        if text is not None and text != "":
            logger.warning("Code extraction received text with only whitespace.")
        return ""

    # Strategy 1: Find the first fenced code block. This is the most reliable.
    # The pattern looks for ```, optionally followed by a language tag, then captures
    # everything non-greedily until the next ```.
    fenced_code_pattern = r"```(?:[a-zA-Z]*)?\s*\n(.*?)\n```"
    match = re.search(fenced_code_pattern, text, re.DOTALL)

    if match:
        code = match.group(1).strip()
        if code:
            logger.info(
                "Successfully extracted code using top-down fenced block search."
            )
            # Remove any markdown formatting that might have been included
            code = _remove_markdown_formatting(code)
            # No need for _final_cleanup as the regex captures only the content.
            return code

    # A slightly more lenient version if the first one fails (e.g. no newline)
    fenced_code_pattern_alt = r"```(?:[a-zA-Z]*)?(.*?)```"
    match = re.search(fenced_code_pattern_alt, text, re.DOTALL)
    if match:
        # We need to clean the language identifier here.
        code = _final_cleanup(match.group(1))
        if code:
            logger.info(
                "Successfully extracted code using lenient top-down fenced block search."
            )
            # Remove any markdown formatting that might have been included
            code = _remove_markdown_formatting(code)
            return code

    # Strategy 2: Heuristic Fallback Scan (only if NO fenced blocks were found).
    # This is a safety net for when the LLM completely ignores formatting.
    logger.warning(
        "No fenced code block found. Falling back to heuristic line scanning."
    )

    lines = text.split("\n")
    code_lines = []
    start_index = -1

    # Find the first line that looks like code
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("#")
        ):
            start_index = i
            break

    if start_index == -1:
        logger.error(
            "Heuristic scan failed: No line starting with 'import', 'from', or '#' found."
        )
        return ""

    # From the start index, collect lines until we see clear evidence of prose
    prose_counter = 0
    code_keywords = {
        "import",
        "from",
        "def",
        "class",
        "for",
        "while",
        "if",
        "try",
        "model",
        "gp.",
    }

    for line in lines[start_index:]:
        stripped = line.strip()
        is_prose_candidate = False

        if not stripped:  # Empty lines are not prose, they are part of code formatting
            prose_counter = 0
        else:
            is_indented = line.startswith((" ", "\t"))
            is_comment = stripped.startswith("#")
            # Check if line looks like natural language
            has_code_like_chars = any(
                c in stripped for c in "=[]{}()<>*+/-_.,:"
            ) or any(kw in stripped for kw in code_keywords)

            # A line is a prose candidate if it's unindented, not a comment, and lacks code signals.
            if (
                not is_indented
                and not is_comment
                and not has_code_like_chars
                and len(stripped.split()) > 3
            ):
                is_prose_candidate = True

        if is_prose_candidate:
            prose_counter += 1
        else:
            # Any code-like line resets the counter
            prose_counter = 0
            code_lines.append(line)

        # Stop if we see 2 consecutive lines of prose
        if prose_counter >= 2:
            # Remove the prose lines that were mistakenly added
            code_lines = code_lines[:-prose_counter]
            logger.info(
                f"Heuristic scan stopped after detecting {prose_counter} consecutive prose lines."
            )
            break

    if code_lines:
        # Final cleanup of trailing empty lines
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()

        result = "\n".join(code_lines).strip()
        if result:
            logger.info("Successfully extracted code using heuristic line scanning.")
            # Remove any markdown formatting that might have been included
            result = _remove_markdown_formatting(result)
            return result

    logger.error(
        "All extraction strategies failed. No valid Python code could be extracted."
    )
    return ""
