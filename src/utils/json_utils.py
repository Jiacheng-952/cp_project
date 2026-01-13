# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import json
import json_repair

logger = logging.getLogger(__name__)


def repair_json_output(content: str) -> str:
    """
    Repair and normalize JSON output.

    Args:
        content (str): String content that may contain JSON

    Returns:
        str: Repaired JSON string, or original content if not JSON
    """
    content = content.strip()

    # Remove code block markers if present
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    if content.startswith("```"):
        content = content[3:]  # Remove ```
    if content.endswith("```"):
        content = content[:-3]  # Remove trailing ```

    content = content.strip()

    # Try to extract JSON from the content if it's mixed
    json_start = content.find("{")
    json_end = content.rfind("}")

    if json_start != -1 and json_end != -1 and json_end > json_start:
        # Extract only the JSON part
        json_content = content[json_start : json_end + 1]
        content = json_content

    try:
        # Try to repair and parse JSON
        repaired_content = json_repair.loads(content)
        if not isinstance(repaired_content, dict) and not isinstance(
            repaired_content, list
        ):
            logger.warning("Repaired content is not a valid JSON object or array.")
            return content
        content = json.dumps(repaired_content, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"JSON repair failed: {e}")
        # If json_repair fails, try simple JSON loading
        try:
            simple_content = json.loads(content)
            content = json.dumps(simple_content, ensure_ascii=False)
        except Exception as e2:
            logger.warning(f"Simple JSON parsing also failed: {e2}")

    return content
