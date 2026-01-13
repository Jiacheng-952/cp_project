# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Literal
from langchain_core.language_models import BaseChatModel

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision"]


def get_llm_by_type(llm_type: LLMType) -> BaseChatModel:
    """
    Get LLM instance by type. Imports from src.llms.llm to avoid circular imports.
    """
    from src.llms.llm import get_llm_by_type as _get_llm_by_type

    return _get_llm_by_type(llm_type)


# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "basic",
    "planner": "basic",
    # Core 3-Node Architecture
    "modeler": "basic",  # 建模和编码专家
    "verifier": "basic",  # 综合验证专家
    "corrector": "basic",  # 修正专家
    "reporter": "basic",  # 报告生成专家
    # Other agents
    "podcast_script_writer": "basic",
    "ppt_composer": "basic",
    "prose_writer": "basic",
    "prompt_enhancer": "basic",
}
