"""
Author: LHL
Date: 2025-07-20 21:11:41
LastEditTime: 2025-08-02 15:43:52
FilePath: /deer-flow/src/config/__init__.py
"""

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Configuration management for multi-agent system."""

from typing import Dict, List, Literal, Any

# Define supported search engines
from enum import Enum

# Agent descriptions for prompt templates
AGENT_DESCRIPTIONS = {
    "coordinator": {
        "name": "coordinator",
        "description": "A helpful assistant that coordinates the research task.",
        "role": "coordinator",
        "goal": "To coordinate the research task and ensure proper handoff to the planner.",
        "capabilities": "Can understand user queries and handoff to the planner agent when appropriate.",
    },
    "modeling_expert": {
        "name": "modeling_expert",
        "description": "A mathematical modeling expert specializing in optimization and operations research.",
        "role": "modeling expert",
        "goal": "To analyze modeling problems and research appropriate solution methodologies.",
        "capabilities": "Outputs a Markdown report summarizing findings. Modeling expert can not do math or programming.",
    },
    "coder": {
        "name": "coder",
        "description": "A Python coding expert.",
        "role": "coder",
        "goal": "To write Python code for mathematical modeling and optimization problems.",
        "capabilities": "Can write Python code using mathematical optimization libraries and solve optimization problems.",
    },
    "verifier": {
        "name": "verifier",
        "description": "A mathematical optimization solution verifier.",
        "role": "verifier",
        "goal": "To verify optimization solutions against problem requirements and constraints.",
        "capabilities": "Can check if solutions satisfy all constraints, validate problem formulations, and identify errors in optimization models.",
    },
    "planner": {
        "name": "planner",
        "description": "A strategic planner that breaks down research tasks.",
        "role": "planner",
        "goal": "To create detailed research plans with clear steps.",
        "capabilities": "Can analyze research requirements and create structured execution plans.",
    },
    "reporter": {
        "name": "reporter",
        "description": "A professional report writer.",
        "role": "reporter",
        "goal": "To compile research findings into comprehensive reports.",
        "capabilities": "Can synthesize information from multiple sources into well-structured reports.",
    },
}

from .loader import load_yaml_config
from .tools import SELECTED_SEARCH_ENGINE, SearchEngine
from .questions import BUILT_IN_QUESTIONS, BUILT_IN_QUESTIONS_ZH_CN

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Team configuration
TEAM_MEMBER_CONFIGURATIONS = {
    "modeling_expert": {
        "name": "modeling_expert",
        "desc": (
            "Responsible for searching and collecting relevant information, understanding user needs and conducting research analysis"
        ),
        "desc_for_llm": (
            "Uses search engines and web crawlers to gather information from the internet. "
            "Outputs a Markdown report summarizing findings. Modeling expert can not do math or programming."
        ),
        "is_optional": False,
    },
    "coder": {
        "name": "coder",
        "desc": (
            "Responsible for code implementation, debugging and optimization, handling technical programming tasks"
        ),
        "desc_for_llm": (
            "Executes Python or Bash commands, performs mathematical calculations, and outputs a Markdown report. "
            "Must be used for all mathematical computations."
        ),
        "is_optional": True,
    },
    "verifier": {
        "name": "verifier",
        "desc": (
            "Responsible for verification of optimization solutions and validation of mathematical models"
        ),
        "desc_for_llm": (
            "Verifies optimization solutions by checking constraints satisfaction, model accuracy, and solution feasibility. "
            "Provides detailed feedback for corrections when solutions are invalid."
        ),
        "is_optional": False,
    },
}

TEAM_MEMBERS = list(TEAM_MEMBER_CONFIGURATIONS.keys())

__all__ = [
    # Other configurations
    "TEAM_MEMBERS",
    "TEAM_MEMBER_CONFIGURATIONS",
    "SELECTED_SEARCH_ENGINE",
    "SearchEngine",
    "BUILT_IN_QUESTIONS",
    "BUILT_IN_QUESTIONS_ZH_CN",
    load_yaml_config,
]
