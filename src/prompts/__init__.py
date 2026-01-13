# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
OptAgent Prompts Package

This package contains prompt templates and template management utilities for
the OptAgent optimization workflow system. It provides a Jinja2-based templating
system for managing agent prompts across different workflow phases.

Components:
- template: Core templating functions for loading and applying prompts
- Prompt templates for each agent type:
  - modeler.md: Mathematical modeling and coding prompts
  - verifier.md: Comprehensive verification prompts
  - corrector.md: Solution correction prompts

Example Usage:
    from src.prompts import apply_prompt_template
    
    messages = apply_prompt_template("modeling_expert", state)
    # Returns list of messages with system prompt applied
"""

from .template import apply_prompt_template, get_prompt_template

__all__ = [
    "apply_prompt_template",
    "get_prompt_template",
]
