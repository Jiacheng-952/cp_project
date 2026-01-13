import dataclasses
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.prebuilt.chat_agent_executor import AgentState

from src.config.configuration import Configuration


def should_use_cp_template(problem_statement: str) -> bool:
    """
    For this specialized CP agent, always use CP template.
    
    Args:
        problem_statement: The optimization problem description
        
    Returns:
        True - always use CP template for specialized CP agent
    """
    # For this specialized CP agent, always use CP template
    return True


def get_appropriate_template(problem_statement: str, default_template: str = "modeler") -> str:
    """
    For this specialized CP agent, always use CP template.
    
    Args:
        problem_statement: The optimization problem description
        default_template: Default template to use if no specific match
        
    Returns:
        Template name to use - always returns cp_modeler for specialized CP agent
    """
    # For this specialized CP agent, always use CP template
    return "cp_modeler"

# Initialize Jinja2 environment with secure settings
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(prompt_name: str) -> str:
    try:
        template = env.get_template(f"{prompt_name}.md")
        return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")


def _is_claude_model(model_name: str) -> bool:

    claude_indicators = ["claude", "anthropic"]
    return any(indicator in model_name.lower() for indicator in claude_indicators)


def apply_prompt_template(
    prompt_name: str, state: AgentState, configurable: Optional[Configuration] = None
) -> List[Dict[str, Any]]:
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **state,
    }

    # Add configurable variables if provided
    if configurable:
        state_vars.update(dataclasses.asdict(configurable))

    # Smart template selection based on problem statement
    problem_statement = state.get("problem_statement", "")
    if prompt_name == "modeler" and problem_statement:
        selected_template = get_appropriate_template(problem_statement, "modeler")
    else:
        selected_template = prompt_name

    try:
        template = env.get_template(f"{selected_template}.md")
        system_prompt = template.render(**state_vars)

        # Ensure messages exist in state
        messages = state.get("messages", [])

        # Detect model type from configuration
        try:
            from src.config import load_yaml_config
            from pathlib import Path

            config_path = str(
                (Path(__file__).parent.parent.parent / "conf.yaml").resolve()
            )
            conf = load_yaml_config(config_path)

            # Check basic model configuration
            basic_model_conf = conf.get("BASIC_MODEL", {})
            model_name = basic_model_conf.get("model", "")

            is_claude = _is_claude_model(model_name)
        except Exception:
            # If we can't determine the model type, assume non-Claude format
            is_claude = False

        def _convert_message_to_dict(msg):
            """Convert LangChain message object to dict format."""
            if hasattr(msg, "type") and hasattr(msg, "content"):
                # LangChain message object
                return {
                    "role": msg.type if msg.type != "human" else "user",
                    "content": msg.content,
                }
            elif isinstance(msg, dict):
                # Already a dict
                return msg
            else:
                # Fallback
                return {"role": "user", "content": str(msg)}

        if is_claude:
            # Claude API format: embed system prompt in first user message
            if messages:
                # Create a copy to avoid modifying the original
                formatted_messages = []
                for i, msg in enumerate(messages):
                    msg_dict = _convert_message_to_dict(msg)
                    if i == 0 and msg_dict.get("role") == "user":
                        # Embed system prompt in first user message
                        content = (
                            f"{system_prompt}\n\n---\n\n{msg_dict.get('content', '')}"
                        )
                        formatted_messages.append({"role": "user", "content": content})
                    else:
                        formatted_messages.append(msg_dict)
                return formatted_messages
            else:
                # If no messages, create a single user message with system prompt
                return [{"role": "user", "content": system_prompt}]
        else:
            # OpenAI/standard format: separate system message
            # Convert all messages to dict format for consistency
            converted_messages = [_convert_message_to_dict(msg) for msg in messages]
            return [{"role": "system", "content": system_prompt}] + converted_messages

    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")
