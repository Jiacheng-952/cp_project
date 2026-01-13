# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
CSP/COP Problem Classifier Node Module

This module provides intelligent problem classification using LLM
for determining whether a CP problem is a Constraint Satisfaction Problem (CSP)
or Constraint Optimization Problem (COP).
"""

import logging
import asyncio
from typing import Dict, Any, Tuple

from .types import OptAgentState
from src.config.agents import get_llm_by_type, AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template

logger = logging.getLogger(__name__)

# Prompt template for CSP/COP classification
CSP_COP_CLASSIFICATION_PROMPT = """
You are a Constraint Programming expert specializing in job scheduling and workflow optimization problems. Your task is to distinguish between Constraint Satisfaction Problems (CSP) and Constraint Optimization Problems (COP) in the context of CP (Constraint Programming).

## Definitions:

**CSP (Constraint Satisfaction Problem)**:
- Goal: Find ANY feasible schedule or assignment that satisfies all constraints
- No objective function to optimize
- Answer is either "solution exists" or "no solution"
- Examples: Feasibility checks for shift assignments, resource allocation feasibility

**COP (Constraint Optimization Problem)**:
- Goal: Find the BEST schedule or assignment that satisfies all constraints
- Has an objective function to maximize or minimize
- Answer includes optimal value and solution
- Examples: Minimize makespan in job scheduling, minimize resource usage, maximize throughput

## Special Focus on Job Scheduling Problems:
Many 0-1 integer programming problems in job scheduling are difficult to solve, but CP with global constraints excels at these:
- Job Shop Scheduling Problems (JSSP)
- Flexible Job Shop Scheduling Problems (FJSSP)
- Resource-Constrained Project Scheduling Problems (RCPSP)
- Multi-mode RCPSP
- Assembly line balancing problems
- Staff scheduling with complex constraints
- Machine scheduling with sequence-dependent setup times

## Classification Criteria:

Classify the following problem as either CSP or COP based on these criteria:

1. **Keywords indicating COP**:
   - Minimize, Maximize, Optimize, Minimum, Maximum, Optimal
   - Makespan, Lateness, Tardiness, Earliness, Throughput
   - Cost, Profit, Time, Distance, Weight, Efficiency
   - Best, Worst, Cheapest, Most expensive, Shortest, Longest

2. **Keywords indicating CSP**:
   - Satisfy, Feasible, Possible, Can we, Is it possible
   - Schedule, Assign, Place, Fit, Arrange
   - Without optimization terms

3. **Question Structure**:
   - CSP: "Can we...", "Is it possible...", "Find a solution..."
   - COP: "Minimize...", "Maximize...", "Find the best..."

## Problem Statement:
{problem_statement}

## Instructions:
Analyze the problem statement carefully and determine whether it's a CSP or COP problem.
Pay special attention to job scheduling and workflow optimization contexts.

Respond in the following JSON format:
{{
  "classification": "CSP or COP",
  "confidence": "High/Medium/Low",
  "reasoning": "Brief explanation of your classification decision"
}}
"""

async def csp_cop_classifier_node(state: OptAgentState, config) -> Dict[str, Any]:
    """
    Intelligent CSP/COP problem classifier node for determining problem type.
    
    This node uses LLM to analyze the CP problem statement and determine
    whether it's a Constraint Satisfaction Problem (CSP) or 
    Constraint Optimization Problem (COP).
    
    优化版本：对于简单问题使用纯规则分类，跳过LLM调用
    
    Args:
        state: Current OptAgent state
        config: LangGraph node configuration
        
    Returns:
        State update dictionary with CSP/COP classification
    """
    logger.info("CSP/COP Classifier node analyzing problem type")
    
    # Debug: Print to console to confirm this node is being executed
    print("🔍 CSP/COP Classifier Node: Starting classification...")
    print(f"🔍 Problem statement: {state.get('problem_statement', 'NOT FOUND')[:100]}...")
    
    # Get problem statement from state
    problem_statement = state.get("problem_statement", "")
    
    if not problem_statement:
        logger.warning("No problem statement found for CSP/COP classification")
        return {
            "csp_cop_classification": "UNKNOWN",
            "csp_cop_confidence": "LOW",
            "csp_cop_reasoning": "No problem statement provided"
        }
    
    # 优化：对于简单问题（问题长度<300字符），直接使用规则分类，跳过LLM
    if len(problem_statement) < 300:
        logger.info("Simple problem detected, using rule-based classification (skipping LLM)")
        classification, confidence, reasoning = fallback_csp_cop_classification(problem_statement)
        return {
            "problem_type": classification,
            "csp_cop_classification": classification,
            "csp_cop_confidence": confidence,
            "csp_cop_reasoning": f"Fast rule-based classification: {reasoning}"
        }
    
    try:
        # Prepare classification prompt
        classification_prompt = CSP_COP_CLASSIFICATION_PROMPT.format(
            problem_statement=problem_statement
        )
        
        # Get LLM for classification
        llm = get_llm_by_type(AGENT_LLM_MAP.get("modeler", "basic"))
        
        # Apply LLM timeout from state - 使用较短的超时时间
        try:
            timeout_cfg = state.get("timeout_config", {}) or {}
            llm_timeout = int(timeout_cfg.get("llm_request_timeout", 60))  # 减少到60秒
        except Exception:
            llm_timeout = 60
            
        # Execute classification
        response = await asyncio.wait_for(
            llm.ainvoke([{"role": "user", "content": classification_prompt}]), 
            timeout=llm_timeout
        )
        
        if hasattr(response, "content"):
            classification_content = response.content
        else:
            classification_content = str(response)
            
        logger.info(f"CSP/COP Classification completed: {classification_content}")
        
        # Parse the classification result - properly extract JSON from LLM response
        classification = "UNKNOWN"
        confidence = "LOW"
        reasoning = "Failed to parse classification response"
        
        try:
            # Extract JSON content from the response (handle code block formatting)
            import json
            import re
            
            # Try to extract JSON from code blocks
            json_match = re.search(r'```json\s*({.*?})\s*```', classification_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code block, try to find JSON directly
                json_match = re.search(r'({.*})', classification_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = classification_content
            
            # Parse JSON
            classification_data = json.loads(json_str)
            
            # Extract classification, confidence, and reasoning
            classification = classification_data.get("classification", "UNKNOWN").upper()
            confidence = classification_data.get("confidence", "LOW").capitalize()
            reasoning = classification_data.get("reasoning", "No reasoning provided")
            
            logger.info(f"JSON Parsed - Classification: {classification}, Confidence: {confidence}")
            
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.warning(f"Failed to parse JSON response: {e}, falling back to keyword parsing")
            
            # Fallback to keyword-based parsing
            content_lower = classification_content.lower()
            if "csp" in content_lower and "cop" not in content_lower:
                classification = "CSP"
                confidence = "HIGH"
                reasoning = "LLM classified as CSP (fallback parsing)"
            elif "cop" in content_lower and "csp" not in content_lower:
                classification = "COP"
                confidence = "HIGH"
                reasoning = "LLM classified as COP (fallback parsing)"
            else:
                # Final fallback to rule-based classification
                classification, confidence, reasoning = fallback_csp_cop_classification(problem_statement)
                reasoning = f"Fallback classification: {reasoning}"
        
        logger.info(f"CSP/COP Classification Result: {classification} (Confidence: {confidence})")
        
        return {
            "problem_type": classification,  # Set problem_type for other nodes
            "csp_cop_classification": classification,
            "csp_cop_confidence": confidence,
            "csp_cop_reasoning": reasoning
        }
        
    except Exception as e:
        logger.error(f"CSP/COP classification failed: {str(e)}")
        # Fallback to rule-based classification
        classification, confidence, reasoning = fallback_csp_cop_classification(problem_statement)
        
        return {
            "problem_type": classification,  # Set problem_type for other nodes
            "csp_cop_classification": classification,
            "csp_cop_confidence": confidence,
            "csp_cop_reasoning": f"Fallback classification due to error: {reasoning}"
        }


def fallback_csp_cop_classification(problem_statement: str) -> Tuple[str, str, str]:
    """
    Rule-based fallback classification for CSP/COP problems.
    
    Args:
        problem_statement: The problem statement to classify
        
    Returns:
        Tuple of (classification, confidence, reasoning)
    """
    if not problem_statement:
        return "CSP", "LOW", "Default to CSP when no problem statement provided"
        
    problem_lower = problem_statement.lower()
    
    # Keywords indicating COP (optimization)
    cop_keywords = [
        'minimize', 'maximize', 'optimize', 'minimum', 'maximum', 'optimal',
        'cost', 'profit', 'time', 'distance', 'weight', 'efficiency',
        'best', 'worst', 'cheapest', 'most expensive', 'shortest', 'longest',
        'minimise', 'maximise', 'lowest', 'highest', 'least', 'most'
    ]
    
    # Keywords indicating CSP (feasibility)
    csp_keywords = [
        'satisfy', 'feasible', 'possible', 'can we', 'is it possible',
        'find a solution', 'arrange', 'schedule', 'assign', 'color', 'place', 'fit'
    ]
    
    # Count COP indicators
    cop_count = sum(1 for keyword in cop_keywords if keyword in problem_lower)
    
    # Count CSP indicators
    csp_count = sum(1 for keyword in csp_keywords if keyword in problem_lower)
    
    if cop_count > csp_count:
        return "COP", "MEDIUM", f"Rule-based: {cop_count} COP keywords vs {csp_count} CSP keywords"
    elif csp_count > cop_count:
        return "CSP", "MEDIUM", f"Rule-based: {csp_count} CSP keywords vs {cop_count} COP keywords"
    else:
        # If counts are equal or both zero, check for specific patterns
        if any(word in problem_lower for word in ['minimize', 'maximize', 'optimal']):
            return "COP", "MEDIUM", "Rule-based: Strong optimization keywords detected"
        elif any(word in problem_lower for word in ['satisfy', 'feasible', 'schedule', 'assign']):
            return "CSP", "MEDIUM", "Rule-based: Strong feasibility keywords detected"
        else:
            return "CSP", "LOW", "Rule-based: Default to CSP when unable to determine"