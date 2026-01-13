# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
OptAgent Nodes Module

This module contains the three core nodes for the OptAgent
workflow: modeler, verifier, and corrector.
"""

import logging
import asyncio
import re
from typing import Dict, Any

from .types import OptAgentState


def extract_values_from_stdout(stdout: str) -> tuple:
    """从stdout中统一提取最优值和变量值
    
    Args:
        stdout: 求解器的标准输出内容
        
    Returns:
        tuple: (optimal_value, variable_values)
    """
    optimal_value = None
    variable_values = {}
    
    if not stdout:
        return optimal_value, variable_values
    
    # 改进的最优值提取正则表达式，支持更多格式
    optimal_patterns = [
        r'optimal_value\s*=\s*([\d.-]+)',  # 数字格式
        r'optimal_value\s*=\s*([\w\s]+)',  # 单词格式
        r'optimal\s*value\s*[:=]\s*([\d.-]+)',  # 英文格式
        r'最优值\s*[:=]\s*([\d.-]+)',  # 中文格式
        r'objective\s*value\s*[:=]\s*([\d.-]+)',  # 目标值格式
    ]
    
    for pattern in optimal_patterns:
        optimal_match = re.search(pattern, stdout, re.IGNORECASE)
        if optimal_match:
            optimal_value = optimal_match.group(1).strip()
            break
    
    # 提取所有变量值
    var_patterns = [
        r'(\w+)\s*=\s*(\d+)',  # 整数赋值
        r'(\w+)\s*=\s*([\d.-]+)',  # 浮点数赋值
        r'(\w+)\s*[:=]\s*(\d+)',  # 冒号格式
    ]
    
    for pattern in var_patterns:
        var_matches = re.findall(pattern, stdout)
        for var_name, var_value in var_matches:
            # 跳过常见的关键字
            if var_name.lower() not in ['optimal', 'objective', 'value', 'status', 'result']:
                variable_values[var_name] = var_value
    
    return optimal_value, variable_values
from src.utils.optimality_checker import (
    check_solution_quality, 
    create_convergence_metric, 
    assess_convergence, 
    should_continue_optimization as should_continue_optimization_func
)

logger = logging.getLogger(__name__)

# 快速验证函数
def quick_verify_solution(execution_result: Dict[str, Any], problem_statement: str) -> bool:
    """快速验证解决方案，避免LLM调用
    
    优化版本：增强简单问题的快速验证逻辑
    """
    # 检查执行状态
    if execution_result.get("status") != "PASS":
        return False
    
    # 检查是否有可行解
    if not execution_result.get("has_feasible_solution", False):
        return False
    
    # 对于简单问题（长度<300字符），如果执行成功且有可行解，直接通过
    if len(problem_statement) < 300:
        # 额外检查：确保输出中包含有意义的变量赋值或结果
        stdout = execution_result.get("stdout", "")
        # 检查是否有数值结果或变量赋值
        import re
        has_result = bool(re.search(r'optimal_value\s*=\s*[\d.-]+', stdout) or 
                         re.search(r'[\w]+\s*=\s*\d+', stdout))
        if has_result:
            return True
    
    # 对于中等复杂度问题（长度<500字符），检查是否有错误信息
    if len(problem_statement) < 500:
        if execution_result.get("error_message"):
            return False
        stdout = execution_result.get("stdout", "")
        if "error" in stdout.lower() or "exception" in stdout.lower():
            return False
        # 如果没有明显错误且有可行解，通过验证
        return True
    
    # 检查是否有错误信息
    if execution_result.get("error_message"):
        return False
    
    # 检查标准输出是否有明显错误
    stdout = execution_result.get("stdout", "")
    if "error" in stdout.lower() or "exception" in stdout.lower():
        return False
    
    return True


async def modeler_node(state: OptAgentState, config) -> Dict[str, Any]:
    """
    Modeler node for creating CP mathematical models and Python code.

    Args:
        state: Current OptAgent state
        config: LangGraph node configuration

    Returns:
        State update dictionary with generated CP model and code
    """
    logger.info("Modeler节点生成CP数学模型和代码")

    from src.config.agents import get_llm_by_type, AGENT_LLM_MAP
    from src.prompts.template import apply_prompt_template
    from src.utils.robust_code_extraction import (
        extract_python_code as robust_extract_python_code,
    )

    # Prepare state for template
    template_state = dict(state)
    template_state["correction_needed"] = state.get("correction_needed", False)
    template_state["verification_result"] = state.get("verification_result", "")
    template_state["correction_count"] = state.get("correction_count", 0)

    # For this specialized CP agent, always use CP template
    template_name = "cp_modeler"

    # Apply CP prompt template
    messages = apply_prompt_template(template_name, template_state)
    llm = get_llm_by_type(AGENT_LLM_MAP.get("modeler", "basic"))

    response = await llm.ainvoke(messages)
    
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    # Debug: Log the full response content
    logger.debug(f"LLM response content length: {len(content)}")
    logger.debug(f"LLM response preview: {content[:500]}")

    # Extract mathematical model and code from response
    model_content = ""
    code_content = ""

    # Extract model section (between "## Mathematical Model" and "## Python Implementation")
    import re

    # Try multiple patterns to extract mathematical model
    model_patterns = [
        r"## CP Mathematical Model\s*\n(.*?)(?=## Python Implementation)",  # CP-specific format
        r"## Mathematical Model\s*\n(.*?)(?=## Python Implementation)",     # Original format
        r"## Mathematical Model\s*\n(.*?)(?=##|```|$)",                     # Fallback pattern 1
        r"数学模型\s*\n(.*?)(?=Python实现|```|$)",                          # Chinese format
    ]
    
    model_content = ""
    for pattern in model_patterns:
        model_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if model_match:
            model_content = model_match.group(1).strip()
            logger.debug(f"Model extraction successful with pattern: {pattern[:50]}...")
            break

    # Extract code using robust extraction
    code_content = robust_extract_python_code(content)

    if not model_content and not code_content:
        logger.warning("Failed to extract both model and code from response")
        model_content = content  # Use full content as fallback
    elif not model_content:
        logger.warning("Failed to extract model from response, using fallback")
        # Try to extract any meaningful model content from the beginning
        model_sections = re.findall(r"## .*?\n(.*?)(?=##|```|$)", content, re.DOTALL)
        if model_sections:
            model_content = model_sections[0].strip()

    # Update solution version
    solution_version = state.get("solution_version", 0) + 1

    # Log detailed state information for debugging
    logger.debug(f"Modeler node returning - model length: {len(model_content)}, code length: {len(code_content)}")
    logger.debug(f"Modeler node state keys: {list(state.keys())}")
    
    return {
        "current_model": model_content,
        "current_code": code_content,
        "solution_version": solution_version,
        "correction_needed": False,  # Reset correction flag
        "verification_passed": False,  # Reset verification status
        "verification_failed": False,
        "messages": state.get("messages", [])
        + [{"role": "assistant", "content": content}],
    }


async def verifier_node(state: OptAgentState, config) -> Dict[str, Any]:
    """
    Verifier node for comprehensive CP solution verification.

    Args:
        state: Current OptAgent state
        config: LangGraph node configuration

    Returns:
        State update dictionary with CP verification results
    """
    logger.info("Verifier节点执行CP验证")

    from src.config.agents import get_llm_by_type, AGENT_LLM_MAP
    from src.prompts.template import apply_prompt_template
    from src.tools import cp_solver_executor_tool
    from src.utils.performance_optimizer import get_performance_optimizer, performance_timed

    # Initialize performance optimizer
    optimizer = get_performance_optimizer()

    # Get current model and code
    current_model = state.get("current_model", "")
    current_code = state.get("current_code", "")

    # Log detailed state information for debugging
    logger.debug(f"Verifier节点接收状态 - 有模型: {bool(current_model)}, 有代码: {bool(current_code)}")
    logger.debug(f"Verifier节点状态键: {list(state.keys())}")

    if not current_model or not current_code:
        logger.warning("缺少模型或代码进行验证")
        return {
            "verification_result": "错误: 缺少模型或代码进行验证",
            "verification_passed": False,
            "verification_failed": True,
        }

    try:
        # Check for cached execution result
        cached_result = optimizer.get_cached_execution_result(current_code, None)
        
        if cached_result:
            logger.info("使用缓存的代码执行结果")
            execution_result = cached_result
        else:
            # Execute the code first to get execution results
            logger.info("执行代码 (无缓存)")
            execution_result = cp_solver_executor_tool.invoke(
                {"code": current_code, "timeout_seconds": 60}
            )
            # Cache the result for future use
            optimizer.cache_execution_result(current_code, None, execution_result)

        logger.info(f"代码执行完成: {execution_result['status']}")

        # 首先检查求解器是否报告了OPTIMAL状态
        solution_status = execution_result.get("solution_status", "")
        if solution_status == "OPTIMAL":
            logger.info("求解器报告最优解，自动通过验证")
            
            # 使用求解器的OPTIMAL状态结果
            is_approved = True
            verification_content = "求解器报告最优解，自动通过验证"
            
            # 确保从执行结果中提取最优值和变量值
            optimal_value = execution_result.get("optimal_value")
            variable_values = execution_result.get("variable_values", {})
            
            if optimal_value is not None:
                optimal_value_extracted = True
            else:
                # 如果执行结果中没有optimal_value，尝试从stdout中提取
                stdout = execution_result.get("stdout", "")
                extracted_optimal_value, extracted_vars = extract_values_from_stdout(stdout)
                
                if extracted_optimal_value:
                    optimal_value = extracted_optimal_value
                    optimal_value_extracted = True
                else:
                    # 如果还是找不到，标记为已提取但值为未知
                    optimal_value = "OPTIMAL_SOLUTION_FOUND"
                    optimal_value_extracted = True
                
                # 如果变量值为空，使用提取的变量值
                if not variable_values:
                    variable_values = extracted_vars
        # 其次尝试快速验证
        elif quick_verify_solution(execution_result, state.get("problem_statement", "")):
            logger.info("快速验证通过，跳过详细LLM验证")
            
            # 使用快速验证结果
            is_approved = True
            verification_content = "快速验证通过: 代码执行成功且有可行解"
        else:
            # 使用原有的详细验证流程
            # Perform quick verification first (rule-based)
            quick_verification_result = optimizer.quick_verifier.quick_verify(
                execution_result, state.get("problem_statement", "")
            )

            # If quick verification passes and we have a feasible solution, skip detailed LLM verification
            if (quick_verification_result["quick_verification_passed"] and 
                execution_result.get("has_feasible_solution", False)):
                logger.info("快速验证通过，跳过详细LLM验证")
                
                # Use quick verification result
                is_approved = True
                verification_content = f"""快速验证通过基于规则检查:
- 基本解决方案检查: {quick_verification_result['rule_results']['basic_solution_check']}
- 约束满足: {quick_verification_result['rule_results']['constraint_satisfaction']}
- 目标验证: {quick_verification_result['rule_results']['objective_validation']}
验证时间: {quick_verification_result['verification_time_ms']}ms
"""
            else:
                # Proceed with detailed LLM verification
                logger.info("执行详细LLM验证")
                
                # Prepare verification input
                template_state = dict(state)
                template_state["execution_result"] = execution_result

                # Apply prompt template
                messages = apply_prompt_template("verifier", template_state)

                # Check if code execution failed
                execution_failed = (
                    execution_result["status"] != "PASS"
                    or not execution_result["executed"]
                    or execution_result.get("error_message")
                )

                if execution_failed:
                    # If execution failed, create a simple rejection message
                    verification_task = f"""
# VERIFICATION TASK - EXECUTION ERROR

## Original Problem Statement
{state.get('problem_statement', 'Not available')}

## Code Execution Results
- Status: {execution_result['status']}
- Error Message: {execution_result.get('error_message', 'Code execution failed')}
- Stderr: {execution_result['stderr']}

The code failed to execute properly. Please reject and report the error.
"""
                else:
                    # If execution succeeded, do solution verification
                    verification_task = f"""
# VERIFICATION TASK - SOLUTION VALIDATION

## Original Problem Statement
{state.get('problem_statement', 'Not available')}

## Solution Results
- Optimal Value: {execution_result.get('optimal_value', 'Not found')}
- Solution Status: {execution_result.get('solution_status', 'Unknown')}
- Execution Output:
{execution_result['stdout']}

Please verify if this solution is reasonable and satisfies all constraints from the original problem.
"""
            messages.append({"role": "user", "content": verification_task})

            # LLM verification
            llm = get_llm_by_type(AGENT_LLM_MAP.get("verifier", "basic"))
            
            response = await llm.ainvoke(messages)
            
            if hasattr(response, "content"):
                verification_content = response.content
            else:
                verification_content = str(response)

            logger.debug(f"Verification completed: {verification_content}\n")

            # Parse verification result using robust parsing
            from src.utils.robust_parsing import extract_verification_status

            is_approved, confidence, detail = extract_verification_status(
                verification_content, "approval"
            )
            logger.info(f"Verification result: {detail}")

        # Extract optimal value and variable values if verification passed
        optimal_value = None
        optimal_value_extracted = False
        
        # Check if this is a CP problem with a feasible solution (especially for CSP)
        # This should be set regardless of verification approval status to ensure
        # consistent reporting even when verification fails
        has_feasible_solution = execution_result.get("has_feasible_solution", False)
        
        # Extract variable values from execution result or stdout
        variable_values = execution_result.get("variable_values", {})
        
        # 如果变量值为空，尝试从stdout中提取
        if not variable_values:
            stdout = execution_result.get("stdout", "")
            _, extracted_vars = extract_values_from_stdout(stdout)
            variable_values = extracted_vars

        if is_approved:
            # Try to extract optimal value from execution results first
            exec_optimal_value = execution_result.get("optimal_value")
            if exec_optimal_value is not None:
                # Special handling for CSP problems that indicate feasible solutions
                if exec_optimal_value == "FEASIBLE_SOLUTION_FOUND":
                    optimal_value = "FEASIBLE_SOLUTION_FOUND"
                    optimal_value_extracted = True
                elif exec_optimal_value is not None:
                    optimal_value = exec_optimal_value
                    optimal_value_extracted = True
            else:
                # Try to extract from verification response
                from src.utils.robust_optimal_value_extraction import (
                    extract_optimal_value,
                )

                extracted_value = extract_optimal_value(verification_content)
                if extracted_value is not None:
                    optimal_value = extracted_value
                    optimal_value_extracted = True
                # For CSP problems, if we have a feasible solution but no optimal value, still consider it valid
                elif has_feasible_solution:
                    optimal_value = "FEASIBLE_SOLUTION_FOUND"
                    optimal_value_extracted = True

        # Record verification history for loop detection
        verification_history = state.get("verification_history", [])
        verification_history.append(verification_content)

        # Track error patterns
        error_patterns = state.get("error_patterns", {})
        if not is_approved:
            # Extract key error indicators for pattern matching
            error_key = (
                verification_content[:200] if verification_content else "unknown_error"
            )
            error_patterns[error_key] = error_patterns.get(error_key, 0) + 1
        
        # Determine problem type for COP optimization tracking
        # First try csp_cop_classification, then fallback to problem_type
        problem_type = state.get("csp_cop_classification", state.get("problem_type", "UNKNOWN"))
        
        # Debug: Log available state keys and problem type
        available_keys = [key for key in state.keys() if 'csp' in key.lower() or 'cop' in key.lower() or 'problem' in key.lower()]
        logger.info(f"Verifier node - Available problem-related keys: {available_keys}")
        logger.info(f"Verifier node - Problem type: {problem_type}")
        
        # For COP problems, track and compare optimal values
        # Initialize is_better_solution with default value
        is_better_solution = False
        solution_quality_assessment = None
        
        if problem_type == "COP" and optimal_value_extracted and optimal_value != "FEASIBLE_SOLUTION_FOUND":
            current_best_optimal_value = state.get("current_best_optimal_value", None)
            current_best_solution = state.get("current_best_solution", {})
            
            # Check if this is a better solution
            try:
                current_optimal_float = float(optimal_value)
                
                # Enhanced optimality check for known problems
                from src.utils.optimality_checker import check_solution_quality
                
                # Perform solution quality assessment
                problem_statement = state.get("problem_statement", "")
                quality_result = check_solution_quality(problem_statement, current_optimal_float, execution_result)
                solution_quality_assessment = quality_result
                
                logger.info(f"Solution quality assessment: {quality_result['assessment']}")
                
                if current_best_optimal_value is None:
                    # First solution found
                    is_better_solution = True
                    
                    # Check if this solution is already optimal for known problems
                    if quality_result.get("is_optimal", False):
                        logger.info("First solution found is optimal!")
                else:
                    # Check if this solution is better
                    # For minimization problems, lower is better
                    # For maximization problems, higher is better
                    # We assume minimization by default (can be enhanced with problem analysis)
                    if current_optimal_float < float(current_best_optimal_value):
                        is_better_solution = True
                        
                        # Check if we've reached optimality for known problems
                        if quality_result.get("is_optimal", False):
                            logger.info("Optimal solution found!")
                            
            except (ValueError, TypeError):
                # Cannot compare numerically, use string comparison as fallback
                if current_best_optimal_value is None:
                    is_better_solution = True
            
        # Initialize updated_state early to avoid scope issues
        updated_state = {
            "verification_result": verification_content,
            "verification_passed": is_approved,
            "verification_failed": not is_approved,
            "verification_history": verification_history,
            "error_patterns": error_patterns,
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": verification_content}],
        }

        if problem_type == "COP" and is_better_solution:
            logger.info(f"Found better COP solution: {optimal_value} (previous best: {current_best_optimal_value})")
            updated_state["current_best_optimal_value"] = optimal_value
            updated_state["current_best_solution"] = {
                "model": current_model,
                "code": current_code,
                "variable_values": variable_values,
                "execution_result": execution_result
            }
            updated_state["solution_improved"] = True
            
            # Store solution quality assessment for optimality tracking
            if solution_quality_assessment:
                updated_state["solution_quality_assessment"] = solution_quality_assessment
                
                # If solution is optimal for known problem, mark it as such
                if solution_quality_assessment.get("is_optimal", False):
                    updated_state["solution_is_optimal"] = True
                    logger.info("Marked solution as optimal based on known problem type")

        # Always store variable values and feasible solution status regardless of verification result
        # This ensures consistent reporting throughout the system
        updated_state["has_feasible_solution"] = has_feasible_solution
        updated_state["variable_values"] = variable_values
        
        # For CSP problems: output results if we have a feasible solution
        # For COP problems: continue optimization even if we have a feasible solution
        if problem_type == "CSP":
            should_output_results = is_approved or has_feasible_solution
            logger.info(f"CSP problem - should_output_results: {should_output_results}")
        elif problem_type == "COP":
            # For COP problems, we should continue optimization even if we have a feasible solution
            # Use advanced convergence detection for stopping decisions
            optimization_iterations = state.get("optimization_iterations", 0)
            max_optimization_iterations = state.get("max_optimization_iterations", 3)
            
            # Enhanced stopping condition with convergence detection
            is_solution_optimal = solution_quality_assessment and solution_quality_assessment.get("is_optimal", False)
            
            # Get current optimal value for convergence analysis
            current_optimal_float = None
            if optimal_value_extracted and optimal_value != "FEASIBLE_SOLUTION_FOUND":
                try:
                    current_optimal_float = float(optimal_value)
                except (ValueError, TypeError):
                    pass
            
            # Use advanced convergence detection for unknown problems
            if current_optimal_float is not None:
                # Get optimization history for convergence analysis
                optimization_history = state.get("optimization_history", [])
                
                # Create convergence metric for current iteration
                previous_value = None
                if optimization_history:
                    previous_value = optimization_history[-1].current_value
                
                current_metric = create_convergence_metric(
                    iteration=optimization_iterations,
                    current_value=current_optimal_float,
                    previous_value=previous_value
                )
                
                # Update optimization history
                optimization_history.append(current_metric)
                updated_state["optimization_history"] = optimization_history
                
                # Use advanced decision making for unknown problems
                if not is_solution_optimal:
                    decision_result = should_continue_optimization_func(
                        problem_statement=state.get("problem_statement", ""),
                        current_value=current_optimal_float,
                        optimization_history=optimization_history,
                        execution_result=execution_result
                    )
                    
                    logger.info(f"Convergence analysis: {decision_result['reason']} (confidence: {decision_result['confidence']})")
                    
                    # Use convergence-based decision
                    should_continue_optimization = (
                        optimization_iterations < max_optimization_iterations and 
                        decision_result["continue"]
                    )
                else:
                    # Known optimal solution found
                    should_continue_optimization = False
            else:
                # Enhanced fallback logic for unknown problems when optimal value cannot be extracted
                # Use solution quality assessment and convergence analysis even without exact optimal value
                
                # Check if we have any optimization history for convergence analysis
                optimization_history = state.get("optimization_history", [])
                
                if optimization_history:
                    # We have some optimization history, use convergence analysis
                    # Try to estimate current value from execution result
                    estimated_value = None
                    if execution_result.get("optimal_value"):
                        try:
                            estimated_value = float(execution_result["optimal_value"])
                        except (ValueError, TypeError):
                            pass
                    
                    if estimated_value is not None:
                        # Use convergence analysis with estimated value
                        decision_result = should_continue_optimization_func(
                            problem_statement=state.get("problem_statement", ""),
                            current_value=estimated_value,
                            optimization_history=optimization_history,
                            execution_result=execution_result
                        )
                        
                        logger.info(f"Fallback convergence analysis: {decision_result['reason']} (confidence: {decision_result['confidence']})")
                        
                        should_continue_optimization = (
                            optimization_iterations < max_optimization_iterations and 
                            decision_result["continue"]
                        )
                    else:
                        # Basic fallback: continue if we haven't reached max iterations
                        should_continue_optimization = optimization_iterations < max_optimization_iterations
                        logger.info(f"Fallback: no value available, continuing based on iteration count: {optimization_iterations < max_optimization_iterations}")
                else:
                    # First iteration, no history available - use basic logic
                    should_continue_optimization = optimization_iterations < max_optimization_iterations
                    logger.info(f"First iteration fallback: continuing based on iteration count: {optimization_iterations < max_optimization_iterations}")
            
            # CRITICAL FIX: If solver execution failed, DO NOT continue optimization
            # Check if execution was successful
            if execution_result.get("status") == "FAIL" or not execution_result.get("executed", False):
                should_continue_optimization = False
                logger.info("CRITICAL: Solver execution failed, stopping optimization and triggering correction")
            
            if should_continue_optimization:
                should_output_results = False
                updated_state["optimization_iterations"] = optimization_iterations + 1
                logger.info(f"COP problem - continuing optimization (iteration {optimization_iterations + 1}/{max_optimization_iterations})")
            else:
                should_output_results = True
                if is_solution_optimal:
                    logger.info(f"COP problem - stopping optimization: optimal solution found")
                elif current_optimal_float is not None and optimization_history:
                    # Check convergence status for logging
                    convergence_result = assess_convergence(optimization_history)
                    if convergence_result["converged"]:
                        logger.info(f"COP problem - stopping optimization: converged after {optimization_iterations} iterations")
                    elif convergence_result["stagnated"]:
                        logger.info(f"COP problem - stopping optimization: stagnated after {optimization_iterations} iterations")
                    else:
                        logger.info(f"COP problem - stopping optimization after {optimization_iterations} iterations")
                else:
                    logger.info(f"COP problem - stopping optimization after {optimization_iterations} iterations")
        else:
            # Default behavior for unknown problem types
            should_output_results = is_approved or has_feasible_solution
            logger.info(f"Unknown problem type - should_output_results: {should_output_results}")
        
        if should_output_results:
            logger.info("Solution verification passed or feasible solution found - stopping optimization")
            updated_state["solution_complete"] = True
            if optimal_value_extracted:
                # Handle special case for CSP problems
                if optimal_value == "FEASIBLE_SOLUTION_FOUND":
                    updated_state["optimal_value"] = "FEASIBLE_SOLUTION_FOUND"
                    updated_state["optimal_value_extracted"] = True
                    logger.info("Feasible solution found for CSP problem")
                else:
                    try:
                        updated_state["optimal_value"] = float(optimal_value)
                        updated_state["optimal_value_extracted"] = True
                        logger.info(f"Optimal value extracted: {optimal_value}")
                    except (ValueError, TypeError):
                        # If we can't convert to float but have a feasible solution, still mark as successful
                        updated_state["optimal_value"] = "FEASIBLE_SOLUTION_FOUND"
                        updated_state["optimal_value_extracted"] = True
                        logger.info("Feasible solution found (non-numeric optimal value)")
            else:
                # If we have a feasible solution but no optimal value (CSP case), still mark as successful
                if has_feasible_solution:
                    updated_state["optimal_value"] = "FEASIBLE_SOLUTION_FOUND"
                    updated_state["optimal_value_extracted"] = True
                    logger.info("Feasible solution found (CSP problem)")
                else:
                    updated_state["optimal_value_extracted"] = False
                    logger.warning("Could not extract optimal value")
        else:
            logger.info("Solution verification failed and no feasible solution found, needs correction")
            updated_state["correction_needed"] = True

            # Log verification failure
            try:
                failure_logger = config.get("configurable", {}).get("failure_logger")
                if failure_logger:
                    problem_id = state.get("problem_id", "unknown")
                    attempt = state.get("current_attempt", 0)
                    failure_logger.log_verification_failure(
                        problem_id=problem_id,
                        problem_statement=state.get("problem_statement", ""),
                        current_model=current_model,
                        current_code=current_code,
                        verification_result=verification_content,
                        attempt=attempt,
                        correction_count=state.get("correction_count", 0),
                    )
            except Exception as log_error:
                logger.debug(f"Failed to log verification failure: {log_error}")

        return updated_state

    except Exception as e:
        from src.utils.error_handling import classify_error

        error_type, is_retryable = classify_error(e)
        error_msg = str(e)
        logger.error(f"Verification failed ({error_type.value}): {error_msg}")

        return {
            "verification_result": f"VERIFICATION ERROR ({error_type.value}): {error_msg}",
            "verification_passed": False,
            "verification_failed": True,
            "correction_needed": True,
            "error_type": error_type.value,
            "is_retryable": is_retryable,
        }


async def corrector_node(state: OptAgentState, config) -> Dict[str, Any]:
    """
    Corrector node for fixing solutions based on verification feedback.

    This node analyzes verification feedback and generates corrected
    mathematical models and Python code.

    Args:
        state: Current OptAgent state
        config: LangGraph node configuration

    Returns:
        State update dictionary with corrected solution
    """
    logger.info("Corrector node fixing solution based on verification feedback")

    from src.config.agents import get_llm_by_type, AGENT_LLM_MAP
    from src.prompts.template import apply_prompt_template
    from src.utils.robust_code_extraction import (
        extract_python_code as robust_extract_python_code,
    )

    # Prepare state for template
    template_state = dict(state)

    # Apply prompt template
    messages = apply_prompt_template("corrector", template_state)

    # Add specific correction task
    correction_task = f"""
# CORRECTION TASK

## Original Problem Statement
{state.get('problem_statement', 'Not available')}

## Previous Mathematical Model
{state.get('current_model', 'Not available')}

## Previous Python Code
```python
{state.get('current_code', 'Not available')}
```

## Verification Feedback
{state.get('verification_result', 'Not available')}

"""
    
    # Prepare correction task based on verification feedback and problem type
    # First try csp_cop_classification, then fallback to problem_type
    problem_type = state.get("csp_cop_classification", state.get("problem_type", "UNKNOWN"))
    optimization_iterations = state.get("optimization_iterations", 0)
    
    # Enhanced correction guidance with optimality information
    solution_quality = state.get("solution_quality_assessment", {})
    known_optimal_value = solution_quality.get("known_optimal_value")
    quality_gap = solution_quality.get("quality_gap")
    
    if problem_type == "COP" and optimization_iterations > 0:
        # For COP problems during optimization iterations, focus on improving the solution
        current_optimal_value = state.get("current_best_optimal_value", None)
        
        # Enhanced correction guidance with optimality information
        optimality_guidance = ""
        if known_optimal_value is not None:
            if quality_gap is not None and quality_gap > 0:
                optimality_guidance = f"""

## OPTIMALITY GUIDANCE
This is a known optimization problem. The theoretical optimal value is {known_optimal_value}.
Current solution has a gap of {quality_gap:.2f} units ({quality_gap/known_optimal_value*100:.1f}%) from optimal.

Please focus on closing this gap by:
- Exploring different search strategies
- Tightening constraint formulations
- Considering alternative modeling approaches
"""
            else:
                optimality_guidance = f"""

## OPTIMALITY GUIDANCE
This is a known optimization problem. The theoretical optimal value is {known_optimal_value}.
Current solution is close to optimal - focus on fine-tuning.
"""
        
        correction_task += f"""
You are optimizing a Constraint Optimization Problem (COP). We are currently on optimization iteration {optimization_iterations}.

Current best optimal value: {current_optimal_value if current_optimal_value else 'Not yet established'}
{optimality_guidance}

Please analyze the feedback and provide an IMPROVED mathematical model and Python code that:
1. Maintains all constraints from the previous model
2. Attempts to find a BETTER optimal value (lower for minimization, higher for maximization)
3. May try different search strategies or constraint formulations
4. Focuses on improving the objective function value

If the current solution is already good, suggest minor improvements or alternative approaches.
"""
    else:
        # For CSP problems or first iteration of COP, focus on correctness
        correction_task += f"""
Please analyze the feedback and provide corrected mathematical model and Python code.
"""
    
    messages.append({"role": "user", "content": correction_task})

    # Execute correction
    llm = get_llm_by_type(AGENT_LLM_MAP.get("corrector", "basic"))

    response = await llm.ainvoke(messages)
    
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    # Extract corrected model and code (same format as modeler)
    model_content = ""
    code_content = ""

    # Extract model section (between "## Mathematical Model" and "## Python Implementation")
    import re

    model_match = re.search(
        r"## Mathematical Model\s*\n(.*?)(?=## Python Implementation)",
        content,
        re.DOTALL,
    )
    if model_match:
        model_content = model_match.group(1).strip()

    # Extract code using robust extraction
    code_content = robust_extract_python_code(content)

    if not model_content and not code_content:
        logger.warning("Failed to extract corrected model and code from response")
        model_content = content  # Use full content as fallback

    # Update correction count and solution version
    correction_count = state.get("correction_count", 0) + 1
    solution_version = state.get("solution_version", 0) + 1

    # Add to correction history
    correction_history = state.get("correction_history", [])
    correction_history.append(
        f"Correction #{correction_count}: {state.get('verification_result', '')[:100]}..."
    )

    return {
        "current_model": model_content,
        "current_code": code_content,
        "solution_version": solution_version,
        "correction_count": correction_count,
        "correction_history": correction_history,
        "correction_needed": False,  # Reset correction flag
        "verification_passed": False,  # Reset verification status
        "verification_failed": False,
        "messages": state.get("messages", [])
        + [{"role": "assistant", "content": content}],
    }


async def visualizer_node(state: OptAgentState, config) -> Dict[str, Any]:
    """
    Visualizer node for creating charts and visualizations based on the optimization solution.

    This node generates appropriate visualizations to help understand the optimization
    problem structure, solution, and results.

    Args:
        state: Current OptAgent state
        config: LangGraph node configuration

    Returns:
        State update dictionary with visualization code and status
    """
    logger.info("Visualizer node generating charts and visualizations")

    from src.config.agents import get_llm_by_type, AGENT_LLM_MAP
    from src.prompts.template import apply_prompt_template
    from src.utils.robust_code_extraction import (
        extract_python_code as robust_extract_python_code,
    )
    import os
    import tempfile
    import subprocess
    import sys

    # Prepare state for template
    template_state = dict(state)
    # Ensure output_dir is in template state
    template_state["output_dir"] = state.get("output_dir", ".")

    # Apply prompt template
    messages = apply_prompt_template("visualizer", template_state)
    llm = get_llm_by_type(AGENT_LLM_MAP.get("visualizer", "basic"))

    response = await llm.ainvoke(messages)
    
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    # Extract Python code for visualization
    visualization_code = robust_extract_python_code(content)

    if not visualization_code:
        logger.warning("No valid Python visualization code found in response")
        return {
            "visualization_code": "",
            "visualization_status": "failed",
            "visualization_error": "No valid Python code found in LLM response",
        }

    # Try to execute visualization code to generate plots
    visualization_status = "success"
    visualization_error = None
    plot_files = []

    try:
        # Get the output directory for saving plots
        output_dir = state.get("output_dir", ".")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create a temporary file to execute the visualization code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            # Write the visualization code directly (output_dir is already passed via template)
            temp_file.write(visualization_code)
            temp_file_path = temp_file.name

        # Execute the visualization code (run from project root, not output_dir)
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout for visualization
            cwd=os.getcwd(),  # Run from current working directory (project root)
        )

        if result.returncode != 0:
            visualization_status = "failed"
            visualization_error = (
                result.stderr or "Unknown error during visualization execution"
            )
            logger.error(f"Visualization execution failed: {visualization_error}")
        else:
            # Find generated plot files
            if os.path.exists(output_dir):
                plot_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
            logger.info(
                f"Visualization successful. Generated {len(plot_files)} plot files: {plot_files}"
            )

        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

    except subprocess.TimeoutExpired:
        visualization_status = "failed"
        visualization_error = "Visualization execution timed out"
        logger.error("Visualization execution timed out")
    except Exception as e:
        visualization_status = "failed"
        visualization_error = str(e)
        logger.error(f"Error executing visualization code: {e}")

    return {
        "visualization_code": visualization_code,
        "visualization_status": visualization_status,
        "visualization_error": visualization_error,
        "plot_files": plot_files,
        "messages": state.get("messages", [])
        + [{"role": "assistant", "content": content}],
    }


def reporter_node(state: OptAgentState, config) -> Dict[str, Any]:
    """
    Reporter node for generating final solution reports.

    This node generates a deterministic final report without using LLM.

    Args:
        state: Current OptAgent state
        config: LangGraph node configuration

    Returns:
        State update dictionary with final solution report
    """
    logger.info("Reporter generating final solution report")
    
    # Debug: Log the entire state to see what's available
    logger.info(f"Reporter state keys: {list(state.keys())}")
    logger.info(f"Reporter state variable_values: {state.get('variable_values', 'NOT FOUND')}")

    # For COP problems, use the best solution found during optimization
    # First try csp_cop_classification, then fallback to problem_type
    problem_type = state.get("csp_cop_classification", state.get("problem_type", "UNKNOWN"))
    
    if problem_type == "COP" and state.get("current_best_solution"):
        # Use the best solution found during optimization iterations
        best_solution = state.get("current_best_solution", {})
        mathematical_model = best_solution.get("model", state.get("current_model", "No mathematical model available"))
        optimization_code = best_solution.get("code", state.get("current_code", "No optimization code available"))
        variable_values = best_solution.get("variable_values", state.get("variable_values", {}))
        optimal_value = state.get("current_best_optimal_value", state.get("optimal_value", 0.0))
        logger.info(f"COP optimization complete - using best solution with optimal value: {optimal_value}")
    else:
        # Use the current solution (for CSP or single-iteration COP)
        mathematical_model = state.get("current_model", "No mathematical model available")
        optimization_code = state.get("current_code", "No optimization code available")
        variable_values = state.get("variable_values", {})
        optimal_value = state.get("optimal_value", 0.0)
    
    problem_statement = state.get("problem_statement", "No problem statement provided")
    verification_result = state.get("verification_result", "No verification performed")

    # Convert to strings if needed
    if isinstance(problem_statement, list):
        problem_statement = "\n".join(str(item) for item in problem_statement)
    elif not isinstance(problem_statement, str):
        problem_statement = str(problem_statement)

    if isinstance(mathematical_model, list):
        mathematical_model = "\n".join(str(item) for item in mathematical_model)
    elif not isinstance(mathematical_model, str):
        mathematical_model = str(mathematical_model)

    if isinstance(optimization_code, list):
        optimization_code = "\n".join(str(item) for item in optimization_code)
    elif not isinstance(optimization_code, str):
        optimization_code = str(optimization_code)

    # Get solution status
    verification_passed = state.get("verification_passed", False)
    optimal_value_extracted = state.get("optimal_value_extracted", False)
    has_feasible_solution = state.get("has_feasible_solution", False)
    
    # Debug: Check variable_values status
    logger.info(f"Reporter: variable_values type={type(variable_values)}, length={len(variable_values)}, content={variable_values}")

    # Determine solution type and display values
    # CRITICAL FIX: Respect the original problem classification (CSP/COP) from state
    # Do not override based solely on execution results
    
    # Get the original problem classification from state
    original_problem_type = state.get("csp_cop_classification", state.get("problem_type", "UNKNOWN"))
    logger.info(f"Reporter: Original problem type = {original_problem_type}")
    
    if optimal_value_extracted and optimal_value == "INFEASIBLE":
        # Explicitly handle infeasible problems
        solution_type = "No Valid Solution (Infeasible)"
        optimal_value_display = "INFEASIBLE"
    elif original_problem_type == "COP":
        # This is a COP problem - respect the original classification
        if optimal_value_extracted and optimal_value != "FEASIBLE_SOLUTION_FOUND":
            # Found optimal solution for COP
            solution_type = "Optimal Solution (COP)"
            try:
                optimal_value_display = f"{float(optimal_value):.6f}"
            except (ValueError, TypeError):
                optimal_value_display = str(optimal_value)
        elif has_feasible_solution:
            # COP problem with feasible solution but no optimal value (e.g., timeout)
            solution_type = "Feasible Solution (COP - Optimization Incomplete)"
            optimal_value_display = None
        else:
            # COP problem with no solution found
            solution_type = "No Valid Solution (COP)"
            optimal_value_display = None
    elif original_problem_type == "CSP":
        # This is a CSP problem - respect the original classification
        if has_feasible_solution:
            solution_type = "Feasible Solution (CSP)"
            optimal_value_display = None  # CSP problems don't have meaningful optimal values
        else:
            solution_type = "No Valid Solution (CSP)"
            optimal_value_display = None
    else:
        # Fallback for unknown problem types
        if has_feasible_solution:
            solution_type = "Feasible Solution"
            optimal_value_display = None
        else:
            solution_type = "No Valid Solution"
            optimal_value_display = None

    # Generate report
    report_lines = [
        "# Optimization Solution Report",
        "",
        "## Problem Description",
        problem_statement,
        "",
        "## Mathematical Model",
        mathematical_model,
        "",
        "## Python Implementation",
        f"```python\n{optimization_code}\n```",
        "",
        "## Results",
    ]

    # Ensure consistent status reporting - if we have a feasible solution, 
    # the status should reflect that regardless of verification_passed
    # This prevents contradictions like "Status: ✅ Successfully solved" with "Feasible Solution: No"
    effective_verification_passed = verification_passed or has_feasible_solution

    if effective_verification_passed:
        report_lines.extend(
            [
                f"- **Status**: ✅ Successfully solved",
                f"- **Solution Type**: {solution_type}",
                f"- **Optimal Value**: {optimal_value_display}",
                f"- **Feasible Solution**: {'Yes' if has_feasible_solution else 'No'}",
                f"- **Verification**: {'✅ Passed' if verification_passed else '⚠️ Feasible solution found but verification incomplete'}",
            ]
        )
        
        # 添加实际的变量值显示
        if variable_values:
            report_lines.extend([
                "",
                "## Variable Values",
            ])
            for var_name, var_value in variable_values.items():
                report_lines.append(f"- {var_name} = {var_value}")
        else:
            # Debug: Add information about why variable values are missing
            report_lines.extend([
                "",
                "## Variable Values",
                "- No variable values found in solution"
            ])
            logger.warning(f"Variable values missing from report. Current value: {variable_values}")
    else:
        correction_count = state.get("correction_count", 0)
        max_corrections = state.get("max_corrections", 5)
        # Note: has_feasible_solution is already retrieved above, no need to retrieve again

        if correction_count >= max_corrections:
            report_lines.extend(
                [
                    f"- **Status**: ❌ Solution failed (max corrections exceeded)",
                    f"- **Solution Type**: {solution_type}",
                    f"- **Optimal Value**: {optimal_value_display}",
                    f"- **Feasible Solution**: {'Yes' if has_feasible_solution else 'No'}",
                    f"- **Verification**: ❌ Failed after {correction_count} corrections",
                ]
            )
        else:
            report_lines.extend(
                [
                    f"- **Status**: ⚠️ Solution in progress",
                    f"- **Solution Type**: {solution_type}",
                    f"- **Optimal Value**: {optimal_value_display}",
                    f"- **Feasible Solution**: {'Yes' if has_feasible_solution else 'No'}",
                    f"- **Verification**: ❌ Failed ({correction_count} corrections attempted)",
                ]
            )

    # Include visualization information if available
    visualization_status = state.get("visualization_status")
    if visualization_status:
        plot_files = state.get("plot_files", [])
        report_lines.extend(
            [
                "",
                "## Visualization",
                f"- **Status**: {visualization_status}",
                f"- **Generated Plots**: {', '.join(plot_files) if plot_files else 'None'}",
            ]
        )
        if state.get("visualization_error"):
            report_lines.append(f"- **Error**: {state.get('visualization_error')}")

    report_content = "\n".join(report_lines)

    # Create final solution package
    # Ensure consistent status reporting - if we have a feasible solution, 
    # the status should reflect that regardless of verification_passed
    # This prevents contradictions in the final solution status
    effective_verification_passed = verification_passed or has_feasible_solution
    
    final_solution = {
        "problem_statement": str(state.get("problem_statement", "")),
        "mathematical_model": str(state.get("current_model", "")),
        "optimization_code": str(state.get("current_code", "")),
        "verification_result": str(state.get("verification_result", "")),
        "optimal_value": optimal_value if optimal_value_extracted else None,
        "optimal_value_extracted": optimal_value_extracted,
        "has_feasible_solution": has_feasible_solution,
        "variable_values": variable_values,  # 包含变量值
        "final_report": str(report_content),
        "solution_status": "SUCCESS" if effective_verification_passed else "PARTIAL",
        "correction_count": state.get("correction_count", 0),
        "solution_version": state.get("solution_version", 0),
        "visualization_status": visualization_status,
        "plot_files": state.get("plot_files", []),
    }

    return {
        "final_solution": final_solution,
        "solution_complete": True,
        "messages": state.get("messages", [])
        + [{"role": "assistant", "content": str(report_content)}],
    }
