# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def robust_approval_check(text: str) -> Tuple[bool, float]:
    """
    健壮的批准检查 - 更严格的检查，避免误判

    Args:
        text: 要检查的文本内容

    Returns:
        Tuple[bool, float]: (是否通过, 置信度 0.0-1.0)
    """
    if not text or not isinstance(text, str):
        return False, 0.0

    confidence = 0.0
    text_upper = text.upper()

    # 新的简化格式检查：直接的APPROVED/REJECTED指示
    if "✅ APPROVED" in text:
        confidence += 0.9
    elif "✅APPROVED" in text:  # 无空格版本
        confidence += 0.8
    elif "✅ Approved" in text:  # 大小写变化
        confidence += 0.8
    elif re.search(r"^✅\s+APPROVED", text, re.MULTILINE | re.IGNORECASE):
        confidence += 0.8
    elif re.search(r"\bAPPROVED\b", text, re.IGNORECASE):
        confidence += 0.6

    # 检查结论性语句
    if re.search(r"Optimal\s+Value.*\d", text, re.IGNORECASE):
        confidence += 0.3  # 有提取到最优值是好信号

    # 否定检查：拒绝指标（更强力的否定）
    if "❌ REJECTED" in text:
        confidence -= 0.9  # 明确的拒绝
    elif "❌REJECTED" in text:
        confidence -= 0.8
    elif "❌ Rejected" in text:
        confidence -= 0.8
    elif re.search(r"^❌\s+REJECTED", text, re.MULTILINE | re.IGNORECASE):
        confidence -= 0.8
    elif "REJECTED" in text_upper:
        confidence -= 0.6

    # 执行错误检查
    if re.search(r"Execution\s+Error\s+Found", text, re.IGNORECASE):
        confidence -= 0.8
    if re.search(r"Issues\s+Found", text, re.IGNORECASE):
        confidence -= 0.6
    if re.search(r"SyntaxError", text, re.IGNORECASE):
        confidence -= 0.8  # 语法错误是致命问题
    if re.search(r"RuntimeError", text, re.IGNORECASE):
        confidence -= 0.8

    # 最终判断 - 提高通过门槛
    is_approved = confidence > 0.7  # 从0.5提高到0.7，更严格

    logger.debug(
        f"Approval check: text_length={len(text)}, confidence={confidence:.2f}, approved={is_approved}"
    )

    return is_approved, confidence


def robust_faithful_check(text: str) -> Tuple[bool, float]:
    """
    健壮的忠实度检查

    Args:
        text: 要检查的文本内容

    Returns:
        Tuple[bool, float]: (是否忠实, 置信度 0.0-1.0)
    """
    if not text or not isinstance(text, str):
        return False, 0.0

    confidence = 0.0
    text_upper = text.upper()

    # 主要检查：精确匹配
    if "✅ FAITHFUL" in text:
        confidence += 0.8
    elif "✅FAITHFUL" in text:  # 无空格版本
        confidence += 0.7
    elif "✅ Faithful" in text:  # 大小写变化
        confidence += 0.7

    # 次要检查：关键词模式
    if re.search(r"\bFAITHFUL\b", text, re.IGNORECASE):
        confidence += 0.4
    if re.search(r"\bVERIFICATION.*RESULT.*FAITHFUL\b", text, re.IGNORECASE):
        confidence += 0.3
    if re.search(r"\bCODE.*FAITHFUL\b", text, re.IGNORECASE):
        confidence += 0.3

    # 否定检查：拒绝指标
    if "❌" in text or "UNFAITHFUL" in text_upper:
        confidence -= 0.5
    if re.search(r"\bMISMATCH\b", text, re.IGNORECASE):
        confidence -= 0.3
    if re.search(r"\bINCORRECT\b", text, re.IGNORECASE):
        confidence -= 0.3

    is_faithful = confidence > 0.5

    logger.debug(
        f"Faithful check: text_length={len(text)}, confidence={confidence:.2f}, faithful={is_faithful}"
    )

    return is_faithful, confidence


def robust_pass_check(text: str) -> Tuple[bool, float]:
    """
    健壮的通过检查（用于语法验证等）

    Args:
        text: 要检查的文本内容

    Returns:
        Tuple[bool, float]: (是否通过, 置信度 0.0-1.0)
    """
    if not text or not isinstance(text, str):
        return False, 0.0

    confidence = 0.0
    text_upper = text.upper()

    # 主要检查：精确匹配
    if text.startswith("PASS"):
        confidence += 0.8
    elif "PASS:" in text:
        confidence += 0.7
    elif re.search(r"\bPASS\b", text, re.IGNORECASE):
        confidence += 0.6

    # 次要检查：成功指标
    if re.search(r"\bSUCCESS\b", text, re.IGNORECASE):
        confidence += 0.4
    if re.search(r"\bVALID\b", text, re.IGNORECASE):
        confidence += 0.3
    if "✅" in text:
        confidence += 0.3

    # 否定检查：失败指标
    if text.startswith("FAIL"):
        confidence -= 0.8
    elif "FAIL:" in text:
        confidence -= 0.7
    elif re.search(r"\bFAIL\b", text, re.IGNORECASE):
        confidence -= 0.6
    if "❌" in text:
        confidence -= 0.3
    if re.search(r"\bERROR\b", text, re.IGNORECASE):
        confidence -= 0.4

    is_pass = confidence > 0.5

    logger.debug(
        f"Pass check: text_length={len(text)}, confidence={confidence:.2f}, pass={is_pass}"
    )

    return is_pass, confidence


def extract_verification_status(
    text: str, verification_type: str = "approval"
) -> Tuple[bool, float, str]:
    """
    统一的验证状态提取函数

    Args:
        text: 要检查的文本内容
        verification_type: 验证类型 ("approval", "faithful", "pass")

    Returns:
        Tuple[bool, float, str]: (验证结果, 置信度, 详细信息)
    """
    if verification_type == "approval":
        result, confidence = robust_approval_check(text)
        status = "APPROVED" if result else "REJECTED"
    elif verification_type == "faithful":
        result, confidence = robust_faithful_check(text)
        status = "FAITHFUL" if result else "UNFAITHFUL"
    elif verification_type == "pass":
        result, confidence = robust_pass_check(text)
        status = "PASS" if result else "FAIL"
    else:
        raise ValueError(f"Unknown verification type: {verification_type}")

    detail = f"Status: {status}, Confidence: {confidence:.2f}"

    logger.info(
        f"Verification extraction - Type: {verification_type}, Result: {result}, Detail: {detail}"
    )

    return result, confidence, detail
