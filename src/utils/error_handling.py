# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """错误类型分类"""

    NETWORK = "network"  # 网络相关错误
    LLM_SERVICE = "llm_service"  # LLM服务错误
    VALIDATION = "validation"  # 验证错误
    CODE_EXECUTION = "code_execution"  # 代码执行错误
    STATE_MANAGEMENT = "state_management"  # 状态管理错误
    UNKNOWN = "unknown"  # 未知错误


class RetryableError(Exception):
    """可重试的错误"""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.max_retries = max_retries
        self.retry_delay = retry_delay


class NonRetryableError(Exception):
    """不可重试的错误"""

    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN):
        super().__init__(message)
        self.error_type = error_type


def classify_error(error: Exception) -> Tuple[ErrorType, bool]:
    """
    对错误进行分类并判断是否可重试

    Args:
        error: 异常对象

    Returns:
        Tuple[ErrorType, bool]: (错误类型, 是否可重试)
    """
    error_msg = str(error).lower()
    error_type = type(error).__name__.lower()

    # 网络相关错误（通常可重试）
    if any(
        keyword in error_msg
        for keyword in ["timeout", "connection", "network", "http", "ssl", "socket"]
    ) or any(keyword in error_type for keyword in ["timeout", "connection", "http"]):
        return ErrorType.NETWORK, True

    # LLM服务错误（部分可重试）
    if (
        any(
            keyword in error_msg
            for keyword in [
                "rate limit",
                "quota",
                "service unavailable",
                "internal server error",
                "bad gateway",
                "gateway timeout",
            ]
        )
        or "openai" in error_type
        or "anthropic" in error_type
    ):
        # 速率限制和服务不可用可重试，其他不可重试
        retryable = any(
            keyword in error_msg
            for keyword in [
                "rate limit",
                "quota",
                "service unavailable",
                "internal server error",
                "bad gateway",
                "gateway timeout",
                "429",
                "500",
                "502",
                "503",
                "504",
            ]
        )
        return ErrorType.LLM_SERVICE, retryable

    # 验证错误（通常不可重试）
    if (
        any(
            keyword in error_msg
            for keyword in ["validation", "parse", "format", "syntax error"]
        )
        or "validation" in error_type
    ):
        return ErrorType.VALIDATION, False

    # 代码执行错误（通常不可重试）
    if any(
        keyword in error_msg
        for keyword in ["execution", "runtime", "import error", "module not found"]
    ) or error_type in ["syntaxerror", "importerror", "modulenotfounderror"]:
        return ErrorType.CODE_EXECUTION, False

    # 状态管理错误（通常不可重试）
    if any(
        keyword in error_msg for keyword in ["state", "key error", "attribute error"]
    ) or error_type in ["keyerror", "attributeerror"]:
        return ErrorType.STATE_MANAGEMENT, False

    # 默认未知错误（谨慎重试）
    return ErrorType.UNKNOWN, False


async def retry_with_backoff(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> Any:
    """
    带指数退避的异步重试机制

    Args:
        func: 要重试的函数
        args: 函数位置参数
        kwargs: 函数关键字参数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        backoff_factor: 退避因子
        jitter: 是否添加抖动

    Returns:
        函数执行结果

    Raises:
        最后一次尝试的异常
    """
    if kwargs is None:
        kwargs = {}

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            last_exception = e
            error_type, is_retryable = classify_error(e)

            # 如果是最后一次尝试或错误不可重试，直接抛出
            if attempt == max_retries or not is_retryable:
                logger.error(
                    f"Function {func.__name__} failed after {attempt + 1} attempts: {e}"
                )
                raise e

            # 计算延迟时间
            delay = min(base_delay * (backoff_factor**attempt), max_delay)

            # 添加抖动
            if jitter:
                import random

                delay *= 0.5 + random.random() * 0.5

            logger.warning(
                f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            await asyncio.sleep(delay)

    # 这行代码理论上不会执行到，但为了类型安全
    if last_exception:
        raise last_exception


class SafeStateManager:
    """安全的状态管理器"""

    @staticmethod
    def safe_get(state: Dict[str, Any], key: str, default: Any = None) -> Any:
        """安全地从状态中获取值"""
        try:
            return state.get(key, default)
        except Exception as e:
            logger.warning(f"Error getting state key '{key}': {e}")
            return default

    @staticmethod
    def safe_set(state: Dict[str, Any], key: str, value: Any) -> bool:
        """安全地向状态中设置值"""
        try:
            state[key] = value
            return True
        except Exception as e:
            logger.error(f"Error setting state key '{key}': {e}")
            return False

    @staticmethod
    def safe_increment(
        state: Dict[str, Any], key: str, increment: int = 1, default: int = 0
    ) -> int:
        """安全地递增状态值"""
        try:
            current_value = state.get(key, default)
            if not isinstance(current_value, int):
                current_value = default
            new_value = current_value + increment
            state[key] = new_value
            return new_value
        except Exception as e:
            logger.error(f"Error incrementing state key '{key}': {e}")
            return state.get(key, default)

    @staticmethod
    def validate_state_consistency(state: Dict[str, Any]) -> bool:
        """验证状态一致性"""
        try:
            # 检查关键字段的类型
            checks = [
                ("model_correction_count", int),
                ("code_correction_count", int),
                ("model_verified", bool),
                ("code_verified", bool),
                ("solution_complete", bool),
            ]

            for key, expected_type in checks:
                if key in state and not isinstance(state[key], expected_type):
                    logger.warning(
                        f"State inconsistency: {key} should be {expected_type}, got {type(state[key])}"
                    )
                    return False

            # 检查计数器的合理性
            model_count = state.get("model_correction_count", 0)
            code_count = state.get("code_correction_count", 0)
            max_model = state.get("max_model_corrections", 5)
            max_code = state.get("max_code_corrections", 5)

            if model_count < 0 or code_count < 0:
                logger.warning(
                    "State inconsistency: correction counts cannot be negative"
                )
                return False

            if model_count > max_model * 2 or code_count > max_code * 2:
                logger.warning(
                    "State inconsistency: correction counts exceed reasonable limits"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating state consistency: {e}")
            return False


def create_safe_fallback_state(original_state: Dict[str, Any]) -> Dict[str, Any]:
    """创建安全的回退状态"""
    try:
        # 保留关键的原始信息
        safe_state = {
            "problem_statement": original_state.get("problem_statement", ""),
            "messages": original_state.get("messages", []),
            "workflow_phase": "error_recovery",
            "model_correction_count": original_state.get("model_correction_count", 0),
            "code_correction_count": original_state.get("code_correction_count", 0),
            "max_model_corrections": original_state.get("max_model_corrections", 5),
            "max_code_corrections": original_state.get("max_code_corrections", 5),
            "model_verification_passed": False,
            "code_verification_passed": False,
            "solution_complete": False,
            "error_recovery": True,
            "last_error_timestamp": time.time(),
        }

        logger.info("Created safe fallback state for error recovery")
        return safe_state

    except Exception as e:
        logger.error(f"Failed to create fallback state: {e}")
        # 最小化的紧急状态
        return {
            "workflow_phase": "emergency",
            "error_recovery": True,
            "last_error_timestamp": time.time(),
        }


class CircuitBreaker:
    """熔断器模式实现"""

    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_proceed(self) -> bool:
        """检查是否可以继续执行"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """记录成功执行"""
        self.failure_count = 0
        if self.state != "CLOSED":
            self.state = "CLOSED"
            logger.info("Circuit breaker reset to CLOSED state")

    def record_failure(self):
        """记录失败执行"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def is_open(self) -> bool:
        """检查熔断器是否打开"""
        return self.state == "OPEN"
