# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
OptAgent统一超时配置系统
支持各个环节独立配置超时时间
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeoutConfig:
    """OptAgent统一超时配置类"""

    # ===== 总体工作流超时 =====
    total_workflow_timeout: int = 600  # 总工作流超时（秒）- 默认10分钟
    single_problem_timeout: int = 300  # 单个问题超时（秒）- 默认5分钟

    # ===== 各阶段超时设置 =====
    # Phase 1: 建模阶段
    modeling_timeout: int = 120  # 建模专家超时 - 默认2分钟
    model_verification_timeout: int = 180  # 模型验证超时 - 默认3分钟

    # Phase 2: 编码阶段
    coding_timeout: int = 120  # 代码生成超时 - 默认2分钟
    code_verification_timeout: int = (
        180  # 代码验证超时 - 默认3分钟（整合语法、忠实度、执行准备检查）
    )

    # Phase 3: 结果验证和最终化阶段
    result_verification_timeout: int = 120  # 结果验证超时 - 默认2分钟
    reporting_timeout: int = 60  # 报告生成超时 - 默认1分钟

    # ===== 底层执行超时 =====
    code_execution_timeout: int = 30  # 代码执行超时 - 默认30秒
    llm_request_timeout: int = 60  # LLM请求超时 - 默认1分钟（简化提示词后可以大幅减少）
    agent_recursion_limit: int = 15  # Agent递归限制
    langgraph_recursion_limit: int = 30  # LangGraph递归限制

    # ===== 批量评估模式超时 =====
    batch_mode: bool = False  # 是否启用批量模式
    batch_modeling_timeout: int = 60  # 批量模式建模超时 - 默认1分钟
    batch_model_verification_timeout: int = 90  # 批量模式模型验证超时 - 默认1.5分钟
    batch_code_verification_timeout: int = 90  # 批量模式代码验证超时 - 默认1.5分钟
    batch_result_verification_timeout: int = 60  # 批量模式结果验证超时 - 默认1分钟
    batch_execution_timeout: int = 30  # 批量模式执行超时 - 默认30秒

    # ===== 重试设置 =====
    max_retries: int = 3  # 最大重试次数
    retry_delay: float = 1.0  # 重试延迟（秒）

    # ===== 环境变量映射 =====
    env_var_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "OPTAG_TOTAL_TIMEOUT": "total_workflow_timeout",
            "OPTAG_PROBLEM_TIMEOUT": "single_problem_timeout",
            "OPTAG_MODELING_TIMEOUT": "modeling_timeout",
            "OPTAG_MODEL_VERIFICATION_TIMEOUT": "model_verification_timeout",
            "OPTAG_CODING_TIMEOUT": "coding_timeout",
            "OPTAG_CODE_VERIFICATION_TIMEOUT": "code_verification_timeout",
            "OPTAG_RESULT_VERIFICATION_TIMEOUT": "result_verification_timeout",
            "OPTAG_REPORTING_TIMEOUT": "reporting_timeout",
            "OPTAG_CODE_EXEC_TIMEOUT": "code_execution_timeout",
            "OPTAG_LLM_TIMEOUT": "llm_request_timeout",
            "OPTAG_AGENT_RECURSION_LIMIT": "agent_recursion_limit",
            "OPTAG_LANGGRAPH_RECURSION_LIMIT": "langgraph_recursion_limit",
            "OPTAG_BATCH_MODE": "batch_mode",
            "OPTAG_MAX_RETRIES": "max_retries",
        }
    )

    def __post_init__(self):
        """初始化后处理，从环境变量加载配置"""
        self.load_from_environment()
        self.validate_timeouts()

    def load_from_environment(self):
        """从环境变量加载超时配置"""
        for env_var, field_name in self.env_var_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if field_name == "batch_mode":
                        # 布尔值处理
                        setattr(
                            self,
                            field_name,
                            env_value.lower() in ["true", "1", "yes", "on"],
                        )
                    elif field_name == "retry_delay":
                        # 浮点数处理
                        setattr(self, field_name, float(env_value))
                    else:
                        # 整数处理
                        setattr(self, field_name, int(env_value))
                    logger.info(
                        f"Loaded timeout config from environment: {field_name} = {getattr(self, field_name)}"
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid environment variable {env_var}={env_value}: {e}"
                    )

    def validate_timeouts(self):
        """验证超时配置的合理性"""
        # 确保所有超时值为正数
        timeout_fields = [
            "total_workflow_timeout",
            "single_problem_timeout",
            "modeling_timeout",
            "model_verification_timeout",
            "coding_timeout",
            "code_verification_timeout",
            "result_verification_timeout",
            "reporting_timeout",
            "code_execution_timeout",
            "llm_request_timeout",
        ]

        for field_name in timeout_fields:
            value = getattr(self, field_name)
            if value <= 0:
                logger.warning(
                    f"Invalid timeout value for {field_name}: {value}, setting to default"
                )
                setattr(self, field_name, 60)  # 默认1分钟

        # 验证递归限制
        if self.agent_recursion_limit <= 0:
            self.agent_recursion_limit = 25
        if self.langgraph_recursion_limit <= 0:
            self.langgraph_recursion_limit = 50

        # 验证层级关系：总超时应该大于各阶段超时之和
        stage_timeouts_sum = (
            self.modeling_timeout
            + self.model_verification_timeout
            + self.coding_timeout
            + self.code_verification_timeout
            + self.result_verification_timeout
            + self.reporting_timeout
        )

        if self.single_problem_timeout < stage_timeouts_sum:
            recommended_timeout = int(stage_timeouts_sum * 1.5)
            logger.warning(
                f"Single problem timeout ({self.single_problem_timeout}s) is less than "
                f"sum of stage timeouts ({stage_timeouts_sum}s). "
                f"Recommending at least {recommended_timeout}s"
            )

    def get_stage_timeout(self, stage: str) -> int:
        """获取指定阶段的超时时间"""
        stage_mapping = {
            "modeling": "modeling_timeout",
            "model_verification": "model_verification_timeout",
            "coding": "coding_timeout",
            "code_verification": "code_verification_timeout",
            "result_verification": "result_verification_timeout",
            "reporting": "reporting_timeout",
            "code_execution": "code_execution_timeout",
            "llm_request": "llm_request_timeout",
        }

        field_name = stage_mapping.get(stage)
        if not field_name:
            logger.warning(f"Unknown stage: {stage}, using default timeout")
            return 60

        timeout = getattr(self, field_name)

        # 批量模式下使用更短的超时
        if self.batch_mode:
            if stage in ["modeling", "model_verification"]:
                return min(timeout, self.batch_modeling_timeout)
            elif stage in ["code_verification", "result_verification"]:
                return min(
                    timeout,
                    (
                        self.batch_code_verification_timeout
                        if stage == "code_verification"
                        else self.batch_result_verification_timeout
                    ),
                )
            elif stage == "code_execution":
                return min(timeout, self.batch_execution_timeout)

        return timeout

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_workflow_timeout": self.total_workflow_timeout,
            "single_problem_timeout": self.single_problem_timeout,
            "modeling_timeout": self.modeling_timeout,
            "model_verification_timeout": self.model_verification_timeout,
            "coding_timeout": self.coding_timeout,
            "code_verification_timeout": self.code_verification_timeout,
            "result_verification_timeout": self.result_verification_timeout,
            "reporting_timeout": self.reporting_timeout,
            "code_execution_timeout": self.code_execution_timeout,
            "llm_request_timeout": self.llm_request_timeout,
            "agent_recursion_limit": self.agent_recursion_limit,
            "langgraph_recursion_limit": self.langgraph_recursion_limit,
            "batch_mode": self.batch_mode,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TimeoutConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    @classmethod
    def create_batch_config(
        cls, base_config: Optional["TimeoutConfig"] = None
    ) -> "TimeoutConfig":
        """创建批量评估专用配置"""
        if base_config is None:
            base_config = cls()

        # 创建批量模式配置
        batch_config = cls(
            total_workflow_timeout=base_config.total_workflow_timeout,
            single_problem_timeout=min(
                base_config.single_problem_timeout, 300
            ),  # 最多5分钟
            # 压缩各阶段超时
            modeling_timeout=min(base_config.modeling_timeout, 60),  # 1分钟
            model_verification_timeout=min(
                base_config.model_verification_timeout, 90
            ),  # 1.5分钟
            coding_timeout=min(base_config.coding_timeout, 60),  # 1分钟
            syntax_validation_timeout=min(
                base_config.syntax_validation_timeout, 20
            ),  # 20秒
            fidelity_verification_timeout=min(
                base_config.fidelity_verification_timeout, 90
            ),  # 1.5分钟
            execution_validation_timeout=min(
                base_config.execution_validation_timeout, 60
            ),  # 1分钟
            reporting_timeout=min(base_config.reporting_timeout, 30),  # 30秒
            # 压缩底层超时
            code_execution_timeout=min(base_config.code_execution_timeout, 30),  # 30秒
            llm_request_timeout=min(base_config.llm_request_timeout, 60),  # 1分钟
            # 启用批量模式
            batch_mode=True,
            batch_modeling_timeout=60,
            batch_verification_timeout=90,
            batch_execution_timeout=30,
            # 减少重试
            max_retries=min(base_config.max_retries, 2),
        )

        return batch_config

    def print_config(self):
        """打印当前配置"""
        print("OptAgent超时配置:")
        print("=" * 50)
        print(f"总工作流超时: {self.total_workflow_timeout}秒")
        print(f"单题超时: {self.single_problem_timeout}秒")
        print(f"批量模式: {'启用' if self.batch_mode else '禁用'}")
        print()
        print("各阶段超时:")
        print(f"  建模: {self.modeling_timeout}秒")
        print(f"  模型验证: {self.model_verification_timeout}秒")
        print(f"  编码: {self.coding_timeout}秒")
        print(f"  语法验证: {self.syntax_validation_timeout}秒")
        print(f"  忠实度验证: {self.fidelity_verification_timeout}秒")
        print(f"  执行验证: {self.execution_validation_timeout}秒")
        print(f"  报告生成: {self.reporting_timeout}秒")
        print()
        print("底层超时:")
        print(f"  代码执行: {self.code_execution_timeout}秒")
        print(f"  LLM请求: {self.llm_request_timeout}秒")
        print(f"  Agent递归限制: {self.agent_recursion_limit}")
        print(f"  LangGraph递归限制: {self.langgraph_recursion_limit}")


# 全局默认配置实例
_default_timeout_config = None


def get_timeout_config() -> TimeoutConfig:
    """获取全局超时配置"""
    global _default_timeout_config
    if _default_timeout_config is None:
        _default_timeout_config = TimeoutConfig()
    return _default_timeout_config


def set_timeout_config(config: TimeoutConfig):
    """设置全局超时配置"""
    global _default_timeout_config
    _default_timeout_config = config


def get_stage_timeout(stage: str) -> int:
    """快捷函数：获取指定阶段的超时时间"""
    return get_timeout_config().get_stage_timeout(stage)


def is_batch_mode() -> bool:
    """快捷函数：检查是否为批量模式"""
    return get_timeout_config().batch_mode
