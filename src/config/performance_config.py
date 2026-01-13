# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Performance Configuration Module for OptAgent

This module provides configuration management for performance optimization settings,
allowing easy tuning and experimentation with different optimization strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CacheConfig:
    """Configuration for caching mechanisms."""
    enable_code_cache: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600  # 1 hour
    enable_adaptive_cache: bool = True
    adaptive_cache_threshold: float = 0.7  # 70% hit rate threshold


@dataclass
class TimeoutConfig:
    """Configuration for timeout management."""
    enable_dynamic_timeout: bool = True
    default_llm_timeout: int = 60
    simple_problem_timeout: int = 30
    complex_problem_timeout: int = 90
    timeout_estimation_window: int = 100  # Number of problems to analyze


@dataclass
class VerificationConfig:
    """Configuration for verification optimization."""
    enable_quick_verification: bool = True
    quick_verification_threshold: float = 0.8  # Confidence threshold
    enable_parallel_verification: bool = True
    max_parallel_tasks: int = 3
    verification_fallback_enabled: bool = True


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    enable_performance_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    report_export_enabled: bool = True
    report_export_path: str = "performance_reports"
    real_time_alerts_enabled: bool = False


@dataclass
class PerformanceOptimizationConfig:
    """Comprehensive performance optimization configuration."""
    
    # Component configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global optimization settings
    optimization_mode: str = "balanced"  # balanced, aggressive, conservative
    enable_all_optimizations: bool = True
    optimization_level: int = 2  # 0: none, 1: basic, 2: advanced, 3: experimental
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "cache": {
                "enable_code_cache": self.cache.enable_code_cache,
                "cache_max_size": self.cache.cache_max_size,
                "cache_ttl_seconds": self.cache.cache_ttl_seconds,
                "enable_adaptive_cache": self.cache.enable_adaptive_cache,
                "adaptive_cache_threshold": self.cache.adaptive_cache_threshold,
            },
            "timeout": {
                "enable_dynamic_timeout": self.timeout.enable_dynamic_timeout,
                "default_llm_timeout": self.timeout.default_llm_timeout,
                "simple_problem_timeout": self.timeout.simple_problem_timeout,
                "complex_problem_timeout": self.timeout.complex_problem_timeout,
                "timeout_estimation_window": self.timeout.timeout_estimation_window,
            },
            "verification": {
                "enable_quick_verification": self.verification.enable_quick_verification,
                "quick_verification_threshold": self.verification.quick_verification_threshold,
                "enable_parallel_verification": self.verification.enable_parallel_verification,
                "max_parallel_tasks": self.verification.max_parallel_tasks,
                "verification_fallback_enabled": self.verification.verification_fallback_enabled,
            },
            "monitoring": {
                "enable_performance_monitoring": self.monitoring.enable_performance_monitoring,
                "metrics_collection_interval": self.monitoring.metrics_collection_interval,
                "report_export_enabled": self.monitoring.report_export_enabled,
                "report_export_path": self.monitoring.report_export_path,
                "real_time_alerts_enabled": self.monitoring.real_time_alerts_enabled,
            },
            "global": {
                "optimization_mode": self.optimization_mode,
                "enable_all_optimizations": self.enable_all_optimizations,
                "optimization_level": self.optimization_level,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PerformanceOptimizationConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if "cache" in config_dict:
            cache_dict = config_dict["cache"]
            config.cache = CacheConfig(**cache_dict)
        
        if "timeout" in config_dict:
            timeout_dict = config_dict["timeout"]
            config.timeout = TimeoutConfig(**timeout_dict)
        
        if "verification" in config_dict:
            verification_dict = config_dict["verification"]
            config.verification = VerificationConfig(**verification_dict)
        
        if "monitoring" in config_dict:
            monitoring_dict = config_dict["monitoring"]
            config.monitoring = MonitoringConfig(**monitoring_dict)
        
        if "global" in config_dict:
            global_dict = config_dict["global"]
            config.optimization_mode = global_dict.get("optimization_mode", "balanced")
            config.enable_all_optimizations = global_dict.get("enable_all_optimizations", True)
            config.optimization_level = global_dict.get("optimization_level", 2)
        
        return config


# Preset configurations for different scenarios

BALANCED_CONFIG = PerformanceOptimizationConfig(
    optimization_mode="balanced",
    optimization_level=2,
    cache=CacheConfig(
        enable_code_cache=True,
        cache_max_size=1000,
        cache_ttl_seconds=3600,
        enable_adaptive_cache=True,
        adaptive_cache_threshold=0.7,
    ),
    timeout=TimeoutConfig(
        enable_dynamic_timeout=True,
        default_llm_timeout=60,
        simple_problem_timeout=30,
        complex_problem_timeout=90,
        timeout_estimation_window=100,
    ),
    verification=VerificationConfig(
        enable_quick_verification=True,
        quick_verification_threshold=0.8,
        enable_parallel_verification=True,
        max_parallel_tasks=3,
        verification_fallback_enabled=True,
    ),
    monitoring=MonitoringConfig(
        enable_performance_monitoring=True,
        metrics_collection_interval=60,
        report_export_enabled=True,
        report_export_path="performance_reports",
        real_time_alerts_enabled=False,
    ),
)

AGGRESSIVE_CONFIG = PerformanceOptimizationConfig(
    optimization_mode="aggressive",
    optimization_level=3,
    cache=CacheConfig(
        enable_code_cache=True,
        cache_max_size=2000,
        cache_ttl_seconds=7200,  # 2 hours
        enable_adaptive_cache=True,
        adaptive_cache_threshold=0.6,
    ),
    timeout=TimeoutConfig(
        enable_dynamic_timeout=True,
        default_llm_timeout=45,
        simple_problem_timeout=20,
        complex_problem_timeout=75,
        timeout_estimation_window=50,
    ),
    verification=VerificationConfig(
        enable_quick_verification=True,
        quick_verification_threshold=0.7,
        enable_parallel_verification=True,
        max_parallel_tasks=5,
        verification_fallback_enabled=False,
    ),
)

CONSERVATIVE_CONFIG = PerformanceOptimizationConfig(
    optimization_mode="conservative",
    optimization_level=1,
    cache=CacheConfig(
        enable_code_cache=True,
        cache_max_size=500,
        cache_ttl_seconds=1800,  # 30 minutes
        enable_adaptive_cache=False,
    ),
    timeout=TimeoutConfig(
        enable_dynamic_timeout=False,
        default_llm_timeout=90,
    ),
    verification=VerificationConfig(
        enable_quick_verification=True,
        quick_verification_threshold=0.9,
        enable_parallel_verification=False,
        verification_fallback_enabled=True,
    ),
)

# Default configuration
DEFAULT_CONFIG = BALANCED_CONFIG


def get_performance_config(mode: str = "balanced") -> PerformanceOptimizationConfig:
    """Get performance configuration based on mode."""
    configs = {
        "balanced": BALANCED_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
    }
    
    return configs.get(mode.lower(), DEFAULT_CONFIG)