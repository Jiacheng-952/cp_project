# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Performance Monitor Module for OptAgent

This module provides performance monitoring and metrics collection for the OptAgent system,
allowing tracking of optimization effectiveness and system performance.
"""

import time
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics to track."""
    NODE_EXECUTION_TIME = "node_execution_time"
    LLM_CALL_TIME = "llm_call_time"
    CODE_EXECUTION_TIME = "code_execution_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    VERIFICATION_TIME = "verification_time"
    TOTAL_PROCESSING_TIME = "total_processing_time"


@dataclass
class PerformanceRecord:
    """Individual performance record."""
    timestamp: str
    node_name: str
    metric: PerformanceMetric
    value: float
    unit: str
    metadata: Dict[str, Any]


@dataclass
class NodePerformanceSummary:
    """Performance summary for a specific node."""
    node_name: str
    total_executions: int
    average_execution_time: float
    min_execution_time: float
    max_execution_time: float
    cache_hit_rate: float
    optimization_effectiveness: float


class PerformanceMonitor:
    """Performance monitoring and metrics collection system."""
    
    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self.records: List[PerformanceRecord] = []
        self.node_timers: Dict[str, Dict[str, float]] = {}
        self.cache_stats: Dict[str, Dict[str, int]] = {
            "code_execution": {"hits": 0, "misses": 0}
        }
    
    def start_timer(self, node_name: str, timer_type: str):
        """Start a timer for a specific node and timer type."""
        if node_name not in self.node_timers:
            self.node_timers[node_name] = {}
        
        self.node_timers[node_name][timer_type] = time.time()
    
    def stop_timer(self, node_name: str, timer_type: str) -> float:
        """Stop a timer and return the elapsed time."""
        if (node_name not in self.node_timers or 
            timer_type not in self.node_timers[node_name]):
            return 0.0
        
        start_time = self.node_timers[node_name][timer_type]
        elapsed_time = time.time() - start_time
        
        # Record the metric
        self.record_metric(
            node_name=node_name,
            metric=PerformanceMetric.NODE_EXECUTION_TIME,
            value=elapsed_time,
            unit="seconds",
            metadata={"timer_type": timer_type}
        )
        
        # Clean up timer
        del self.node_timers[node_name][timer_type]
        
        return elapsed_time
    
    def record_metric(self, node_name: str, metric: PerformanceMetric, 
                     value: float, unit: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        record = PerformanceRecord(
            timestamp=datetime.now().isoformat(),
            node_name=node_name,
            metric=metric,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        self.records.append(record)
        
        # Maintain record limit
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
        
        logger.debug(f"Recorded metric: {node_name}.{metric.value} = {value} {unit}")
    
    def record_cache_hit(self, cache_type: str):
        """Record a cache hit."""
        if cache_type not in self.cache_stats:
            self.cache_stats[cache_type] = {"hits": 0, "misses": 0}
        
        self.cache_stats[cache_type]["hits"] += 1
    
    def record_cache_miss(self, cache_type: str):
        """Record a cache miss."""
        if cache_type not in self.cache_stats:
            self.cache_stats[cache_type] = {"hits": 0, "misses": 0}
        
        self.cache_stats[cache_type]["misses"] += 1
    
    def get_cache_hit_rate(self, cache_type: str) -> float:
        """Calculate cache hit rate for a specific cache type."""
        if cache_type not in self.cache_stats:
            return 0.0
        
        stats = self.cache_stats[cache_type]
        total = stats["hits"] + stats["misses"]
        
        if total == 0:
            return 0.0
        
        return stats["hits"] / total
    
    def get_node_performance_summary(self, node_name: str) -> NodePerformanceSummary:
        """Get performance summary for a specific node."""
        node_records = [r for r in self.records if r.node_name == node_name]
        
        if not node_records:
            return NodePerformanceSummary(
                node_name=node_name,
                total_executions=0,
                average_execution_time=0.0,
                min_execution_time=0.0,
                max_execution_time=0.0,
                cache_hit_rate=0.0,
                optimization_effectiveness=0.0
            )
        
        execution_times = [
            r.value for r in node_records 
            if r.metric == PerformanceMetric.NODE_EXECUTION_TIME
        ]
        
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
        else:
            avg_time = min_time = max_time = 0.0
        
        cache_hit_rate = self.get_cache_hit_rate("code_execution")
        
        # Calculate optimization effectiveness (placeholder for now)
        optimization_effectiveness = 0.0
        if len(execution_times) > 10:
            # Simple heuristic: if average time is decreasing, effectiveness is positive
            recent_times = execution_times[-10:]
            older_times = execution_times[-20:-10] if len(execution_times) >= 20 else execution_times[:10]
            
            if older_times and recent_times:
                old_avg = sum(older_times) / len(older_times)
                new_avg = sum(recent_times) / len(recent_times)
                optimization_effectiveness = (old_avg - new_avg) / old_avg if old_avg > 0 else 0.0
        
        return NodePerformanceSummary(
            node_name=node_name,
            total_executions=len(node_records),
            average_execution_time=avg_time,
            min_execution_time=min_time,
            max_execution_time=max_time,
            cache_hit_rate=cache_hit_rate,
            optimization_effectiveness=max(0.0, min(1.0, optimization_effectiveness))
        )
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        node_names = set(r.node_name for r in self.records)
        
        node_summaries = {}
        for node_name in node_names:
            node_summaries[node_name] = asdict(self.get_node_performance_summary(node_name))
        
        # Calculate overall system metrics
        total_executions = sum(s["total_executions"] for s in node_summaries.values())
        
        # Cache statistics
        cache_stats = {}
        for cache_type, stats in self.cache_stats.items():
            cache_stats[cache_type] = {
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": self.get_cache_hit_rate(cache_type)
            }
        
        # Performance trends (simplified)
        recent_records = self.records[-100:] if len(self.records) > 100 else self.records
        if recent_records:
            avg_recent_time = sum(r.value for r in recent_records if r.metric == PerformanceMetric.NODE_EXECUTION_TIME) / len(recent_records)
        else:
            avg_recent_time = 0.0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(self.records),
            "total_executions": total_executions,
            "node_summaries": node_summaries,
            "cache_statistics": cache_stats,
            "average_recent_execution_time": avg_recent_time,
            "system_health": self._calculate_system_health(node_summaries)
        }
        
        return report
    
    def _calculate_system_health(self, node_summaries: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall system health status."""
        if not node_summaries:
            return "UNKNOWN"
        
        # Simple health calculation based on execution times and cache hit rates
        avg_times = [s["average_execution_time"] for s in node_summaries.values()]
        hit_rates = [s["cache_hit_rate"] for s in node_summaries.values()]
        
        if not avg_times:
            return "UNKNOWN"
        
        avg_time = sum(avg_times) / len(avg_times)
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        
        if avg_time < 30 and avg_hit_rate > 0.3:
            return "EXCELLENT"
        elif avg_time < 60 and avg_hit_rate > 0.2:
            return "GOOD"
        elif avg_time < 120:
            return "FAIR"
        else:
            return "POOR"
    
    def export_report(self, filepath: str):
        """Export performance report to a JSON file."""
        report = self.get_system_performance_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Performance report exported to: {filepath}")
    
    def clear_records(self):
        """Clear all performance records."""
        self.records.clear()
        self.node_timers.clear()
        self.cache_stats.clear()
        logger.info("Performance records cleared")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def performance_timed(node_name: str, timer_type: str = "execution"):
    """Decorator to measure and record function execution time."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                monitor.start_timer(node_name, timer_type)
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed_time = monitor.stop_timer(node_name, timer_type)
                    logger.info(f"{node_name}.{timer_type} completed in {elapsed_time:.2f}s")
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                monitor.start_timer(node_name, timer_type)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_time = monitor.stop_timer(node_name, timer_type)
                    logger.info(f"{node_name}.{timer_type} completed in {elapsed_time:.2f}s")
            
            return sync_wrapper
    
    return decorator