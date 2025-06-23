"""
TRADINO UNSCHLAGBAR - Performance Optimizer
Hochleistungs-Performance-Monitor für Real-time Trading
"""

import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio
import logging
from contextlib import contextmanager
import functools

@dataclass
class PerformanceMetrics:
    """Performance Metriken Container"""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    timestamp: float = field(default_factory=time.time)

class PerformanceOptimizer:
    """
    Hochleistungs Performance Optimizer für RL Trading System
    Ziel: Sub-5ms Action Generation, -30% Memory, +25% CPU Efficiency
    """
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_targets = {
            'action_generation': 0.005,  # 5ms Ziel
            'signal_generation': 0.003,  # 3ms Ziel
            'memory_reduction': 0.30,    # 30% Reduktion
            'cpu_improvement': 0.25      # 25% Verbesserung
        }
        
        self.baseline_metrics = {}
        self.optimization_callbacks = {}
        self.monitoring_active = True
        self._lock = threading.Lock()
        
        # Performance Tracking
        self.execution_times = defaultdict(list)
        self.resource_usage = defaultdict(list)
        
        logging.info("🚀 PerformanceOptimizer initialisiert - Ziel: Sub-5ms Actions")
    
    @contextmanager
    def measure_performance(self, operation_name: str):
        """Context Manager für Performance Measurement"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_time = (end_time - start_time) * 1000  # ms
            memory_delta = end_memory - start_memory
            
            self._record_performance(operation_name, execution_time, memory_delta)
    
    def _record_performance(self, operation: str, exec_time: float, memory_delta: float):
        """Performance Daten aufzeichnen"""
        with self._lock:
            self.execution_times[operation].append(exec_time)
            self.resource_usage[operation].append(memory_delta)
            
            # Nur letzte 1000 Measurements behalten
            if len(self.execution_times[operation]) > 1000:
                self.execution_times[operation] = self.execution_times[operation][-1000:]
                self.resource_usage[operation] = self.resource_usage[operation][-1000:]
    
    def performance_decorator(self, operation_name: str):
        """Decorator für automatische Performance Messung"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure_performance(operation_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, PerformanceMetrics]:
        """Aktuelle Performance Summary"""
        summary = {}
        
        with self._lock:
            for operation, times in self.execution_times.items():
                if not times:
                    continue
                
                metrics = PerformanceMetrics(
                    execution_time=np.mean(times),
                    memory_usage=np.mean(self.resource_usage.get(operation, [0])),
                    latency_p95=np.percentile(times, 95),
                    latency_p99=np.percentile(times, 99),
                    throughput=1000.0 / np.mean(times) if np.mean(times) > 0 else 0
                )
                summary[operation] = metrics
        
        return summary
    
    def check_performance_targets(self) -> Dict[str, bool]:
        """Überprüfe ob Performance-Ziele erreicht wurden"""
        summary = self.get_performance_summary()
        results = {}
        
        # Action Generation Target
        if 'action_generation' in summary:
            target_met = summary['action_generation'].execution_time <= self.performance_targets['action_generation']
            results['action_generation_target'] = target_met
        
        # Signal Generation Target
        if 'signal_generation' in summary:
            target_met = summary['signal_generation'].execution_time <= self.performance_targets['signal_generation']
            results['signal_generation_target'] = target_met
        
        return results
    
    def optimize_automatically(self):
        """Automatische Performance Optimierung"""
        summary = self.get_performance_summary()
        
        for operation, metrics in summary.items():
            # Wenn Latenz zu hoch, Optimierungen vorschlagen
            if metrics.execution_time > self.performance_targets.get(f"{operation}_time", 0.01):
                self._suggest_optimizations(operation, metrics)
    
    def _suggest_optimizations(self, operation: str, metrics: PerformanceMetrics):
        """Optimierungsvorschläge basierend auf Metriken"""
        suggestions = []
        
        if metrics.execution_time > 0.005:  # 5ms
            suggestions.append("Enable caching for frequent operations")
            suggestions.append("Consider parallel processing")
        
        if metrics.memory_usage > 100:  # 100MB
            suggestions.append("Implement memory pooling")
            suggestions.append("Optimize data structures")
        
        logging.info(f"🔧 Optimierungsvorschläge für {operation}: {suggestions}")
        return suggestions
    
    def set_baseline(self, operation: str):
        """Baseline Performance für Vergleiche setzen"""
        summary = self.get_performance_summary()
        if operation in summary:
            self.baseline_metrics[operation] = summary[operation]
            logging.info(f"📊 Baseline für {operation} gesetzt: {summary[operation].execution_time:.3f}ms")
    
    def get_improvement_percentage(self, operation: str) -> Optional[float]:
        """Verbesserung in Prozent seit Baseline"""
        if operation not in self.baseline_metrics:
            return None
        
        current = self.get_performance_summary().get(operation)
        if not current:
            return None
        
        baseline_time = self.baseline_metrics[operation].execution_time
        current_time = current.execution_time
        
        improvement = ((baseline_time - current_time) / baseline_time) * 100
        return improvement
    
    async def continuous_monitoring(self, interval: float = 1.0):
        """Kontinuierliches Performance Monitoring"""
        while self.monitoring_active:
            try:
                # System Resource Monitoring
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                # GPU Monitoring (falls verfügbar)
                gpu_usage = self._get_gpu_usage()
                
                # Log kritische Performance Issues
                if cpu_percent > 80:
                    logging.warning(f"⚠️ Hohe CPU Usage: {cpu_percent:.1f}%")
                
                if memory_info.percent > 85:
                    logging.warning(f"⚠️ Hoher Memory Usage: {memory_info.percent:.1f}%")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logging.error(f"❌ Monitoring Error: {e}")
                await asyncio.sleep(interval)
    
    def _get_gpu_usage(self) -> float:
        """GPU Usage ermitteln (falls verfügbar)"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return 0.0
    
    def generate_performance_report(self) -> str:
        """Detaillierter Performance Report"""
        summary = self.get_performance_summary()
        targets = self.check_performance_targets()
        
        report = []
        report.append("🚀 TRADINO UNSCHLAGBAR - Performance Report")
        report.append("=" * 50)
        
        for operation, metrics in summary.items():
            report.append(f"\n📊 {operation.upper()}")
            report.append(f"  Execution Time: {metrics.execution_time:.4f}ms")
            report.append(f"  Memory Usage: {metrics.memory_usage:.2f}MB")
            report.append(f"  Throughput: {metrics.throughput:.1f} ops/sec")
            report.append(f"  P95 Latency: {metrics.latency_p95:.4f}ms")
            report.append(f"  P99 Latency: {metrics.latency_p99:.4f}ms")
            
            # Improvement seit Baseline
            improvement = self.get_improvement_percentage(operation)
            if improvement is not None:
                status = "✅" if improvement > 0 else "❌"
                report.append(f"  Improvement: {status} {improvement:+.1f}%")
        
        report.append(f"\n🎯 TARGET STATUS")
        for target, met in targets.items():
            status = "✅ ERREICHT" if met else "❌ VERFEHLT"
            report.append(f"  {target}: {status}")
        
        return "\n".join(report)

# Global Performance Optimizer Instance
performance_optimizer = PerformanceOptimizer()

# Convenience Decorators
def measure_action_generation(func):
    """Decorator für Action Generation Performance"""
    return performance_optimizer.performance_decorator('action_generation')(func)

def measure_signal_generation(func):
    """Decorator für Signal Generation Performance"""
    return performance_optimizer.performance_decorator('signal_generation')(func)

def measure_model_prediction(func):
    """Decorator für Model Prediction Performance"""
    return performance_optimizer.performance_decorator('model_prediction')(func) 