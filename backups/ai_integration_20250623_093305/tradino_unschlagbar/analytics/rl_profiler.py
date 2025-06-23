"""
TRADINO UNSCHLAGBAR - RL Performance Profiler
Deep Performance Analysis für RL Trading System
"""

import cProfile
import pstats
import time
import psutil
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import functools
import tracemalloc
import linecache
import os
import sys

@dataclass
class ProfileData:
    """Profile Daten Container"""
    function_name: str
    total_time: float
    cum_time: float
    call_count: int
    per_call_time: float
    memory_usage: float = 0.0
    cpu_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass 
class MemorySnapshot:
    """Memory Usage Snapshot"""
    timestamp: float
    total_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    gc_objects: int
    peak_memory_mb: float

class DetailedProfiler:
    """Detaillierter Performance Profiler"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.memory_profiler_active = False
        self.memory_snapshots = deque(maxlen=1000)
        self.function_profiles = defaultdict(list)
        self.profiling_active = False
        
    def start_profiling(self):
        """Starte detailliertes Profiling"""
        self.profiler.enable()
        self.profiling_active = True
        
        # Memory Tracing
        try:
            tracemalloc.start()
            self.memory_profiler_active = True
        except RuntimeError:
            # Tracemalloc already started
            self.memory_profiler_active = True
        
        logging.info("📊 Detailliertes Profiling gestartet")
    
    def stop_profiling(self):
        """Stoppe Profiling"""
        self.profiler.disable()
        self.profiling_active = False
        
        if self.memory_profiler_active:
            try:
                tracemalloc.stop()
            except RuntimeError:
                pass  # Tracemalloc not started
            self.memory_profiler_active = False
        
        logging.info("📊 Profiling gestoppt")
    
    def get_profile_stats(self, sort_by: str = 'cumulative') -> Dict[str, Any]:
        """Hole detaillierte Profile Statistiken"""
        if not self.profiling_active:
            return {}
        
        stats = pstats.Stats(self.profiler)
        stats.sort_stats(sort_by)
        
        # Extract top functions
        profile_data = []
        for func_info, (cc, nc, tt, ct) in stats.stats.items():
            filename, line_num, func_name = func_info
            
            profile_entry = ProfileData(
                function_name=f"{filename}:{func_name}:{line_num}",
                total_time=tt,
                cum_time=ct,
                call_count=cc,
                per_call_time=tt/cc if cc > 0 else 0
            )
            profile_data.append(profile_entry)
        
        return {
            'total_functions': len(profile_data),
            'top_functions': sorted(profile_data, key=lambda x: x.cum_time, reverse=True)[:20],
            'total_time': sum(p.total_time for p in profile_data),
            'total_calls': sum(p.call_count for p in profile_data)
        }
    
    def profile_memory_usage(self):
        """Profile Memory Usage"""
        if not self.memory_profiler_active:
            return None
        
        # System Memory
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Tracemalloc Memory
        try:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_current = current / 1024 / 1024  # MB
            tracemalloc_peak = peak / 1024 / 1024  # MB
        except RuntimeError:
            tracemalloc_current = 0
            tracemalloc_peak = 0
        
        # GC Objects
        import gc
        gc_objects = len(gc.get_objects())
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_memory_mb=memory_info.total / 1024 / 1024,
            available_memory_mb=memory_info.available / 1024 / 1024,
            memory_percent=memory_info.percent,
            process_memory_mb=process_memory,
            gc_objects=gc_objects,
            peak_memory_mb=tracemalloc_peak
        )
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def get_memory_leaks(self) -> List[Dict]:
        """Erkenne potentielle Memory Leaks"""
        if len(self.memory_snapshots) < 10:
            return []
        
        recent_snapshots = list(self.memory_snapshots)[-10:]
        memory_trend = [s.process_memory_mb for s in recent_snapshots]
        
        # Berechne Trend
        x = np.arange(len(memory_trend))
        slope = np.polyfit(x, memory_trend, 1)[0]
        
        leaks = []
        if slope > 1.0:  # Memory increase > 1MB per snapshot
            leaks.append({
                'type': 'memory_leak_suspected',
                'trend_mb_per_snapshot': slope,
                'current_memory_mb': memory_trend[-1],
                'increase_over_10_snapshots': memory_trend[-1] - memory_trend[0]
            })
        
        # GC Objects Trend
        gc_trend = [s.gc_objects for s in recent_snapshots]
        gc_slope = np.polyfit(x, gc_trend, 1)[0]
        
        if gc_slope > 100:  # More than 100 objects per snapshot
            leaks.append({
                'type': 'object_leak_suspected',
                'trend_objects_per_snapshot': gc_slope,
                'current_objects': gc_trend[-1],
                'increase_over_10_snapshots': gc_trend[-1] - gc_trend[0]
            })
        
        return leaks

class RLPerformanceProfiler:
    """
    Hochleistungs Performance Profiler für RL Trading System
    Deep Analysis, Memory Profiling, Bottleneck Detection
    """
    
    def __init__(self):
        self.detailed_profiler = DetailedProfiler()
        
        # Performance Tracking
        self.function_times = defaultdict(list)
        self.memory_usage_history = deque(maxlen=1000)
        self.cpu_usage_history = deque(maxlen=1000)
        self.gpu_usage_history = deque(maxlen=1000)
        
        # Bottleneck Detection
        self.bottlenecks = defaultdict(list)
        self.performance_alerts = []
        
        # Benchmark Results
        self.benchmark_results = {}
        self.baseline_performance = {}
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_task = None
        
        logging.info("📊 RL Performance Profiler initialisiert")
    
    def profile_function(self, func_name: str = None):
        """Decorator für Function Profiling"""
        def decorator(func):
            nonlocal func_name
            if func_name is None:
                func_name = f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()
                    
                    execution_time = (end_time - start_time) * 1000  # ms
                    memory_delta = end_memory - start_memory
                    
                    # Record performance
                    self.function_times[func_name].append({
                        'execution_time_ms': execution_time,
                        'memory_delta_mb': memory_delta,
                        'timestamp': time.time()
                    })
                    
                    # Bottleneck Detection
                    if execution_time > 10.0:  # > 10ms
                        self.bottlenecks[func_name].append({
                            'execution_time_ms': execution_time,
                            'memory_delta_mb': memory_delta,
                            'timestamp': time.time(),
                            'args_count': len(args),
                            'kwargs_count': len(kwargs)
                        })
                    
                    return result
                    
                except Exception as e:
                    end_time = time.perf_counter()
                    execution_time = (end_time - start_time) * 1000
                    
                    self.performance_alerts.append({
                        'type': 'function_error',
                        'function': func_name,
                        'execution_time_ms': execution_time,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                    
                    raise
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Hole aktuelle Memory Usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Hole GPU Usage falls verfügbar"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        
        # Alternative: nvidia-ml-py
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            pass
        
        return 0.0
    
    async def continuous_system_monitoring(self, interval: float = 1.0):
        """Kontinuierliches System Monitoring"""
        while self.monitoring_active:
            try:
                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage_history.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent
                })
                
                # Memory Usage
                memory_info = psutil.virtual_memory()
                process_memory = self._get_memory_usage()
                self.memory_usage_history.append({
                    'timestamp': time.time(),
                    'system_memory_percent': memory_info.percent,
                    'process_memory_mb': process_memory,
                    'available_memory_gb': memory_info.available / 1024 / 1024 / 1024
                })
                
                # GPU Usage
                gpu_usage = self._get_gpu_usage()
                self.gpu_usage_history.append({
                    'timestamp': time.time(),
                    'gpu_usage_percent': gpu_usage
                })
                
                # Memory Profiling
                memory_snapshot = self.detailed_profiler.profile_memory_usage()
                
                # Performance Alerts
                if cpu_percent > 90:
                    self.performance_alerts.append({
                        'type': 'high_cpu_usage',
                        'value': cpu_percent,
                        'timestamp': time.time()
                    })
                
                if memory_info.percent > 85:
                    self.performance_alerts.append({
                        'type': 'high_memory_usage',
                        'value': memory_info.percent,
                        'timestamp': time.time()
                    })
                
                # Memory Leak Detection
                memory_leaks = self.detailed_profiler.get_memory_leaks()
                for leak in memory_leaks:
                    self.performance_alerts.append({
                        'type': 'memory_leak_detected',
                        'leak_info': leak,
                        'timestamp': time.time()
                    })
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logging.error(f"❌ System Monitoring Error: {e}")
                await asyncio.sleep(interval)
    
    def start_monitoring(self):
        """Starte kontinuierliches Monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.detailed_profiler.start_profiling()
            
            # Starte Monitoring Task
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            self.monitor_task = loop.create_task(self.continuous_system_monitoring())
            
            logging.info("📊 Performance Monitoring gestartet")
    
    def stop_monitoring(self):
        """Stoppe Monitoring"""
        self.monitoring_active = False
        self.detailed_profiler.stop_profiling()
        
        if self.monitor_task:
            self.monitor_task.cancel()
        
        logging.info("📊 Performance Monitoring gestoppt")
    
    def benchmark_rl_components(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark RL System Komponenten"""
        benchmark_results = {}
        
        # Action Generation Benchmark
        if 'action_generation_func' in test_data:
            func = test_data['action_generation_func']
            test_states = test_data.get('test_states', [])
            
            times = []
            for state in test_states[:100]:  # Test 100 states
                start_time = time.perf_counter()
                try:
                    action = func(state)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # ms
                except Exception as e:
                    logging.error(f"❌ Action Generation Benchmark Error: {e}")
            
            if times:
                benchmark_results['action_generation'] = {
                    'avg_time_ms': np.mean(times),
                    'std_time_ms': np.std(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'p95_time_ms': np.percentile(times, 95),
                    'p99_time_ms': np.percentile(times, 99),
                    'throughput_actions_per_sec': 1000.0 / np.mean(times)
                }
        
        # Model Prediction Benchmark
        if 'models' in test_data:
            models = test_data['models']
            test_features = test_data.get('test_features', [])
            
            for model_name, model in models.items():
                times = []
                for features in test_features[:50]:  # Test 50 predictions
                    start_time = time.perf_counter()
                    try:
                        if hasattr(model, 'predict'):
                            prediction = model.predict(features)
                        elif callable(model):
                            prediction = model(features)
                        else:
                            continue
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)
                    except Exception as e:
                        logging.error(f"❌ Model {model_name} Benchmark Error: {e}")
                
                if times:
                    benchmark_results[f'model_{model_name}'] = {
                        'avg_time_ms': np.mean(times),
                        'p95_time_ms': np.percentile(times, 95),
                        'throughput_predictions_per_sec': 1000.0 / np.mean(times)
                    }
        
        # Trading Pipeline Benchmark
        if 'trading_pipeline' in test_data:
            pipeline = test_data['trading_pipeline']
            test_market_data = test_data.get('test_market_data', [])
            
            times = []
            for market_data in test_market_data[:50]:
                start_time = time.perf_counter()
                try:
                    result = pipeline(market_data)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)
                except Exception as e:
                    logging.error(f"❌ Trading Pipeline Benchmark Error: {e}")
            
            if times:
                benchmark_results['trading_pipeline'] = {
                    'avg_time_ms': np.mean(times),
                    'p95_time_ms': np.percentile(times, 95),
                    'throughput_trades_per_sec': 1000.0 / np.mean(times)
                }
        
        self.benchmark_results.update(benchmark_results)
        return benchmark_results
    
    def detect_performance_regressions(self) -> List[Dict]:
        """Erkenne Performance Regressions"""
        regressions = []
        
        for func_name, measurements in self.function_times.items():
            if len(measurements) < 20:  # Nicht genug Daten
                continue
            
            # Vergleiche letzte 10 mit vorherigen 10 Messungen
            recent_times = [m['execution_time_ms'] for m in measurements[-10:]]
            previous_times = [m['execution_time_ms'] for m in measurements[-20:-10]]
            
            if not previous_times:
                continue
            
            recent_avg = np.mean(recent_times)
            previous_avg = np.mean(previous_times)
            
            # Regression wenn 20% langsamer
            if recent_avg > previous_avg * 1.2:
                regression_percent = ((recent_avg - previous_avg) / previous_avg) * 100
                
                regressions.append({
                    'function': func_name,
                    'regression_percent': regression_percent,
                    'previous_avg_ms': previous_avg,
                    'recent_avg_ms': recent_avg,
                    'severity': 'high' if regression_percent > 50 else 'medium'
                })
        
        return regressions
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Detaillierte Bottleneck Analyse"""
        analysis = {
            'top_bottlenecks': [],
            'memory_intensive_functions': [],
            'frequent_slow_functions': []
        }
        
        # Top Bottlenecks by execution time
        all_bottlenecks = []
        for func_name, bottlenecks in self.bottlenecks.items():
            for bottleneck in bottlenecks:
                all_bottlenecks.append({
                    'function': func_name,
                    **bottleneck
                })
        
        # Sort by execution time
        top_bottlenecks = sorted(
            all_bottlenecks, 
            key=lambda x: x['execution_time_ms'], 
            reverse=True
        )[:10]
        
        analysis['top_bottlenecks'] = top_bottlenecks
        
        # Memory intensive functions
        for func_name, measurements in self.function_times.items():
            memory_deltas = [m['memory_delta_mb'] for m in measurements if m['memory_delta_mb'] > 0]
            if memory_deltas:
                avg_memory_usage = np.mean(memory_deltas)
                if avg_memory_usage > 10:  # > 10MB average
                    analysis['memory_intensive_functions'].append({
                        'function': func_name,
                        'avg_memory_usage_mb': avg_memory_usage,
                        'max_memory_usage_mb': max(memory_deltas),
                        'call_count': len(measurements)
                    })
        
        # Frequently slow functions
        for func_name, measurements in self.function_times.items():
            slow_calls = [m for m in measurements if m['execution_time_ms'] > 5.0]
            if len(slow_calls) > 5:  # More than 5 slow calls
                analysis['frequent_slow_functions'].append({
                    'function': func_name,
                    'slow_call_count': len(slow_calls),
                    'total_calls': len(measurements),
                    'slow_call_percentage': (len(slow_calls) / len(measurements)) * 100,
                    'avg_slow_time_ms': np.mean([m['execution_time_ms'] for m in slow_calls])
                })
        
        return analysis
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Hole umfassende Performance Metriken"""
        metrics = {
            'profiler_status': {
                'monitoring_active': self.monitoring_active,
                'detailed_profiling_active': self.detailed_profiler.profiling_active,
                'memory_profiling_active': self.detailed_profiler.memory_profiler_active
            },
            'function_performance': {},
            'system_performance': {},
            'bottlenecks': self.get_bottleneck_analysis(),
            'benchmark_results': self.benchmark_results,
            'performance_alerts_count': len(self.performance_alerts)
        }
        
        # Function Performance Summary
        for func_name, measurements in self.function_times.items():
            if measurements:
                times = [m['execution_time_ms'] for m in measurements]
                memory_deltas = [m['memory_delta_mb'] for m in measurements]
                
                metrics['function_performance'][func_name] = {
                    'call_count': len(measurements),
                    'avg_time_ms': np.mean(times),
                    'p95_time_ms': np.percentile(times, 95),
                    'total_time_ms': np.sum(times),
                    'avg_memory_delta_mb': np.mean(memory_deltas),
                    'max_memory_delta_mb': np.max(memory_deltas)
                }
        
        # System Performance Summary
        if self.cpu_usage_history:
            recent_cpu = [h['cpu_percent'] for h in list(self.cpu_usage_history)[-10:]]
            metrics['system_performance']['avg_cpu_percent'] = np.mean(recent_cpu)
            metrics['system_performance']['max_cpu_percent'] = np.max(recent_cpu)
        
        if self.memory_usage_history:
            recent_memory = [h['process_memory_mb'] for h in list(self.memory_usage_history)[-10:]]
            metrics['system_performance']['avg_memory_mb'] = np.mean(recent_memory)
            metrics['system_performance']['max_memory_mb'] = np.max(recent_memory)
        
        if self.gpu_usage_history:
            recent_gpu = [h['gpu_usage_percent'] for h in list(self.gpu_usage_history)[-10:] if h['gpu_usage_percent'] > 0]
            if recent_gpu:
                metrics['system_performance']['avg_gpu_percent'] = np.mean(recent_gpu)
                metrics['system_performance']['max_gpu_percent'] = np.max(recent_gpu)
        
        return metrics
    
    def generate_performance_report(self) -> str:
        """Generiere umfassenden Performance Report"""
        report = []
        report.append("🚀 TRADINO UNSCHLAGBAR - Performance Analysis Report")
        report.append("=" * 60)
        
        # System Overview
        report.append(f"\n📊 SYSTEM OVERVIEW")
        if self.cpu_usage_history:
            recent_cpu = [h['cpu_percent'] for h in list(self.cpu_usage_history)[-10:]]
            report.append(f"  Average CPU Usage: {np.mean(recent_cpu):.1f}%")
        
        if self.memory_usage_history:
            recent_memory = [h['process_memory_mb'] for h in list(self.memory_usage_history)[-10:]]
            report.append(f"  Process Memory Usage: {np.mean(recent_memory):.1f}MB")
        
        if self.gpu_usage_history:
            recent_gpu = [h['gpu_usage_percent'] for h in list(self.gpu_usage_history)[-10:]]
            avg_gpu = np.mean([g for g in recent_gpu if g > 0])
            if avg_gpu > 0:
                report.append(f"  Average GPU Usage: {avg_gpu:.1f}%")
        
        # Function Performance
        report.append(f"\n⚡ FUNCTION PERFORMANCE")
        for func_name, measurements in list(self.function_times.items())[:10]:
            if measurements:
                times = [m['execution_time_ms'] for m in measurements]
                report.append(f"  {func_name}:")
                report.append(f"    Avg Time: {np.mean(times):.3f}ms")
                report.append(f"    Call Count: {len(measurements)}")
                report.append(f"    P95 Time: {np.percentile(times, 95):.3f}ms")
        
        # Bottlenecks
        bottleneck_analysis = self.get_bottleneck_analysis()
        if bottleneck_analysis['top_bottlenecks']:
            report.append(f"\n🔍 TOP BOTTLENECKS")
            for bottleneck in bottleneck_analysis['top_bottlenecks'][:5]:
                report.append(f"  {bottleneck['function']}: {bottleneck['execution_time_ms']:.2f}ms")
        
        # Performance Alerts
        if self.performance_alerts:
            recent_alerts = [a for a in self.performance_alerts if time.time() - a['timestamp'] < 3600]  # Last hour
            if recent_alerts:
                report.append(f"\n⚠️ RECENT PERFORMANCE ALERTS ({len(recent_alerts)})")
                for alert in recent_alerts[-5:]:
                    report.append(f"  {alert['type']}: {alert.get('value', 'N/A')}")
        
        # Benchmarks
        if self.benchmark_results:
            report.append(f"\n🏁 BENCHMARK RESULTS")
            for benchmark_name, results in self.benchmark_results.items():
                if 'avg_time_ms' in results:
                    report.append(f"  {benchmark_name}: {results['avg_time_ms']:.3f}ms avg")
        
        # Performance Regressions
        regressions = self.detect_performance_regressions()
        if regressions:
            report.append(f"\n📉 PERFORMANCE REGRESSIONS ({len(regressions)})")
            for regression in regressions[:3]:
                report.append(f"  {regression['function']}: +{regression['regression_percent']:.1f}% slower")
        
        # Memory Leak Detection
        memory_leaks = self.detailed_profiler.get_memory_leaks()
        if memory_leaks:
            report.append(f"\n🚨 MEMORY LEAK ALERTS ({len(memory_leaks)})")
            for leak in memory_leaks:
                report.append(f"  {leak['type']}: {leak.get('trend_mb_per_snapshot', 'N/A')} MB/snapshot")
        
        return "\n".join(report)
    
    def export_profile_data(self, filename: str):
        """Exportiere Profile Daten"""
        profile_data = {
            'function_times': dict(self.function_times),
            'bottlenecks': dict(self.bottlenecks),
            'benchmark_results': self.benchmark_results,
            'performance_alerts': self.performance_alerts,
            'system_stats': {
                'cpu_history': list(self.cpu_usage_history),
                'memory_history': list(self.memory_usage_history),
                'gpu_history': list(self.gpu_usage_history)
            },
            'detailed_profiler_stats': self.detailed_profiler.get_profile_stats()
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        logging.info(f"📁 Profile Daten exportiert: {filename}")
    
    def generate_performance_plots(self, output_dir: str = "performance_plots"):
        """Generiere Performance Visualisierungen"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            os.makedirs(output_dir, exist_ok=True)
            
            # CPU Usage Plot
            if self.cpu_usage_history:
                plt.figure(figsize=(12, 6))
                timestamps = [h['timestamp'] for h in self.cpu_usage_history]
                cpu_values = [h['cpu_percent'] for h in self.cpu_usage_history]
                
                plt.plot(timestamps, cpu_values, label='CPU Usage %')
                plt.title('CPU Usage Over Time')
                plt.xlabel('Timestamp')
                plt.ylabel('CPU Usage (%)')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/cpu_usage.png")
                plt.close()
            
            # Memory Usage Plot
            if self.memory_usage_history:
                plt.figure(figsize=(12, 6))
                timestamps = [h['timestamp'] for h in self.memory_usage_history]
                memory_values = [h['process_memory_mb'] for h in self.memory_usage_history]
                
                plt.plot(timestamps, memory_values, label='Process Memory (MB)')
                plt.title('Memory Usage Over Time')
                plt.xlabel('Timestamp')
                plt.ylabel('Memory Usage (MB)')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/memory_usage.png")
                plt.close()
            
            # Function Performance Plot
            if self.function_times:
                plt.figure(figsize=(14, 8))
                
                func_names = []
                avg_times = []
                
                for func_name, measurements in list(self.function_times.items())[:10]:
                    if measurements:
                        times = [m['execution_time_ms'] for m in measurements]
                        func_names.append(func_name.split('.')[-1][:20])  # Short name
                        avg_times.append(np.mean(times))
                
                plt.barh(func_names, avg_times)
                plt.title('Average Function Execution Times')
                plt.xlabel('Execution Time (ms)')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/function_performance.png")
                plt.close()
            
            logging.info(f"📊 Performance Plots generiert in: {output_dir}")
            
        except ImportError:
            logging.warning("⚠️ Matplotlib nicht verfügbar für Plot-Generierung")
    
    def cleanup(self):
        """Cleanup Profiler Resources"""
        self.stop_monitoring()
        
        # Clear data
        self.function_times.clear()
        self.memory_usage_history.clear()
        self.cpu_usage_history.clear()
        self.gpu_usage_history.clear()
        self.bottlenecks.clear()
        self.performance_alerts.clear()
        
        logging.info("🧹 Performance Profiler bereinigt")

# Global Profiler Instance
rl_profiler = RLPerformanceProfiler()

# Convenience Decorators
def profile_rl_function(func_name: str = None):
    """Decorator für RL Function Profiling"""
    return rl_profiler.profile_function(func_name)

def profile_critical_path(func):
    """Decorator für kritische Performance Pfade"""
    return rl_profiler.profile_function(f"CRITICAL_{func.__name__}")(func)