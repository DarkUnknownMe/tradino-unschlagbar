"""
TRADINO UNSCHLAGBAR - Parallel RL Engine
Multi-Threading und Async Processing für RL Algorithms
"""

import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import time
import queue
import logging
from dataclasses import dataclass, field
import numpy as np
from functools import wraps
import weakref

@dataclass
class ParallelConfig:
    """Parallel Processing Konfiguration"""
    max_workers: int = mp.cpu_count()
    thread_pool_size: int = 8
    process_pool_size: int = 4
    async_batch_size: int = 32
    queue_timeout: float = 1.0
    load_balancing: bool = True
    memory_limit_mb: int = 2048

@dataclass
class TaskResult:
    """Task Execution Result"""
    task_id: str
    result: Any
    execution_time: float
    worker_id: str
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class LoadBalancer:
    """Intelligenter Load Balancer für Worker Distribution"""
    
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.worker_loads = [0] * num_workers
        self.worker_performance = [1.0] * num_workers
        self.task_history = {}
        self.lock = threading.Lock()
    
    def get_optimal_worker(self, task_complexity: float = 1.0) -> int:
        """Ermittle optimalen Worker basierend auf Load und Performance"""
        with self.lock:
            effective_loads = [
                load / performance 
                for load, performance in zip(self.worker_loads, self.worker_performance)
            ]
            optimal_worker = np.argmin(effective_loads)
            self.worker_loads[optimal_worker] += task_complexity
            return optimal_worker
    
    def update_worker_performance(self, worker_id: int, execution_time: float, 
                                 expected_time: float = 1.0):
        """Update Worker Performance basierend auf Execution Time"""
        with self.lock:
            performance_factor = expected_time / max(execution_time, 0.001)
            alpha = 0.1
            self.worker_performance[worker_id] = (
                alpha * performance_factor + 
                (1 - alpha) * self.worker_performance[worker_id]
            )
    
    def release_worker_load(self, worker_id: int, task_complexity: float = 1.0):
        """Reduziere Worker Load nach Task Completion"""
        with self.lock:
            self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - task_complexity)

class AsyncTaskQueue:
    """High-Performance Async Task Queue"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.results = {}
        self.pending_tasks = set()
        self.completed_tasks = {}
        
    async def submit_task(self, task_id: str, coro, priority: int = 0) -> str:
        """Submit Async Task"""
        task_item = {
            'id': task_id,
            'coro': coro,
            'priority': priority,
            'submitted_at': time.time()
        }
        await self.queue.put(task_item)
        self.pending_tasks.add(task_id)
        return task_id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Hole Task Result"""
        start_time = time.time()
        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timeout")
            await asyncio.sleep(0.01)
        
        result = self.completed_tasks[task_id]
        del self.completed_tasks[task_id]
        return result
    
    async def process_tasks(self, worker_func: Callable):
        """Process Tasks in Queue"""
        while True:
            try:
                task_item = await self.queue.get()
                task_id = task_item['id']
                
                start_time = time.perf_counter()
                try:
                    result = await worker_func(task_item['coro'])
                    execution_time = time.perf_counter() - start_time
                    
                    task_result = TaskResult(
                        task_id=task_id,
                        result=result,
                        execution_time=execution_time,
                        worker_id=asyncio.current_task().get_name(),
                        success=True
                    )
                    
                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    task_result = TaskResult(
                        task_id=task_id,
                        result=None,
                        execution_time=execution_time,
                        worker_id=asyncio.current_task().get_name(),
                        success=False,
                        error=str(e)
                    )
                
                self.completed_tasks[task_id] = task_result
                self.pending_tasks.discard(task_id)
                self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"❌ Task Processing Error: {e}")

class ParallelRLEngine:
    """
    Hochleistungs Parallel Processing Engine für RL Algorithms
    Ziel: +25% CPU Efficiency, Multi-Threading, Async I/O
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        
        # Executors
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="TRADINO-Thread"
        )
        self.process_executor = ProcessPoolExecutor(
            max_workers=self.config.process_pool_size
        )
        
        # Load Balancing
        self.load_balancer = LoadBalancer(self.config.max_workers)
        
        # Async Task Management
        self.async_queue = AsyncTaskQueue()
        self.async_workers = []
        
        # Performance Tracking
        self.execution_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0,
            'cpu_utilization': [],
            'memory_usage': []
        }
        
        # Resource Monitoring
        self.monitoring_active = True
        self.resource_monitor_task = None
        
        logging.info(f"🚀 Parallel RL Engine initialisiert - {self.config.max_workers} Workers")
    
    def parallel_rl_prediction(self, models: Dict[str, Any], features_batch: List[np.ndarray]) -> Dict[str, List]:
        """Parallele RL Model Predictions"""
        
        def predict_single_model(model_name_data):
            model_name, model, features = model_name_data
            predictions = []
            
            for feature_set in features:
                try:
                    pred = model.predict(feature_set)
                    predictions.append(pred)
                except Exception as e:
                    logging.error(f"❌ Prediction Error {model_name}: {e}")
                    predictions.append(None)
            
            return model_name, predictions
        
        # Prepare tasks
        tasks = [
            (name, model, features_batch) 
            for name, model in models.items()
        ]
        
        # Execute in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_model = {
                executor.submit(predict_single_model, task): task[0] 
                for task in tasks
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    name, predictions = future.result(timeout=5.0)
                    results[name] = predictions
                except Exception as e:
                    logging.error(f"❌ Parallel Prediction Error {model_name}: {e}")
                    results[model_name] = [None] * len(features_batch)
        
        return results
    
    async def async_market_data_processing(self, data_sources: List[Callable], 
                                         processing_func: Callable) -> List[Any]:
        """Asynchrone Market Data Processing"""
        
        async def process_single_source(source_func):
            try:
                # Fetch data
                if asyncio.iscoroutinefunction(source_func):
                    raw_data = await source_func()
                else:
                    loop = asyncio.get_event_loop()
                    raw_data = await loop.run_in_executor(self.thread_executor, source_func)
                
                # Process data
                if asyncio.iscoroutinefunction(processing_func):
                    processed_data = await processing_func(raw_data)
                else:
                    processed_data = await loop.run_in_executor(
                        self.thread_executor, processing_func, raw_data
                    )
                
                return processed_data
                
            except Exception as e:
                logging.error(f"❌ Async Data Processing Error: {e}")
                return None
        
        # Process all sources concurrently
        tasks = [process_single_source(source) for source in data_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [
            result for result in results 
            if result is not None and not isinstance(result, Exception)
        ]
        
        return successful_results
    
    def parallel_feature_engineering(self, raw_data_list: List[Any], 
                                   feature_functions: List[Callable]) -> List[Dict]:
        """Parallele Feature Engineering"""
        
        def engineer_features_for_data(data_item):
            features = {}
            for func in feature_functions:
                try:
                    func_name = func.__name__
                    features[func_name] = func(data_item)
                except Exception as e:
                    logging.error(f"❌ Feature Engineering Error {func.__name__}: {e}")
                    features[func_name] = None
            return features
        
        # Use ThreadPoolExecutor für CPU-bound feature engineering
        with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
            feature_results = list(executor.map(engineer_features_for_data, raw_data_list))
        
        return feature_results
    
    async def async_rl_training_step(self, agents: Dict[str, Any], experiences: List[Any]) -> Dict[str, Any]:
        """Asynchrone RL Training Steps"""
        
        async def train_single_agent(agent_name_data):
            agent_name, agent, agent_experiences = agent_name_data
            
            try:
                # Training Step
                loop = asyncio.get_event_loop()
                training_result = await loop.run_in_executor(
                    self.thread_executor,
                    lambda: agent.train_step(agent_experiences)
                )
                
                return agent_name, training_result
                
            except Exception as e:
                logging.error(f"❌ Async Training Error {agent_name}: {e}")
                return agent_name, None
        
        # Distribute experiences among agents
        tasks = []
        experience_per_agent = len(experiences) // len(agents)
        
        for i, (agent_name, agent) in enumerate(agents.items()):
            start_idx = i * experience_per_agent
            end_idx = start_idx + experience_per_agent if i < len(agents) - 1 else len(experiences)
            agent_experiences = experiences[start_idx:end_idx]
            
            tasks.append(train_single_agent((agent_name, agent, agent_experiences)))
        
        # Execute training steps concurrently
        results = await asyncio.gather(*tasks)
        
        return {name: result for name, result in results if result is not None}
    
    def batch_process_with_load_balancing(self, tasks: List[Tuple[Callable, Any]], 
                                        task_complexities: Optional[List[float]] = None) -> List[Any]:
        """Batch Processing mit intelligentem Load Balancing"""
        
        if task_complexities is None:
            task_complexities = [1.0] * len(tasks)
        
        def execute_task_with_balancing(task_data):
            func, args, complexity, task_idx = task_data
            
            # Get optimal worker
            worker_id = self.load_balancer.get_optimal_worker(complexity)
            
            start_time = time.perf_counter()
            try:
                result = func(args)
                execution_time = time.perf_counter() - start_time
                
                # Update performance metrics
                self.load_balancer.update_worker_performance(worker_id, execution_time)
                self.load_balancer.release_worker_load(worker_id, complexity)
                
                return task_idx, result, None
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                self.load_balancer.release_worker_load(worker_id, complexity)
                return task_idx, None, str(e)
        
        # Prepare tasks with metadata
        enhanced_tasks = [
            (*task, complexity, idx) 
            for idx, (task, complexity) in enumerate(zip(tasks, task_complexities))
        ]
        
        # Execute with load balancing
        results = [None] * len(tasks)
        errors = {}
        
        with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
            future_results = executor.map(execute_task_with_balancing, enhanced_tasks)
            
            for task_idx, result, error in future_results:
                if error:
                    errors[task_idx] = error
                    logging.error(f"❌ Task {task_idx} failed: {error}")
                else:
                    results[task_idx] = result
        
        if errors:
            logging.warning(f"⚠️ {len(errors)} tasks failed out of {len(tasks)}")
        
        return results
    
    async def start_async_workers(self, num_workers: int = 4):
        """Starte Async Worker Tasks"""
        
        async def worker_func(coro):
            """Worker Function für Async Tasks"""
            return await coro
        
        # Starte Worker Tasks
        for i in range(num_workers):
            worker_task = asyncio.create_task(
                self.async_queue.process_tasks(worker_func),
                name=f"TRADINO-AsyncWorker-{i}"
            )
            self.async_workers.append(worker_task)
        
        logging.info(f"🚀 {num_workers} Async Workers gestartet")
    
    async def submit_async_task(self, task_id: str, coro, priority: int = 0, 
                               timeout: Optional[float] = None) -> TaskResult:
        """Submit und warte auf Async Task"""
        await self.async_queue.submit_task(task_id, coro, priority)
        return await self.async_queue.get_result(task_id, timeout)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Performance Metriken des Parallel Engines"""
        return {
            'config': {
                'max_workers': self.config.max_workers,
                'thread_pool_size': self.config.thread_pool_size,
                'process_pool_size': self.config.process_pool_size
            },
            'execution_stats': self.execution_stats.copy(),
            'load_balancer': {
                'worker_loads': self.load_balancer.worker_loads.copy(),
                'worker_performance': self.load_balancer.worker_performance.copy()
            },
            'async_queue': {
                'pending_tasks': len(self.async_queue.pending_tasks),
                'completed_tasks': len(self.async_queue.completed_tasks)
            },
            'resource_usage': {
                'cpu_count': mp.cpu_count(),
                'active_threads': threading.active_count()
            }
        }
    
    async def monitor_resources(self, interval: float = 2.0):
        """Kontinuierliches Resource Monitoring"""
        try:
            import psutil
        except ImportError:
            logging.warning("⚠️ psutil nicht verfügbar - Resource Monitoring deaktiviert")
            return
        
        while self.monitoring_active:
            try:
                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.execution_stats['cpu_utilization'].append(cpu_percent)
                
                # Memory Usage
                memory_info = psutil.virtual_memory()
                self.execution_stats['memory_usage'].append(memory_info.percent)
                
                # Keep only recent data
                if len(self.execution_stats['cpu_utilization']) > 100:
                    self.execution_stats['cpu_utilization'] = self.execution_stats['cpu_utilization'][-50:]
                    self.execution_stats['memory_usage'] = self.execution_stats['memory_usage'][-50:]
                
                # Warning bei hoher Auslastung
                if cpu_percent > 90:
                    logging.warning(f"⚠️ Hohe CPU Auslastung: {cpu_percent:.1f}%")
                
                if memory_info.percent > 85:
                    logging.warning(f"⚠️ Hohe Memory Auslastung: {memory_info.percent:.1f}%")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logging.error(f"❌ Resource Monitoring Error: {e}")
                await asyncio.sleep(interval)
    
    def optimize_thread_pool(self):
        """Optimiere Thread Pool Größe basierend auf Performance"""
        avg_cpu = np.mean(self.execution_stats['cpu_utilization'][-10:]) if self.execution_stats['cpu_utilization'] else 50
        
        current_size = self.config.thread_pool_size
        optimal_size = current_size
        
        if avg_cpu < 50:  # Unterausgelastet
            optimal_size = min(current_size + 2, self.config.max_workers)
        elif avg_cpu > 85:  # Überausgelastet
            optimal_size = max(current_size - 1, 2)
        
        if optimal_size != current_size:
            logging.info(f"🔧 Thread Pool Optimierung: {current_size} -> {optimal_size}")
            # Restart thread executor with new size
            self.thread_executor.shutdown(wait=False)
            self.thread_executor = ThreadPoolExecutor(
                max_workers=optimal_size,
                thread_name_prefix="TRADINO-Thread"
            )
            self.config.thread_pool_size = optimal_size
    
    def cleanup(self):
        """Cleanup alle Resources"""
        self.monitoring_active = False
        
        # Cancel async workers
        for worker in self.async_workers:
            worker.cancel()
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logging.info("🧹 Parallel RL Engine bereinigt")

# Global Parallel Engine Instance
parallel_rl_engine = ParallelRLEngine()

# Decorator für automatische Parallelisierung
def parallelize_prediction(max_workers: int = 4):
    """Decorator für parallele Model Predictions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extrahiere batch_data falls vorhanden
            if 'batch_data' in kwargs:
                batch_data = kwargs.pop('batch_data')
                
                # Teile Batch in Chunks
                chunk_size = len(batch_data) // max_workers
                chunks = [
                    batch_data[i:i + chunk_size] 
                    for i in range(0, len(batch_data), chunk_size)
                ]
                
                def process_chunk(chunk):
                    return func(chunk, *args, **kwargs)
                
                # Parallele Verarbeitung
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(process_chunk, chunks))
                
                # Combine results
                if results and isinstance(results[0], list):
                    return [item for sublist in results for item in sublist]
                else:
                    return results
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Async Decorators
def async_market_data(func):
    """Decorator für asynchrone Market Data Processing"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await loop.run_in_executor(
                parallel_rl_engine.thread_executor, 
                func, *args, **kwargs
            )
    return wrapper