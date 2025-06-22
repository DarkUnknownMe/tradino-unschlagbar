"""
TRADINO UNSCHLAGBAR - GPU Accelerator
CUDA/PyTorch GPU Acceleration für Neural Network Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import time
from dataclasses import dataclass
import threading
from queue import Queue, Empty
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class GPUConfig:
    """GPU Konfiguration"""
    device: str = "cuda"
    batch_size: int = 32
    max_memory_mb: int = 4096
    mixed_precision: bool = True
    async_execution: bool = True
    fallback_cpu: bool = True

class GPUMemoryManager:
    """GPU Memory Management für optimale Performance"""
    
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.allocated_tensors = []
        self.memory_pool = {}
        self.lock = threading.Lock()
        
    def allocate_tensor(self, shape: Tuple, dtype=torch.float32, device="cuda") -> torch.Tensor:
        """Allokiere Tensor mit Memory Pool"""
        key = f"{shape}_{dtype}_{device}"
        
        with self.lock:
            if key in self.memory_pool and self.memory_pool[key]:
                tensor = self.memory_pool[key].pop()
                tensor.zero_()
                return tensor
            else:
                return torch.zeros(shape, dtype=dtype, device=device)
    
    def deallocate_tensor(self, tensor: torch.Tensor):
        """Gib Tensor an Memory Pool zurück"""
        if not tensor.is_cuda:
            return
            
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        device = str(tensor.device)
        key = f"{shape}_{dtype}_{device}"
        
        with self.lock:
            if key not in self.memory_pool:
                self.memory_pool[key] = []
            
            if len(self.memory_pool[key]) < 10:  # Limit pool size
                self.memory_pool[key].append(tensor.detach())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Aktuelle GPU Memory Usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            reserved = torch.cuda.memory_reserved() / 1024 / 1024   # MB
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'utilization': allocated / self.max_memory_mb
            }
        return {'allocated_mb': 0, 'reserved_mb': 0, 'utilization': 0}

class BatchProcessor:
    """Batch Processing für effiziente GPU Nutzung"""
    
    def __init__(self, batch_size: int = 32, timeout_ms: float = 10.0):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.input_queue = Queue()
        self.result_futures = {}
        self.processing = False
        self.lock = threading.Lock()
        
    def add_to_batch(self, input_data: torch.Tensor, request_id: str) -> asyncio.Future:
        """Füge Input zu Batch hinzu"""
        future = asyncio.Future()
        
        with self.lock:
            self.input_queue.put((input_data, request_id))
            self.result_futures[request_id] = future
            
        return future
    
    def process_batch(self, model_func, device="cuda"):
        """Verarbeite Batch auf GPU"""
        batch_inputs = []
        request_ids = []
        
        # Sammle Batch
        start_time = time.time()
        while (len(batch_inputs) < self.batch_size and 
               (time.time() - start_time) * 1000 < self.timeout_ms):
            try:
                input_data, request_id = self.input_queue.get(timeout=0.001)
                batch_inputs.append(input_data)
                request_ids.append(request_id)
            except Empty:
                if batch_inputs:  # Process partial batch if timeout
                    break
                continue
        
        if not batch_inputs:
            return
        
        # Batch Processing auf GPU
        try:
            batch_tensor = torch.stack(batch_inputs).to(device)
            with torch.no_grad():
                batch_results = model_func(batch_tensor)
            
            # Verteile Ergebnisse
            for i, request_id in enumerate(request_ids):
                if request_id in self.result_futures:
                    future = self.result_futures[request_id]
                    if not future.done():
                        future.set_result(batch_results[i])
                    del self.result_futures[request_id]
                    
        except Exception as e:
            # Fehler an alle Futures weitergeben
            for request_id in request_ids:
                if request_id in self.result_futures:
                    future = self.result_futures[request_id]
                    if not future.done():
                        future.set_exception(e)
                    del self.result_futures[request_id]

class RLGPUAccelerator:
    """
    Hochleistungs GPU Accelerator für RL Neural Networks
    Ziel: >80% GPU Utilization, Batch Processing, Async Inference
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        
        # GPU Setup
        self.device = self._setup_gpu()
        self.memory_manager = GPUMemoryManager(self.config.max_memory_mb)
        self.batch_processor = BatchProcessor(self.config.batch_size)
        
        # Model Caches für schnelle Inference
        self.model_cache = {}
        self.compiled_models = {}
        
        # Performance Tracking
        self.inference_times = []
        self.gpu_utilization_history = []
        self.batch_efficiency = []
        
        # Async Processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_queue = asyncio.Queue()
        
        logging.info(f"🚀 GPU Accelerator initialisiert - Device: {self.device}")
        
    def _setup_gpu(self) -> str:
        """GPU Setup und Optimierung"""
        if torch.cuda.is_available() and self.config.device == "cuda":
            device = "cuda"
            torch.backends.cudnn.benchmark = True  # Optimize für konsistente Input-Größen
            torch.backends.cudnn.deterministic = False  # Performance über Determinismus
            
            # GPU Info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"🔥 GPU gefunden: {gpu_name} ({gpu_memory:.1f}GB)")
            
            return device
        else:
            if self.config.fallback_cpu:
                logging.warning("⚠️ GPU nicht verfügbar, nutze CPU Fallback")
                return "cpu"
            else:
                raise RuntimeError("GPU nicht verfügbar und CPU Fallback deaktiviert")
    
    def optimize_model_for_inference(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """Optimiere Model für Inference Performance"""
        model = model.to(self.device)
        model.eval()
        
        # JIT Compilation für bessere Performance
        if self.device == "cuda":
            try:
                # TorchScript Compilation
                traced_model = torch.jit.trace(model, example_input.to(self.device))
                traced_model = torch.jit.optimize_for_inference(traced_model)
                logging.info("✅ Model erfolgreich JIT-kompiliert")
                return traced_model
            except Exception as e:
                logging.warning(f"⚠️ JIT Compilation fehlgeschlagen: {e}")
                return model
        
        return model
    
    def register_model(self, name: str, model: nn.Module, example_input: torch.Tensor):
        """Registriere Model für GPU Acceleration"""
        optimized_model = self.optimize_model_for_inference(model, example_input)
        self.model_cache[name] = optimized_model
        logging.info(f"📝 Model '{name}' für GPU registriert")
    
    async def predict_async(self, model_name: str, input_data: torch.Tensor, 
                          batch_processing: bool = True) -> torch.Tensor:
        """Asynchrone Model Prediction mit Batch Processing"""
        if model_name not in self.model_cache:
            raise ValueError(f"Model '{model_name}' nicht registriert")
        
        model = self.model_cache[model_name]
        
        if batch_processing and input_data.dim() == 1:
            # Single input für Batch Processing
            request_id = f"{model_name}_{time.time()}_{id(input_data)}"
            
            def model_func(batch_input):
                return model(batch_input)
            
            future = self.batch_processor.add_to_batch(input_data, request_id)
            
            # Starte Batch Processing wenn nicht aktiv
            if not self.batch_processor.processing:
                self.batch_processor.processing = True
                loop = asyncio.get_event_loop()
                loop.run_in_executor(
                    self.executor, 
                    self.batch_processor.process_batch, 
                    model_func, 
                    self.device
                )
            
            return await future
        
        else:
            # Direkte Inference
            start_time = time.perf_counter()
            
            input_tensor = input_data.to(self.device)
            
            with torch.no_grad():
                if self.config.mixed_precision and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        result = model(input_tensor)
                else:
                    result = model(input_tensor)
            
            inference_time = (time.perf_counter() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            return result.cpu() if result.is_cuda else result
    
    def predict_batch(self, model_name: str, batch_input: torch.Tensor) -> torch.Tensor:
        """Synchrone Batch Prediction"""
        if model_name not in self.model_cache:
            raise ValueError(f"Model '{model_name}' nicht registriert")
        
        model = self.model_cache[model_name]
        start_time = time.perf_counter()
        
        batch_tensor = batch_input.to(self.device)
        
        with torch.no_grad():
            if self.config.mixed_precision and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    results = model(batch_tensor)
            else:
                results = model(batch_tensor)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        batch_size = batch_input.shape[0]
        per_sample_time = inference_time / batch_size
        
        self.batch_efficiency.append({
            'batch_size': batch_size,
            'total_time_ms': inference_time,
            'per_sample_ms': per_sample_time,
            'throughput': batch_size / (inference_time / 1000)
        })
        
        return results.cpu() if results.is_cuda else results
    
    def preprocess_for_gpu(self, data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """Optimiere Daten für GPU Processing"""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data, list):
            tensor = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            tensor = data.float()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Memory-efficient Transfer
        if self.device == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
        
        return tensor
    
    def create_inference_pipeline(self, model_name: str, preprocessing_func=None, 
                                postprocessing_func=None):
        """Erstelle optimierte Inference Pipeline"""
        
        async def pipeline(input_data):
            # Preprocessing
            if preprocessing_func:
                processed_data = preprocessing_func(input_data)
            else:
                processed_data = self.preprocess_for_gpu(input_data)
            
            # GPU Inference
            result = await self.predict_async(model_name, processed_data)
            
            # Postprocessing
            if postprocessing_func:
                final_result = postprocessing_func(result)
            else:
                final_result = result.numpy() if isinstance(result, torch.Tensor) else result
            
            return final_result
        
        return pipeline
    
    def optimize_memory_usage(self):
        """Optimiere GPU Memory Usage"""
        if self.device == "cuda":
            # Clear Cache
            torch.cuda.empty_cache()
            
            # Memory Defragmentation
            torch.cuda.synchronize()
        
        # Clear old memory pool entries
        self.memory_manager.memory_pool.clear()
        
        logging.info("🧹 GPU Memory optimiert")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """GPU Performance Metriken"""
        memory_info = self.memory_manager.get_memory_usage()
        
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        batch_stats = {}
        if self.batch_efficiency:
            batch_stats = {
                'avg_batch_size': np.mean([b['batch_size'] for b in self.batch_efficiency]),
                'avg_throughput': np.mean([b['throughput'] for b in self.batch_efficiency]),
                'avg_per_sample_ms': np.mean([b['per_sample_ms'] for b in self.batch_efficiency])
            }
        
        return {
            'device': self.device,
            'avg_inference_time_ms': avg_inference_time,
            'memory_usage': memory_info,
            'batch_processing': batch_stats,
            'registered_models': list(self.model_cache.keys()),
            'gpu_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    
    def benchmark_model(self, model_name: str, test_input: torch.Tensor, 
                       iterations: int = 100) -> Dict[str, float]:
        """Benchmark Model Performance"""
        if model_name not in self.model_cache:
            raise ValueError(f"Model '{model_name}' nicht registriert")
        
        model = self.model_cache[model_name]
        test_tensor = test_input.to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_tensor)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                result = model(test_tensor)
                
            if self.device == "cuda":
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # ms
        
        return {
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'p95_time_ms': np.percentile(times, 95),
            'p99_time_ms': np.percentile(times, 99),
            'throughput_ops_sec': 1000.0 / np.mean(times)
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """GPU Health Check"""
        health = {
            'gpu_available': torch.cuda.is_available(),
            'memory_ok': True,
            'models_loaded': len(self.model_cache) > 0,
            'batch_processor_ok': True
        }
        
        if torch.cuda.is_available():
            try:
                memory_info = self.memory_manager.get_memory_usage()
                health['memory_ok'] = memory_info['utilization'] < 0.9
                
                # Test GPU Operation
                test_tensor = torch.randn(10, 10).to(self.device)
                result = test_tensor @ test_tensor.T
                health['gpu_operations_ok'] = result.shape == (10, 10)
                
            except Exception as e:
                logging.error(f"❌ GPU Health Check failed: {e}")
                health['gpu_operations_ok'] = False
        
        return health
    
    def cleanup(self):
        """Cleanup GPU Resources"""
        self.model_cache.clear()
        self.compiled_models.clear()
        self.optimize_memory_usage()
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        logging.info("🧹 GPU Accelerator bereinigt")

# Global GPU Accelerator Instance
gpu_accelerator = RLGPUAccelerator()

# Convenience Functions
async def gpu_predict(model_name: str, input_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convenience function für GPU Prediction"""
    result = await gpu_accelerator.predict_async(model_name, input_data)
    return result.numpy() if isinstance(result, torch.Tensor) else result

def register_rl_model(name: str, model: nn.Module, input_shape: Tuple[int, ...]):
    """Convenience function für Model Registration"""
    example_input = torch.randn(1, *input_shape)
    gpu_accelerator.register_model(name, model, example_input)
    logging.info(f"✅ RL Model '{name}' für GPU registriert")