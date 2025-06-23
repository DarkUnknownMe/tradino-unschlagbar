"""
TRADINO UNSCHLAGBAR - RL Cache Manager
Hochperformantes Caching System für Market States und Actions
"""

import time
import threading
import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import logging
import weakref
from functools import wraps
import asyncio

@dataclass
class CacheEntry:
    """Cache Entry mit Metadaten"""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    ttl: Optional[float] = None
    size_bytes: int = 0

class LRUCache:
    """Thread-safe LRU Cache Implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # TTL Check
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.access_count += 1
                entry.last_access = time.time()
                self.hits += 1
                return entry.value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        with self.lock:
            # Calculate size
            size_bytes = len(pickle.dumps(value)) if value is not None else 0
            
            if key in self.cache:
                # Update existing entry
                self.cache[key].value = value
                self.cache[key].timestamp = time.time()
                self.cache[key].size_bytes = size_bytes
                self.cache.move_to_end(key)
            else:
                # Add new entry
                entry = CacheEntry(
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl,
                    size_bytes=size_bytes
                )
                self.cache[key] = entry
                
                # Evict if necessary
                while len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.get_hit_rate(),
                'total_size_mb': total_size / (1024 * 1024)
            }

class RLCacheManager:
    """
    Hochleistungs Cache Manager für RL Trading System
    Ziel: >80% Cache Hit Rate, -30% Memory Usage
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'market_state_cache_size': 5000,
            'action_cache_size': 10000,
            'prediction_cache_size': 3000,
            'feature_cache_size': 2000,
            'default_ttl': 300,  # 5 minutes
            'cleanup_interval': 60  # 1 minute
        }
        
        # Verschiedene Cache-Typen für optimale Performance
        self.market_state_cache = LRUCache(self.config['market_state_cache_size'])
        self.action_cache = LRUCache(self.config['action_cache_size'])
        self.prediction_cache = LRUCache(self.config['prediction_cache_size'])
        self.feature_cache = LRUCache(self.config['feature_cache_size'])
        
        # Memory Pool für häufig verwendete Objekte
        self.memory_pools = {
            'arrays': [],
            'states': [],
            'actions': []
        }
        
        self.cache_stats = {
            'total_requests': 0,
            'total_hits': 0,
            'cache_saves_ms': 0
        }
        
        # Background Cleanup Task
        self.cleanup_active = True
        self.cleanup_task = None
        
        logging.info("🧠 RLCacheManager initialisiert - Ziel: >80% Hit Rate")
        
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generiere eindeutigen Cache Key"""
        # Erstelle hashable representation
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _array_cache_key(self, array: np.ndarray) -> str:
        """Spezielle Cache Key Generierung für NumPy Arrays"""
        if array is None:
            return "none"
        return f"array_{array.shape}_{hash(array.tobytes())}"
    
    def cache_market_state(self, state_data: Dict, ttl: Optional[float] = None) -> str:
        """Cache Market State mit automatischem Key"""
        key = self._generate_cache_key(state_data)
        self.market_state_cache.put(key, state_data, ttl or self.config['default_ttl'])
        return key
    
    def get_market_state(self, key: str) -> Optional[Dict]:
        """Hole Market State aus Cache"""
        return self.market_state_cache.get(key)
    
    def cache_action(self, state_hash: str, action: Any, confidence: float = 1.0) -> str:
        """Cache RL Action mit State-Action Mapping"""
        key = f"action_{state_hash}_{confidence:.3f}"
        action_data = {
            'action': action,
            'confidence': confidence,
            'state_hash': state_hash
        }
        self.action_cache.put(key, action_data, ttl=30)  # Kurze TTL für Actions
        return key
    
    def get_cached_action(self, state_hash: str, min_confidence: float = 0.8) -> Optional[Any]:
        """Hole passende Action aus Cache"""
        # Suche nach ähnlichen State-Action Paaren
        for cache_key in list(self.action_cache.cache.keys()):
            if f"action_{state_hash}" in cache_key:
                action_data = self.action_cache.get(cache_key)
                if action_data and action_data['confidence'] >= min_confidence:
                    return action_data['action']
        return None
    
    def cache_prediction(self, model_name: str, features: np.ndarray, prediction: Any) -> str:
        """Cache Model Prediction"""
        feature_key = self._array_cache_key(features)
        key = f"pred_{model_name}_{feature_key}"
        
        pred_data = {
            'prediction': prediction,
            'model': model_name,
            'features_shape': features.shape,
            'timestamp': time.time()
        }
        
        self.prediction_cache.put(key, pred_data, ttl=60)  # 1 minute TTL
        return key
    
    def get_cached_prediction(self, model_name: str, features: np.ndarray) -> Optional[Any]:
        """Hole Prediction aus Cache"""
        feature_key = self._array_cache_key(features)
        key = f"pred_{model_name}_{feature_key}"
        
        pred_data = self.prediction_cache.get(key)
        return pred_data['prediction'] if pred_data else None
    
    def cached_feature_engineering(self, func):
        """Decorator für Feature Engineering Caching"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"features_{func.__name__}_{self._generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = self.feature_cache.get(cache_key)
            if cached_result is not None:
                self.cache_stats['total_hits'] += 1
                return cached_result
            
            # Compute and cache result
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            compute_time = (time.perf_counter() - start_time) * 1000
            
            self.cache_stats['cache_saves_ms'] += compute_time
            self.feature_cache.put(cache_key, result, ttl=self.config['default_ttl'])
            
            self.cache_stats['total_requests'] += 1
            return result
        
        return wrapper
    
    def get_memory_pool_object(self, pool_name: str, factory_func=None):
        """Hole Objekt aus Memory Pool oder erstelle neues"""
        if pool_name in self.memory_pools and self.memory_pools[pool_name]:
            return self.memory_pools[pool_name].pop()
        elif factory_func:
            return factory_func()
        else:
            return None
    
    def return_to_memory_pool(self, pool_name: str, obj: Any):
        """Gib Objekt an Memory Pool zurück"""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = []
        
        # Limit pool size
        if len(self.memory_pools[pool_name]) < 100:
            # Reset/clean object if possible
            if hasattr(obj, 'reset'):
                obj.reset()
            self.memory_pools[pool_name].append(obj)
    
    def get_overall_hit_rate(self) -> float:
        """Gesamte Cache Hit Rate"""
        total_hits = (self.market_state_cache.hits + 
                     self.action_cache.hits + 
                     self.prediction_cache.hits + 
                     self.feature_cache.hits)
        
        total_requests = (self.market_state_cache.hits + self.market_state_cache.misses +
                         self.action_cache.hits + self.action_cache.misses +
                         self.prediction_cache.hits + self.prediction_cache.misses +
                         self.feature_cache.hits + self.feature_cache.misses)
        
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Detaillierte Cache Statistiken"""
        return {
            'overall_hit_rate': self.get_overall_hit_rate(),
            'market_state_cache': self.market_state_cache.get_stats(),
            'action_cache': self.action_cache.get_stats(),
            'prediction_cache': self.prediction_cache.get_stats(),
            'feature_cache': self.feature_cache.get_stats(),
            'memory_pools': {name: len(pool) for name, pool in self.memory_pools.items()},
            'performance_savings_ms': self.cache_stats['cache_saves_ms'],
            'total_cache_requests': self.cache_stats['total_requests']
        }
    
    async def cleanup_expired_entries(self):
        """Background Task für Cache Cleanup"""
        while self.cleanup_active:
            try:
                current_time = time.time()
                
                # Cleanup in allen Caches
                for cache in [self.market_state_cache, self.action_cache, 
                             self.prediction_cache, self.feature_cache]:
                    with cache.lock:
                        expired_keys = []
                        for key, entry in cache.cache.items():
                            if entry.ttl and current_time - entry.timestamp > entry.ttl:
                                expired_keys.append(key)
                        
                        for key in expired_keys:
                            del cache.cache[key]
                
                await asyncio.sleep(self.config['cleanup_interval'])
                
            except Exception as e:
                logging.error(f"❌ Cache Cleanup Error: {e}")
                await asyncio.sleep(self.config['cleanup_interval'])
    
    def start_background_cleanup(self):
        """Starte Background Cleanup Task"""
        if not self.cleanup_task:
            loop = asyncio.get_event_loop()
            self.cleanup_task = loop.create_task(self.cleanup_expired_entries())
    
    def optimize_cache_sizes(self):
        """Automatische Cache-Größen Optimierung"""
        stats = self.get_cache_statistics()
        
        # Vergrößere Caches mit hoher Hit Rate
        for cache_name, cache_stats in stats.items():
            if isinstance(cache_stats, dict) and 'hit_rate' in cache_stats:
                hit_rate = cache_stats['hit_rate']
                current_size = cache_stats['max_size']
                
                if hit_rate > 0.9 and current_size < 10000:
                    # Erhöhe Cache Size um 20%
                    new_size = int(current_size * 1.2)
                    logging.info(f"🔧 Erhöhe {cache_name} Cache Size: {current_size} -> {new_size}")
                    
                elif hit_rate < 0.5 and current_size > 500:
                    # Reduziere Cache Size um 20%
                    new_size = int(current_size * 0.8)
                    logging.info(f"🔧 Reduziere {cache_name} Cache Size: {current_size} -> {new_size}")
    
    def clear_all_caches(self):
        """Lösche alle Caches"""
        self.market_state_cache.clear()
        self.action_cache.clear()
        self.prediction_cache.clear()
        self.feature_cache.clear()
        
        for pool in self.memory_pools.values():
            pool.clear()
        
        logging.info("🧹 Alle Caches geleert")

# Global Cache Manager Instance
rl_cache_manager = RLCacheManager()

# Convenience Decorators
def cache_features(func):
    """Decorator für Feature Caching"""
    return rl_cache_manager.cached_feature_engineering(func)

def cache_model_prediction(model_name: str):
    """Decorator für Model Prediction Caching"""
    def decorator(func):
        @wraps(func)
        def wrapper(features, *args, **kwargs):
            # Try cache first
            cached_pred = rl_cache_manager.get_cached_prediction(model_name, features)
            if cached_pred is not None:
                return cached_pred
            
            # Compute and cache
            result = func(features, *args, **kwargs)
            rl_cache_manager.cache_prediction(model_name, features, result)
            return result
        return wrapper
    return decorator 