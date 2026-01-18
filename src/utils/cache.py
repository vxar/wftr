"""
Caching utilities for API responses and computed data
Provides intelligent caching to reduce API calls and improve performance
"""
import time
import hashlib
import json
from typing import Any, Optional, Dict, Callable
from functools import wraps
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TTLCache:
    """Time-to-live cache with automatic cleanup"""
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """
        Initialize TTL cache
        
        Args:
            ttl_seconds: Time to live for cache entries in seconds
            max_size: Maximum number of entries in cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        current_time = time.time()
        
        # Check if expired
        if current_time - entry['timestamp'] > self.ttl_seconds:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return None
        
        # Update access time for LRU
        self._access_times[key] = current_time
        return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp"""
        current_time = time.time()
        
        # Remove oldest entry if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[key] = {
            'value': value,
            'timestamp': current_time
        }
        self._access_times[key] = current_time
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


class FileCache:
    """Persistent file-based cache for larger data"""
    
    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 3600):
        """
        Initialize file cache
        
        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time to live for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        # Use hash to avoid filename issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                entry = json.load(f)
            
            # Check if expired
            current_time = time.time()
            if current_time - entry['timestamp'] > self.ttl_seconds:
                cache_path.unlink()  # Remove expired file
                return None
            
            return entry['value']
        
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except:
                pass
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in file cache"""
        cache_path = self._get_cache_path(key)
        
        try:
            entry = {
                'value': value,
                'timestamp': time.time()
            }
            
            with open(cache_path, 'w') as f:
                json.dump(entry, f)
        
        except (IOError, TypeError) as e:
            logger.warning(f"Error writing cache file {cache_path}: {e}")
    
    def clear(self) -> None:
        """Clear all cache files"""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except:
                pass


def cached(ttl_seconds: int = 300, max_size: int = 1000, use_file_cache: bool = False):
    """
    Decorator for caching function results
    
    Args:
        ttl_seconds: Time to live for cache entries
        max_size: Maximum cache size (for memory cache)
        use_file_cache: Use file cache instead of memory cache
    """
    def decorator(func: Callable) -> Callable:
        # Initialize appropriate cache
        if use_file_cache:
            cache = FileCache(ttl_seconds=ttl_seconds)
        else:
            cache = TTLCache(ttl_seconds=ttl_seconds, max_size=max_size)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            try:
                # Try to create a hashable key from args and kwargs
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            except Exception:
                # If we can't create a proper key, just call the function
                return func(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        # Add cache management methods to wrapper
        wrapper.cache_clear = cache.clear
        wrapper.cache_size = lambda: cache.size
        
        return wrapper
    
    return decorator


# Global cache instances
memory_cache = TTLCache(ttl_seconds=300, max_size=1000)
file_cache = FileCache(ttl_seconds=3600)


def get_cache_key(ticker: str, timeframe: str, count: int, **kwargs) -> str:
    """
    Generate consistent cache key for market data
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Data timeframe (e.g., '1m', '5m')
        count: Number of data points
        **kwargs: Additional parameters
    
    Returns:
        Cache key string
    """
    key_parts = [ticker, timeframe, str(count)]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()
