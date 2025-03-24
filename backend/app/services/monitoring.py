"""
Performance monitoring service for Mini-RAG.

This module provides performance monitoring capabilities,
tracking response times for various components of the system.
"""

import time
from typing import Dict, List, Deque
from collections import deque
import logging
import threading

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread lock for monitoring operations
monitor_lock = threading.Lock()

class PerformanceMonitor:
    """Monitor for tracking system performance metrics.
    
    This class collects and analyzes timing data for various
    components of the Mini-RAG system, helping identify
    performance bottlenecks and optimization opportunities.
    """
    
    def __init__(self, window_size=100):
        """Initialize the performance monitor.
        
        Args:
            window_size: Number of samples to keep for each metric
        """
        self.response_times: Deque[float] = deque(maxlen=window_size)
        self.embedding_times: Deque[float] = deque(maxlen=window_size)
        self.search_times: Deque[float] = deque(maxlen=window_size)
        self.llm_times: Deque[float] = deque(maxlen=window_size)
        self.memory_snapshots: Deque[Dict[str, float]] = deque(maxlen=window_size)
        self.window_size = window_size
        self.total_requests = 0
        self.start_time = time.time()
        logger.info(f"Performance monitor initialized with window size {window_size}")
    
    def record_response_time(self, elapsed: float):
        """Record overall response time.
        
        Args:
            elapsed: Time in seconds
        """
        with monitor_lock:
            self.response_times.append(elapsed)
            self.total_requests += 1
    
    def record_embedding_time(self, elapsed: float):
        """Record embedding generation time.
        
        Args:
            elapsed: Time in seconds
        """
        with monitor_lock:
            self.embedding_times.append(elapsed)
    
    def record_search_time(self, elapsed: float):
        """Record vector search time.
        
        Args:
            elapsed: Time in seconds
        """
        with monitor_lock:
            self.search_times.append(elapsed)
    
    def record_llm_time(self, elapsed: float):
        """Record LLM generation time.
        
        Args:
            elapsed: Time in seconds
        """
        with monitor_lock:
            self.llm_times.append(elapsed)
    
    def record_memory_usage(self, memory_info: Dict[str, float]):
        """Record memory usage snapshot.
        
        Args:
            memory_info: Dictionary with memory usage information
        """
        with monitor_lock:
            self.memory_snapshots.append(memory_info)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics.
        
        Returns:
            Dictionary with statistics for each tracked metric
        """
        with monitor_lock:
            stats = {
                "response": self._calc_stats(self.response_times),
                "embedding": self._calc_stats(self.embedding_times),
                "search": self._calc_stats(self.search_times),
                "llm": self._calc_stats(self.llm_times),
                "general": {
                    "total_requests": self.total_requests,
                    "uptime_seconds": time.time() - self.start_time,
                    "requests_per_minute": self.total_requests / ((time.time() - self.start_time) / 60)
                }
            }
            
            # Add memory stats if available
            if self.memory_snapshots:
                latest = self.memory_snapshots[-1]
                avg_used = sum(snap.get("used_mb", 0) for snap in self.memory_snapshots) / len(self.memory_snapshots)
                
                stats["memory"] = {
                    "current_used_mb": latest.get("used_mb", 0),
                    "current_available_mb": latest.get("available_mb", 0),
                    "current_percent": latest.get("percent", 0),
                    "avg_used_mb": avg_used
                }
            
            return stats
    
    def _calc_stats(self, times: Deque[float]) -> Dict[str, float]:
        """Calculate statistics for a time series.
        
        Args:
            times: Collection of time measurements
            
        Returns:
            Dictionary with statistics
        """
        if not times:
            return {"avg": 0, "min": 0, "max": 0, "p95": 0, "count": 0}
        
        times_list = list(times)
        times_list.sort()
        p95_idx = int(len(times_list) * 0.95)
        
        return {
            "avg": sum(times_list) / len(times_list),
            "min": min(times_list),
            "max": max(times_list),
            "p95": times_list[p95_idx] if p95_idx < len(times_list) else times_list[-1],
            "count": len(times_list)
        }
    
    def reset(self):
        """Reset all collected statistics."""
        with monitor_lock:
            self.response_times.clear()
            self.embedding_times.clear()
            self.search_times.clear()
            self.llm_times.clear()
            self.memory_snapshots.clear()
            self.total_requests = 0
            self.start_time = time.time()
            logger.info("Performance monitor reset")


# Create singleton instance
performance_monitor = PerformanceMonitor() 