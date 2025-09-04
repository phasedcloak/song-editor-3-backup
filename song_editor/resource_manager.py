#!/usr/bin/env python3
"""
Resource Management for Song Editor 3

Handles memory management, performance monitoring, and resource cleanup.
"""

import gc
import psutil
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import signal


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    percent: float = 0.0
    available_mb: float = 0.0
    total_mb: float = 0.0


@dataclass
class PerformanceStats:
    """Performance statistics."""
    cpu_percent: float = 0.0
    memory: MemoryStats = field(default_factory=MemoryStats)
    thread_count: int = 0
    open_files: int = 0
    timestamp: float = field(default_factory=time.time)


class ResourceManager:
    """Manages system resources and performance monitoring."""

    def __init__(self, memory_limit_mb: Optional[int] = None, cpu_limit_percent: Optional[float] = None):
        self.memory_limit_mb = memory_limit_mb or (psutil.virtual_memory().total // (1024 * 1024) * 0.8)  # 80% of total
        self.cpu_limit_percent = cpu_limit_percent or 90.0
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_callbacks: List[Callable[[PerformanceStats], None]] = []
        self.cleanup_callbacks: List[Callable[[], None]] = []
        self._stats_history: List[PerformanceStats] = []
        self._lock = threading.Lock()

    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
            name="ResourceMonitor"
        )
        self.monitor_thread.start()
        logging.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logging.info("Resource monitoring stopped")

    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                stats = self.get_performance_stats()

                # Check limits
                if stats.memory.rss_mb > self.memory_limit_mb:
                    logging.warning(
                        f"Memory usage ({stats.memory.rss_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)"
                    )
                    self._trigger_memory_cleanup()

                if stats.cpu_percent > self.cpu_limit_percent:
                    logging.warning(f"CPU usage ({stats.cpu_percent:.1f}%) exceeds limit ({self.cpu_limit_percent}%)")

                # Store in history
                with self._lock:
                    self._stats_history.append(stats)
                    # Keep only last 100 entries
                    if len(self._stats_history) > 100:
                        self._stats_history.pop(0)

                # Call callbacks
                for callback in self.monitor_callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logging.error(f"Error in monitor callback: {e}")

            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")

            time.sleep(interval)

    def get_performance_stats(self) -> PerformanceStats:
        """Get current performance statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        vm = psutil.virtual_memory()

        memory_stats = MemoryStats(
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent=memory_percent,
            available_mb=vm.available / (1024 * 1024),
            total_mb=vm.total / (1024 * 1024)
        )

        return PerformanceStats(
            cpu_percent=process.cpu_percent(),
            memory=memory_stats,
            thread_count=len(process.threads()),
            open_files=len(process.open_files()),
            timestamp=time.time()
        )

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self.get_performance_stats().memory

    def get_stats_history(self, limit: Optional[int] = None) -> List[PerformanceStats]:
        """Get performance statistics history."""
        with self._lock:
            if limit:
                return self._stats_history[-limit:]
            return self._stats_history.copy()

    def add_monitor_callback(self, callback: Callable[[PerformanceStats], None]) -> None:
        """Add a monitoring callback."""
        self.monitor_callbacks.append(callback)

    def remove_monitor_callback(self, callback: Callable[[PerformanceStats], None]) -> None:
        """Remove a monitoring callback."""
        if callback in self.monitor_callbacks:
            self.monitor_callbacks.remove(callback)

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a cleanup callback."""
        self.cleanup_callbacks.append(callback)

    def remove_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Remove a cleanup callback."""
        if callback in self.cleanup_callbacks:
            self.cleanup_callbacks.remove(callback)

    def _trigger_memory_cleanup(self) -> None:
        """Trigger memory cleanup."""
        logging.info("Triggering memory cleanup...")

        # Force garbage collection
        gc.collect()

        # Call cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Error in cleanup callback: {e}")

        # Log memory after cleanup
        after_stats = self.get_memory_stats()
        logging.info(f"Memory after cleanup: {after_stats.rss_mb:.1f}MB")

    def force_cleanup(self) -> None:
        """Force immediate cleanup."""
        self._trigger_memory_cleanup()

    def optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        logging.info("Optimizing memory usage...")

        # Force garbage collection
        gc.collect()

        # Clear any caches if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared")
        except ImportError:
            pass

        logging.info("Memory optimization completed")

    @contextmanager
    def memory_limit_context(self, limit_mb: int):
        """Context manager for temporary memory limits."""
        old_limit = self.memory_limit_mb
        self.memory_limit_mb = limit_mb
        try:
            yield
        finally:
            self.memory_limit_mb = old_limit

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        vm = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        cpu_logical = psutil.cpu_count(logical=True)

        return {
            'cpu_count': cpu_count,
            'cpu_logical': cpu_logical,
            'memory_total_mb': vm.total / (1024 * 1024),
            'memory_available_mb': vm.available / (1024 * 1024),
            'memory_used_percent': vm.percent,
            'platform': os.uname().sysname if hasattr(os, 'uname') else 'Unknown',
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }

    def cleanup_on_exit(self) -> None:
        """Cleanup resources on exit."""
        logging.info("Cleaning up resources...")

        # Stop monitoring
        self.stop_monitoring()

        # Force final cleanup
        self.force_cleanup()

        # Clear callbacks
        self.monitor_callbacks.clear()
        self.cleanup_callbacks.clear()

        logging.info("Resource cleanup completed")


# Global resource manager instance
_resource_manager = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def init_resource_manager(memory_limit_mb: Optional[int] = None) -> ResourceManager:
    """Initialize the global resource manager."""
    global _resource_manager
    _resource_manager = ResourceManager(memory_limit_mb=memory_limit_mb)
    return _resource_manager


# Convenience functions
def start_resource_monitoring(interval: float = 5.0) -> None:
    """Start resource monitoring."""
    get_resource_manager().start_monitoring(interval)


def stop_resource_monitoring() -> None:
    """Stop resource monitoring."""
    get_resource_manager().stop_monitoring()


def get_memory_usage() -> MemoryStats:
    """Get current memory usage."""
    return get_resource_manager().get_memory_stats()


def force_memory_cleanup() -> None:
    """Force memory cleanup."""
    get_resource_manager().force_cleanup()


def optimize_memory() -> None:
    """Optimize memory usage."""
    get_resource_manager().optimize_memory_usage()


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return get_resource_manager().get_system_info()


# Signal handler for graceful shutdown
def _signal_handler(signum, frame):
    """Handle shutdown signals."""
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    if _resource_manager:
        _resource_manager.cleanup_on_exit()


# Register signal handlers
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# Context manager for resource monitoring
@contextmanager
def resource_monitor(interval: float = 5.0):
    """Context manager for resource monitoring."""
    manager = get_resource_manager()
    manager.start_monitoring(interval)
    try:
        yield manager
    finally:
        manager.stop_monitoring()
