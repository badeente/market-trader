import time
import logging
import psutil
import threading
from collections import defaultdict
from functools import wraps
import numpy as np
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfilerContext:
    """Context manager for profiling a code block"""
    def __init__(self, profiler, component_name):
        self.profiler = profiler
        self.component_name = component_name
        self.start_time = None
        self.start_cpu = None
    
    def __enter__(self):
        """Start timing when entering the context"""
        self.start_time = time.time()
        self.start_cpu = self.profiler._process.cpu_percent()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record timing when exiting the context"""
        if exc_type is None:  # Only record if no exception occurred
            execution_time = time.time() - self.start_time
            end_cpu = self.profiler._process.cpu_percent()
            cpu_usage = (self.start_cpu + end_cpu) / 2
            
            self.profiler.timings[self.component_name].append(execution_time)
            self.profiler.cpu_usage[self.component_name].append(cpu_usage)

class PerformanceProfiler:
    def __init__(self, log_dir=None):
        self.timings = defaultdict(list)
        self.cpu_usage = defaultdict(list)
        self._process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None
        self.log_dir = log_dir
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def profile(self, component_name):
        """Create a context manager for profiling a code block"""
        return ProfilerContext(self, component_name)
    
    def start_cpu_monitoring(self):
        """Start continuous CPU monitoring in a separate thread"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_cpu)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_cpu_monitoring(self):
        """Stop CPU monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_cpu(self):
        """Monitor CPU usage continuously"""
        while self._monitoring:
            cpu_percent = self._process.cpu_percent(interval=1)
            self.cpu_usage['overall'].append(cpu_percent)
            time.sleep(0.1)  # Reduce sampling frequency to minimize overhead
    
    def get_statistics(self):
        """Get performance statistics for all monitored components"""
        stats = {}
        
        for component in self.timings.keys():
            times = self.timings[component]
            cpu = self.cpu_usage[component]
            
            if times:
                stats[component] = {
                    'execution_time': {
                        'mean': np.mean(times),
                        'median': np.median(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'std': np.std(times),
                        'total': np.sum(times),
                        'calls': len(times)
                    },
                    'cpu_usage': {
                        'mean': np.mean(cpu) if cpu else 0,
                        'median': np.median(cpu) if cpu else 0,
                        'min': np.min(cpu) if cpu else 0,
                        'max': np.max(cpu) if cpu else 0,
                        'std': np.std(cpu) if cpu else 0
                    }
                }
        
        return stats
    
    def save_statistics(self, filename='performance_stats.txt'):
        """Save performance statistics to a file"""
        if not self.log_dir:
            logger.warning("No log directory specified. Cannot save performance statistics.")
            return
            
        stats = self.get_statistics()
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                f.write("Performance Statistics Report\n")
                f.write("===========================\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Sort components by total execution time
                sorted_components = sorted(
                    stats.items(),
                    key=lambda x: x[1]['execution_time']['total'],
                    reverse=True
                )
                
                for component, data in sorted_components:
                    time_stats = data['execution_time']
                    cpu_stats = data['cpu_usage']
                    
                    f.write(f"\nComponent: {component}\n")
                    f.write("Execution Time:\n")
                    f.write(f"  Total: {time_stats['total']:.2f}s\n")
                    f.write(f"  Calls: {time_stats['calls']}\n")
                    f.write(f"  Mean: {time_stats['mean']:.4f}s\n")
                    f.write(f"  Median: {time_stats['median']:.4f}s\n")
                    f.write(f"  Min: {time_stats['min']:.4f}s\n")
                    f.write(f"  Max: {time_stats['max']:.4f}s\n")
                    f.write(f"  Std Dev: {time_stats['std']:.4f}s\n")
                    
                    f.write("CPU Usage:\n")
                    f.write(f"  Mean: {cpu_stats['mean']:.1f}%\n")
                    f.write(f"  Median: {cpu_stats['median']:.1f}%\n")
                    f.write(f"  Min: {cpu_stats['min']:.1f}%\n")
                    f.write(f"  Max: {cpu_stats['max']:.1f}%\n")
                    f.write(f"  Std Dev: {cpu_stats['std']:.1f}%\n")
                    f.write("\n" + "="*50 + "\n")
                
                logger.info(f"Performance statistics saved to {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving performance statistics: {e}")
    
    def reset(self):
        """Reset all performance metrics"""
        self.timings.clear()
        self.cpu_usage.clear()
