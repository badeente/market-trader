import logging
from collections import OrderedDict
import pandas as pd
from backtester.plot_candlesticks import plot_candlesticks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScreenshotManager:
    def __init__(self, max_cache_size=10000):
        """
        Initialize the screenshot manager with a maximum cache size.
        
        Args:
            max_cache_size (int): Maximum number of screenshots to keep in memory
        """
        self.max_cache_size = max_cache_size
        self._cache = OrderedDict()  # Use OrderedDict for LRU cache
        logger.info(f"Initialized ScreenshotManager with max cache size: {max_cache_size}")
    
    def _generate_cache_key(self, window):
        """
        Generate a unique key for the data window.
        Uses the first and last timestamp plus the shape of the data.
        """
        if isinstance(window, pd.DataFrame):
            start_time = str(window.index[0])
            end_time = str(window.index[-1])
            shape = window.shape
            return f"{start_time}_{end_time}_{shape}"
        return None
    
    def get_screenshot(self, window, num_entries=200, force_regenerate=False):
        """
        Get a screenshot for the given data window. If it exists in cache and
        force_regenerate is False, return cached version. Otherwise, generate new screenshot.
        
        Args:
            window: The data window to generate screenshot from
            num_entries (int): Number of entries to show in screenshot
            force_regenerate (bool): Whether to force regeneration of screenshot
            
        Returns:
            bytes: The screenshot data in PNG format
        """
        if window is None:
            return None
            
        cache_key = self._generate_cache_key(window)
        if cache_key is None:
            logger.warning("Could not generate cache key for window")
            return None
        
        # Return cached screenshot if available and not forcing regeneration
        if not force_regenerate and cache_key in self._cache:
            logger.debug("Using cached screenshot")
            self._cache.move_to_end(cache_key)  # Move to end (most recently used)
            return self._cache[cache_key]
        
        # Generate new screenshot
        try:
            screenshot = plot_candlesticks(window, num_entries=num_entries)
            
            # Add to cache
            self._cache[cache_key] = screenshot
            self._cache.move_to_end(cache_key)
            
            # Remove oldest items if cache is too large
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)  # Remove oldest item
            
            logger.debug(f"Generated new screenshot. Cache size: {len(self._cache)}")
            return screenshot
            
        except Exception as e:
            logger.error(f"Error generating screenshot: {e}")
            return None
    
    def clear_cache(self):
        """Clear the screenshot cache"""
        self._cache.clear()
        logger.info("Screenshot cache cleared")
    
    def get_cache_info(self):
        """Get information about the current cache state"""
        return {
            'current_size': len(self._cache),
            'max_size': self.max_cache_size,
            'memory_usage': sum(len(screenshot) for screenshot in self._cache.values())
        }
