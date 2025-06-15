"""
Utility decorators for common patterns.

This module provides reusable decorators for error handling,
retries, and other cross-cutting concerns.
"""

import functools
import logging
import time
from typing import TypeVar, Callable, Any, Optional, Type, Tuple

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on error with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for each retry
        exceptions: Tuple of exception types to catch and retry
        logger: Optional logger for retry messages
        
    Returns:
        Decorated function that retries on specified errors
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
                
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                            f" - Retrying in {current_delay:.1f}s"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        
            # If we get here, all attempts failed
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"{func.__name__} failed without exception")
                
        return wrapper
    return decorator