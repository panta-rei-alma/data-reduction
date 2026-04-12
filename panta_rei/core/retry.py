"""Simple retry decorator with exponential backoff.

Applied to ALMA API calls and GitHub API calls.
No retry for DB operations or filesystem operations.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 5.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Retry a function on failure with exponential backoff.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts (including the first call).
    delay : float
        Initial delay in seconds between retries.
    backoff : float
        Multiplier applied to delay after each retry.
    exceptions : tuple
        Exception types that trigger a retry.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        log.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    log.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.1fs...",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            # Should not reach here, but satisfy type checker
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
