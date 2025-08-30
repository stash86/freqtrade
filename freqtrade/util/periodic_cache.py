from datetime import UTC, datetime

from cachetools import TTLCache


class PeriodicCache(TTLCache):
    """
    Special cache that expires at "straight" times
    A timer with ttl of 3600 (1h) will expire at every full hour (:00).
    """

    def __init__(self, maxsize, ttl, getsizeof=None):
        def local_timer():
            ts = datetime.now(UTC).timestamp()
            offset = ts % ttl
            return ts - offset

        # Init with smlight offset
        super().__init__(maxsize=maxsize, ttl=ttl - 1e-5, timer=local_timer, getsizeof=getsizeof)

    def log_with_cache(self, message: str, log_func, cache_key: str | None):
        """
        Log message only if it hasn't been logged recently
        :param message: Message to log
        :param log_func: Logger function (logger.info, logger.warning, etc.)
        :param cache_key: Custom cache key (defaults to message)
        :param ttl: Time to live in seconds
        """
        key = cache_key or message
        if key not in self:
            log_func(message)
            self[key] = True
