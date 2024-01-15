"""
Module with decorator for logging time spends.
"""
import logging
import time
from typing import Any, Callable

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')


def report_time(fn_to_wrap: Callable) -> Callable:
    """
    Decorator for logging time spends.
    """
    def _internal(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        res = fn_to_wrap(*args, **kwargs)
        duration = time.time() - start

        logging.info('%s took %2.3f sec', fn_to_wrap.__name__, duration)

        return res

    return _internal
