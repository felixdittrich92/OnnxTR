# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from onnxtr.file_utils import ENV_VARS_TRUE_VALUES

__all__ = ["multithread_exec"]


def multithread_exec(func: Callable[[Any], Any], seq: Iterable[Any], threads: int | None = None) -> Iterator[Any]:
    """Execute a given function in parallel for each element of a given sequence

    >>> from onnxtr.utils.multithreading import multithread_exec
    >>> entries = [1, 4, 8]
    >>> results = multithread_exec(lambda x: x ** 2, entries)

    Args:
        func: function to be executed on each element of the iterable
        seq: iterable
        threads: number of workers to be used for multiprocessing

    Returns:
        iterator of the function's results using the iterable as inputs

    Notes:
        This function uses ThreadPool from multiprocessing package, which uses `/dev/shm` directory for shared memory.
        If you do not have write permissions for this directory (if you run `onnxtr` on AWS Lambda for instance),
        you might want to disable multiprocessing. To achieve that, set 'ONNXTR_MULTIPROCESSING_DISABLE' to 'TRUE'.
    """
    threads = threads if isinstance(threads, int) else min(16, mp.cpu_count())
    items = seq if isinstance(seq, (list, tuple)) else list(seq)
    # Never spawn more workers than items - single-item calls skip pool startup entirely
    threads = min(threads, len(items))
    # Single-thread
    if threads < 2 or os.environ.get("ONNXTR_MULTIPROCESSING_DISABLE", "").upper() in ENV_VARS_TRUE_VALUES:
        results: Iterator[Any] | map[Any] = map(func, items)
    # Multi-threading
    else:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Materialize inside the context so all workers are joined before returning
            results = iter(list(executor.map(func, items)))
    return results
