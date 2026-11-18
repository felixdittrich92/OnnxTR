# Copyright (C) 2021-2026, Mindee | Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import multiprocessing as mp
import os
import threading
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Any

import numpy as np

from onnxtr.file_utils import ENV_VARS_TRUE_VALUES

__all__ = ["multithread_exec", "chunked", "prefetch", "prefetch_chunks"]


def multithread_exec(func: Callable[[Any], Any], seq: Iterable[Any], threads: int | None = None) -> Iterator[Any]:
    """Execute a given function in parallel for each element of a given sequence

    >>> from onnxtr.utils.multithreading import multithread_exec
    >>> entries = [1, 4, 8]
    >>> results = multithread_exec(lambda x: x ** 2, entries)

    Args:
        func: function to be executed on each element of the iterable
        seq: iterable
        threads: number of worker threads to use, defaults to `min(16, cpu_count())`

    Returns:
        iterator of the function's results, in the same order as the inputs

    Notes:
        Parallelism is provided by `concurrent.futures.ThreadPoolExecutor`, so `func` only
        benefits from it if it releases the GIL (I/O, or native code such as NumPy, OpenCV
        or Pillow). No worker threads are spawned for sequences shorter than two items.

        Results are materialized before returning, so the pool is fully shut down and every
        worker joined by the time the iterator is handed back. To opt out of threading
        entirely, set 'ONNXTR_MULTIPROCESSING_DISABLE' to 'TRUE' and `func` will be applied
        lazily on the calling thread.
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


def chunked(seq: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Split an iterable into consecutive lists of at most `size` elements, consuming it lazily

    >>> from onnxtr.utils.multithreading import chunked
    >>> list(chunked(range(5), 2))
    [[0, 1], [2, 3], [4]]

    Args:
        seq: iterable
        size: the maximum number of elements per chunk

    Returns:
        iterator of chunks
    """
    if size < 1:
        raise ValueError("the chunk size must be a positive integer")
    iterator = iter(seq)
    while chunk := list(islice(iterator, size)):
        yield chunk


_EXHAUSTED = object()


def prefetch(seq: Iterable[Any], enabled: bool = True) -> Generator[Any, None, None]:
    """Iterate over an iterable while producing its next element in a background thread

    This overlaps the production of the next element (e.g. decoding the next pages of a document) with the
    processing of the current one (e.g. running the models). At most one element is produced ahead, and the
    iterable is only ever advanced by one thread at a time.

    >>> from onnxtr.utils.multithreading import prefetch
    >>> for item in prefetch(expensive_generator()):
    ...     process(item)

    Args:
        seq: iterable
        enabled: whether the elements are produced in a background thread

    Returns:
        generator over the elements of the iterable: closing it waits for the element being produced, if any

    Notes:
        Prefetching is disabled when 'ONNXTR_MULTIPROCESSING_DISABLE' is set to 'TRUE'.
    """
    if not enabled or os.environ.get("ONNXTR_MULTIPROCESSING_DISABLE", "").upper() in ENV_VARS_TRUE_VALUES:
        yield from seq
        return

    iterator = iter(seq)
    try:
        # On early exit, leaving the context waits for the element being produced, so that the iterable is
        # never closed while it is advanced in the background thread
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="onnxtr-prefetch") as executor:
            future = executor.submit(next, iterator, _EXHAUSTED)
            while (item := future.result()) is not _EXHAUSTED:
                future = executor.submit(next, iterator, _EXHAUSTED)
                yield item
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def prefetch_chunks(seq: Iterable[Any], size: int, enabled: bool = True) -> Generator[list[Any], None, None]:
    """Split an iterable into consecutive lists of at most `size` elements, producing the next chunk in a background
    thread while the current one is processed

    This is how the predictors consume their inputs: a lazy page source (or any iterator) has its next chunk decoded
    ahead, while in-memory inputs (lists, tuples, arrays) have nothing to decode and are only sliced, without a
    background thread.

    >>> from onnxtr.utils.multithreading import prefetch_chunks
    >>> for chunk in prefetch_chunks(lazy_source, 32):
    ...     process(chunk)

    Args:
        seq: iterable
        size: the maximum number of elements per chunk
        enabled: whether the next chunk is produced in a background thread (ignored for in-memory inputs)

    Returns:
        generator of chunks. Closing it (e.g. when the consumer stops early, or on error) stops producing ahead after
        the element in progress rather than completing a whole chunk, and releases the resources held by the
        iteration over `seq` (e.g. an open PDF).

    Notes:
        Prefetching is disabled when 'ONNXTR_MULTIPROCESSING_DISABLE' is set to 'TRUE'.
    """
    in_memory = isinstance(seq, (list, tuple, np.ndarray))
    stop = threading.Event()

    def _until_stopped() -> Iterator[Any]:
        iterator = iter(seq)
        try:
            for item in iterator:
                yield item
                if stop.is_set():
                    return
        finally:
            # Release the resources held by the iteration (e.g. an open PDF) as soon as it is over
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    chunks = prefetch(chunked(_until_stopped(), size), enabled=enabled and not in_memory)
    try:
        # Not `yield from`: on early exit, it would close `chunks` (waiting for a whole chunk) before `stop` is set
        for chunk in chunks:
            yield chunk
    finally:
        stop.set()
        chunks.close()
