from __future__ import annotations

import statistics
import time
from typing import Callable


def benchmark_function(
    fn: Callable[[], None],
    warmup: int = 5,
    runs: int = 20,
) -> dict:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if runs <= 0:
        raise ValueError("runs must be > 0")

    for _ in range(warmup):
        fn()

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        timings.append(end - start)

    return {
        "warmup": warmup,
        "runs": runs,
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "all_seconds": timings,
    }