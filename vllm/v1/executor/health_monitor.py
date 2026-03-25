# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monitor weight loading progress across Ray workers.
Detects stalled workers by checking if shard progress advances.
"""

import threading
import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

try:
    import ray
except ImportError:
    ray = None


class LoadProgressMonitor:
    """Monitors weight loading progress across workers.

    Polls workers for their load progress (shards loaded / total shards).
    If any worker's progress doesn't advance for `stall_timeout` seconds,
    raises an error by killing the stalled workers.

    Usage:
        monitor = LoadProgressMonitor(workers, stall_timeout=120)
        monitor.start()
        try:
            collective_rpc("load_model")
        finally:
            monitor.stop()
    """

    def __init__(self, workers: list, stall_timeout: int = 120,
                 poll_interval: int = 10):
        self.workers = workers
        self.stall_timeout = stall_timeout
        self.poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Track last known progress per worker: (loaded, total, timestamp)
        self._last_progress: dict[int, tuple[int, int, float]] = {}

    def start(self):
        """Start the progress monitoring thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="load-progress-monitor"
        )
        self._thread.start()
        logger.info(
            "Load progress monitor started (stall_timeout=%ds, poll_interval=%ds)",
            self.stall_timeout, self.poll_interval,
        )

    def stop(self):
        """Stop the progress monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Load progress monitor stopped.")

    def _monitor_loop(self):
        """Main monitoring loop running in a background thread."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval)
            if self._stop_event.is_set():
                break

            try:
                self._check_progress()
            except _StallDetected as e:
                logger.error("STALL DETECTED: %s", e)
                # Kill stalled workers to unblock ray.get() in the main thread
                for rank in e.stalled_ranks:
                    try:
                        ray.kill(self.workers[rank], no_restart=True)
                    except Exception:
                        pass
                break
            except Exception as e:
                # Don't let monitoring errors kill the monitor
                logger.warning("Load progress monitor error: %s", e)

    def _check_progress(self):
        """Poll all workers for load progress, detect stalls."""
        now = time.monotonic()

        # Ask all workers for their current load progress
        try:
            refs = [
                worker.execute_method.remote("get_load_progress")
                for worker in self.workers
            ]
            # Short timeout for the poll itself
            ready, not_ready = ray.wait(refs, num_returns=len(refs), timeout=30)
            results = {i: ray.get(ref) for i, ref in enumerate(refs) if ref in ready}
        except Exception as e:
            logger.debug("Failed to poll load progress: %s", e)
            return

        stalled_ranks = []
        for rank in range(len(self.workers)):
            progress = results.get(rank)
            if progress is None:
                # Worker didn't respond to poll — may be stuck
                if rank not in self._last_progress:
                    self._last_progress[rank] = (0, 0, now)
                last_loaded, last_total, last_time = self._last_progress[rank]
                if now - last_time > self.stall_timeout:
                    stalled_ranks.append(rank)
                continue

            loaded, total = progress
            if rank not in self._last_progress:
                self._last_progress[rank] = (loaded, total, now)
                logger.info("Worker %d: loading started (%d/%d shards)", rank, loaded, total)
                continue

            last_loaded, last_total, last_time = self._last_progress[rank]
            if loaded > last_loaded or (loaded == total and total > 0):
                # Progress made — update timestamp
                self._last_progress[rank] = (loaded, total, now)
                if loaded == total and total > 0:
                    logger.info("Worker %d: loading complete (%d/%d shards)", rank, loaded, total)
                elif loaded % 10 == 0:  # Log every 10 shards
                    logger.info("Worker %d: loading progress %d/%d shards", rank, loaded, total)
            else:
                # No progress since last check
                stall_duration = now - last_time
                if stall_duration > self.stall_timeout:
                    stalled_ranks.append(rank)
                elif stall_duration > self.stall_timeout / 2:
                    logger.warning(
                        "Worker %d: no progress for %.0fs (%d/%d shards). "
                        "Will declare stall after %ds.",
                        rank, stall_duration, loaded, total, self.stall_timeout,
                    )

        if stalled_ranks:
            raise _StallDetected(stalled_ranks, self.stall_timeout, self._last_progress)


class _StallDetected(Exception):
    """Raised when stalled workers are detected."""

    def __init__(self, stalled_ranks: list[int], timeout: int,
                 progress: dict[int, tuple[int, int, float]]):
        self.stalled_ranks = stalled_ranks
        details = []
        for rank in stalled_ranks:
            if rank in progress:
                loaded, total, _ = progress[rank]
                details.append(f"rank {rank}: stuck at {loaded}/{total} shards")
            else:
                details.append(f"rank {rank}: no progress reported")
        super().__init__(
            f"Workers stalled for >{timeout}s with no weight loading progress. "
            f"Details: {'; '.join(details)}. "
            f"This likely indicates Ray GCS overload or a network/storage issue. "
            f"Set VLLM_LOAD_STALL_TIMEOUT to adjust (current: {timeout}s)."
        )
