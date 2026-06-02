# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Detect and clean up orphaned vLLM/Ray GPU worker processes.

Orphaned processes occur when NeMo RL triggers an unclean shutdown
(timeout, OOM, signal) and vLLM workers under the Ray distributed
executor aren't properly terminated. They hold GPU memory and NCCL
communicators, blocking subsequent runs.
"""

import os
import signal
import subprocess

from vllm.logger import init_logger

logger = init_logger(__name__)


def cleanup_orphaned_gpu_workers() -> int:
    """Find and kill orphaned vLLM Ray worker processes holding GPU memory.

    Strategy:
    1. Use nvidia-smi to find processes using GPU memory
    2. Filter to Python processes that look like vLLM/Ray workers
    3. Check if they belong to the current Ray session (if Ray is running)
    4. Kill orphans that don't belong to any active Ray session

    Returns:
        Number of orphaned processes killed.
    """
    if os.environ.get("VLLM_CLEANUP_ORPHANS_ON_STARTUP", "1") == "0":
        logger.info("Orphan cleanup disabled (VLLM_CLEANUP_ORPHANS_ON_STARTUP=0)")
        return 0

    try:
        gpu_pids = _get_gpu_process_pids()
    except Exception as e:
        logger.warning("Failed to query GPU processes: %s", e)
        return 0

    if not gpu_pids:
        logger.debug("No GPU processes found — nothing to clean up.")
        return 0

    current_pid = os.getpid()
    current_ray_session = _get_current_ray_session_id()

    killed = 0
    for pid, gpu_mem_mb, cmdline in gpu_pids:
        if pid == current_pid:
            continue

        # Only target vLLM/Ray worker processes
        if not _is_vllm_ray_worker(cmdline):
            continue

        # If Ray is running, check if process belongs to current session
        if current_ray_session and _belongs_to_ray_session(pid, current_ray_session):
            continue

        logger.warning(
            "Killing orphaned vLLM/Ray worker process: pid=%d, gpu_mem=%dMB, cmd=%s",
            pid, gpu_mem_mb, cmdline[:100],
        )
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass  # Already dead
        except PermissionError:
            logger.warning("Permission denied killing pid %d", pid)

    if killed:
        logger.info("Cleaned up %d orphaned GPU worker process(es).", killed)
    return killed


def _get_gpu_process_pids() -> list[tuple[int, int, str]]:
    """Get (pid, gpu_mem_mb, cmdline) for all GPU processes."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
    except FileNotFoundError:
        return []

    processes = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0].strip())
            mem = int(parts[1].strip())
        except ValueError:
            continue

        # Get command line for the process
        cmdline = _get_process_cmdline(pid)
        processes.append((pid, mem, cmdline))

    return processes


def _get_process_cmdline(pid: int) -> str:
    """Get the command line for a process."""
    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            return f.read().replace("\0", " ").strip()
    except (FileNotFoundError, PermissionError):
        return ""


def _is_vllm_ray_worker(cmdline: str) -> bool:
    """Check if a process looks like a vLLM Ray worker."""
    indicators = [
        "ray::RayWorkerWrapper",
        "vllm.v1.worker",
        "vllm.worker",
        "ray::IDLE",  # Ray idle workers can hold GPU memory
    ]
    return any(ind in cmdline for ind in indicators)


def _get_current_ray_session_id() -> str | None:
    """Get the current Ray session ID, if Ray is initialized."""
    try:
        import ray
        if ray.is_initialized():
            return ray.get_runtime_context().get_job_id()
    except Exception:
        pass
    return None


def _belongs_to_ray_session(pid: int, session_id: str) -> bool:
    """Check if a process belongs to the given Ray session.

    This is best-effort — checks environment variables of the process.
    """
    try:
        with open(f"/proc/{pid}/environ", "r") as f:
            environ = f.read()
            return session_id in environ
    except (FileNotFoundError, PermissionError):
        return False  # Can't tell — assume orphan


def register_shutdown_handlers(executor):
    """Register signal handlers and atexit for clean shutdown.

    Ensures all Ray workers are killed and GPU memory released
    even on SIGTERM/SIGINT.
    """
    import atexit

    def _cleanup_on_signal(signum, frame):
        logger.info("Received signal %d, shutting down executor...", signum)
        try:
            executor.shutdown()
        except Exception as e:
            logger.warning("Error during signal-triggered shutdown: %s", e)
        # Re-raise the signal for the default handler
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # Register for SIGTERM (container kill, Slurm timeout)
    signal.signal(signal.SIGTERM, _cleanup_on_signal)

    # atexit for normal Python exit
    atexit.register(executor.shutdown)
