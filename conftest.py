"""Global pytest configuration and cleanup utilities.

This module provides global pytest fixtures and cleanup utilities to ensure
proper resource cleanup and prevent hanging processes.
"""

import gc
import multiprocessing
import time

import pytest


def pytest_configure(config):
    """Configure pytest with global settings."""
    # Set asyncio mode to auto for better event loop handling
    config.option.asyncio_mode = "auto"


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("🚀 Starting pytest session with enhanced cleanup...")
    # No interpreter exit-handler registration here: cleanup runs in
    # pytest_sessionfinish, which receives and preserves the real exit status.


def pytest_sessionfinish(session, exitstatus):
    """Whole-run cleanup; `exitstatus` propagates untouched because the exit is never forced."""
    cleanup_multiprocessing()
    cleanup_pytorch_resources()
    gc.collect()
    # Never force the process exit here: returning propagates `exitstatus` unchanged.


def cleanup_multiprocessing():
    """Clean up all multiprocessing processes."""
    try:
        active_children = multiprocessing.active_children()

        if active_children:
            print(f"🧹 Cleaning up {len(active_children)} multiprocessing processes...")

            # Terminate all processes
            for process in active_children:
                try:
                    if process.is_alive():
                        process.terminate()
                except Exception as e:
                    print(f"Warning: Failed to terminate process {process.pid}: {e}")

            # Wait briefly for termination
            time.sleep(0.1)

            # Force kill any remaining processes
            for process in active_children:
                try:
                    if process.is_alive():
                        process.kill()
                except Exception as e:
                    print(f"Warning: Failed to kill process {process.pid}: {e}")

    except Exception as e:
        print(f"Warning: Error during multiprocessing cleanup: {e}")


def cleanup_pytorch_resources():
    """Clean up PyTorch and CUDA resources."""
    try:
        import torch

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    except Exception as e:
        print(f"Warning: Error during PyTorch cleanup: {e}")


@pytest.fixture(scope="session", autouse=True)
def global_cleanup():
    """Global cleanup fixture that runs after all tests."""
    # No forced exit here — pytest must display results normally
    # Cleanup is handled by pytest_sessionfinish
    return


def pytest_unconfigure(config):
    """Called before test process is exited."""
    # No forced exit here — pytest must display results normally
    # Cleanup is handled by pytest_sessionfinish
    pass
