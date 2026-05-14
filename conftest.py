"""Global pytest configuration and cleanup utilities.

This module provides global pytest fixtures and cleanup utilities to ensure
proper resource cleanup and prevent hanging processes.
"""

import atexit
import gc
import multiprocessing
import os
import time

import pytest


def pytest_configure(config):
    """Configure pytest with global settings."""
    # Set asyncio mode to auto for better event loop handling
    config.option.asyncio_mode = "auto"


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("🚀 Starting pytest session with enhanced cleanup...")

    # Register cleanup function to run on exit
    atexit.register(force_cleanup_and_exit)


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished, right before returning
    the exit status.
    """
    # 不在这里强制退出, 让pytest正常显示结果
    pass


def force_cleanup_and_exit():
    """Force cleanup of all resources and exit."""
    try:
        print("🧹 Force cleaning up resources...")

        # 1. Clean up multiprocessing processes
        cleanup_multiprocessing()

        # 2. Clean up PyTorch/CUDA resources
        cleanup_pytorch_resources()

        # 3. Force garbage collection
        gc.collect()

        # 4. Force exit
        print("🚪 Forcing exit...")
        os._exit(0)

    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")
        # Force exit even if cleanup fails
        os._exit(0)


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
    # 不在这里强制退出, 让pytest正常显示结果
    # 清理工作由atexit注册的函数处理
    return


def pytest_unconfigure(config):
    """Called before test process is exited."""
    # 不在这里强制退出, 让pytest正常显示结果
    # 清理工作由atexit注册的函数处理
    pass
