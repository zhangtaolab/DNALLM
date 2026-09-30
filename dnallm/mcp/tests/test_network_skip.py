"""Offline unit tests for the typed network-skip helper.

These prove the all-leaves rule of ``skip_if_unreachable`` without any
server: a group whose every leaf is an httpx transport error skips the
test, while any non-network leaf re-raises so the test fails honestly.
They run in the fast leg (no ``slow`` marker, no network).
"""

from __future__ import annotations

import httpx
import pytest

from dnallm.mcp.tests._network_skip import skip_if_unreachable


class _FakeGroupError(Exception):
    """Minimal ExceptionGroup-like carrier.

    Duck-typed: only the ``exceptions`` attribute matters to the
    flattener. Avoids the version-dependent ``BaseExceptionGroup``
    constructor on the 3.10 floor.
    """

    def __init__(self, *exceptions: BaseException) -> None:
        super().__init__("fake taskgroup failure")
        self.exceptions = list(exceptions)


class TestSkipIfUnreachable:
    """Test skip_if_unreachable branch behavior."""

    def test_network_leaf_group_skips(self):
        """A group of only network leaves skips with the stable prefix."""
        group = _FakeGroupError(httpx.ConnectError("All connection attempts failed"))

        with pytest.raises(pytest.skip.Exception) as excinfo:
            skip_if_unreachable(group, "helper unit test")

        assert str(excinfo.value).startswith("network-unavailable:")
        assert "helper unit test" in str(excinfo.value)
        assert "ConnectError" in str(excinfo.value)

    def test_non_network_leaf_reraises(self):
        """A non-network exception re-raises instead of skipping."""
        bug = ValueError("real bug hidden in the taskgroup")

        with pytest.raises(ValueError, match="real bug"):
            skip_if_unreachable(bug, "helper unit test")

    def test_mixed_group_reraises_original(self):
        """One network leaf plus one real bug re-raises the original group."""
        group = _FakeGroupError(
            httpx.ConnectError("All connection attempts failed"),
            ValueError("real bug alongside the network failure"),
        )

        with pytest.raises(_FakeGroupError) as excinfo:
            skip_if_unreachable(group, "helper unit test")

        # The ORIGINAL exception object must propagate (all-leaves rule).
        assert excinfo.value is group
