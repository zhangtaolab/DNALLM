"""Typed network-skip helper for the live-server MCP client tests.

The MCP clients fail through httpx inside an anyio TaskGroup, so the
catchable exception is an ``ExceptionGroup`` wrapping an
``httpx.ConnectError`` (MRO: ConnectError -> NetworkError ->
TransportError). This module flattens the group tree and skips a test
only when every leaf is a network-level transport failure; any other
leaf re-raises so the test fails honestly.
"""

import httpx
import pytest

# Network-level failures only. HTTPStatusError (a server that ANSWERED
# with an error status) is deliberately excluded: that is a real test
# failure, not "no server".
NETWORK_ERRORS = (httpx.TransportError,)


def _network_leaves(exc: BaseException) -> list[BaseException]:
    """Flatten an ExceptionGroup tree to its leaf exceptions.

    Duck-typed via the ``exceptions`` attribute so the helper works on
    Python 3.10 (where the ``exceptiongroup`` backport installed by
    anyio provides the same attribute) without importing
    ``BaseExceptionGroup``, which only exists on 3.11+.

    Args:
        exc: the exception to flatten.

    Returns:
        The leaf exceptions of the group tree, or ``[exc]`` itself when
        it carries no ``exceptions`` attribute.
    """
    if hasattr(exc, "exceptions"):
        out: list[BaseException] = []
        for sub in exc.exceptions:
            out.extend(_network_leaves(sub))
        return out
    return [exc]


def skip_if_unreachable(exc: BaseException, action: str) -> None:
    """Skip only when every leaf cause is a network-level failure.

    The skip message carries the stable ``network-unavailable:`` prefix
    followed by *action* and the first leaf's type name, so the junit
    ``<skipped message>`` stays deterministic for allowlist matching.

    Args:
        exc: the caught exception (possibly an ExceptionGroup).
        action: stable label naming what the test was doing.

    Raises:
        BaseException: the original *exc*, re-raised when any leaf is
            not a network error (an honest test failure, never a skip).
    """
    leaves = _network_leaves(exc)
    if leaves and all(isinstance(leaf, NETWORK_ERRORS) for leaf in leaves):
        message = f"network-unavailable: {action} (no server reachable: {type(leaves[0]).__name__})"
        pytest.skip(message)
    raise exc
