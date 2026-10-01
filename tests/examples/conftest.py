"""Locally scoped fixtures for the example execution tests.

Only sandbox/kernel fixtures live here; the shared mock fixtures stay
in ``tests/conftest.py`` and remain usable from this directory through
normal pytest conftest inheritance.  This file must never merge into
the root conftest, which ~1700 tests share.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.examples._execution import (
    NOTEBOOK_EXEC_SPECS,
    assert_tree_clean,
    seed_sandbox,
)


@pytest.fixture
def notebook_sandbox(tmp_path: Path) -> Iterator[Path]:
    """Return a seeded tmp sandbox of the executed notebook's directory.

    Seeds the sole :data:`NOTEBOOK_EXEC_SPECS` entry's parent directory
    (the pilot notebook's dir) over ``tmp_path`` and asserts the repo
    tree stayed clean on teardown -- belt-and-braces beyond the
    in-test guard.  Phase 8 generalizes this fixture over the expanded
    spec dict.
    """
    spec_paths = list(NOTEBOOK_EXEC_SPECS)
    assert len(spec_paths) == 1, "single pilot spec expected; generalize fixture before expanding"
    pilot_dir = Path(spec_paths[0]).parent
    yield seed_sandbox(pilot_dir, tmp_path)
    assert_tree_clean()
