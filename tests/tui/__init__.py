"""Test package for the dnallm.tui wizard.

The ``__init__.py`` marker is load-bearing, not stylistic: ``tests/preflight/``
also has a ``test_structure.py``, and pytest's default import mode cannot
collect two same-named modules that are not packages (it raises an
``import file mismatch`` collection error when both are requested in one run).
Making this a package gives its modules a distinct dotted name.
"""
