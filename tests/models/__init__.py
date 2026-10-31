"""Test package for the ``dnallm.models`` tests.

The ``__init__.py`` marker is load-bearing, not stylistic: ``tests/utils/``
also has a ``test_support.py``, and pytest's default import mode cannot
collect two same-named modules that are not packages (it raises an
``import file mismatch`` collection error when both are requested in one run,
breaking whole-suite collection on CI). Making this a package gives its
modules a distinct dotted name (``models.test_support``).
"""
