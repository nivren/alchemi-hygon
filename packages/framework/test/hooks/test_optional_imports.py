"""Tests for optional hook dependencies at package import boundaries."""

from __future__ import annotations

import subprocess
import sys


def test_hook_packages_do_not_eagerly_import_physicsnemo() -> None:
    script = """
import sys
import nvalchemi.hooks
import nvalchemi.training.hooks
assert 'physicsnemo' not in sys.modules
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
