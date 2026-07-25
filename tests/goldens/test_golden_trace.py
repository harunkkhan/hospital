"""Golden trace (doc 08 §4): the slice run recomputed must match byte-for-byte."""

from __future__ import annotations

import pytest
from _golden_helpers import GOLDENS_DIR, slice_trace_canonical


def test_er_slice_trace_matches_golden() -> None:
    expected = (GOLDENS_DIR / "er_slice_trace.json").read_text()
    actual = slice_trace_canonical()
    if actual != expected:
        # A plain assert would dump megabytes; fail with the procedure instead.
        pytest.fail(
            "er_slice_trace.json drifted from the recomputed slice run. "
            "This is a SEMANTIC change to sim behavior - review it, then "
            "regenerate via `uv run python tests/goldens/regenerate.py --only trace`"
        )
