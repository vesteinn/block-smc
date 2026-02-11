"""Tests for boundary predicates."""

import pytest
from block_smc.boundary import (
    PositionListBoundary,
    FixedIntervalBoundary,
    SMILESBoundary,
    SQLClauseBoundary,
)


class TestPositionListBoundary:
    def test_triggers_at_correct_positions(self):
        boundary = PositionListBoundary([3, 7, 10])
        # Simulate unit_context = [] (no previous units), buffer growing
        tokens = [b"a", b"b", b"c"]
        assert not boundary([], tokens[:2])  # pos 2, not a boundary
        assert boundary([], tokens)  # pos 3, is a boundary

    def test_with_existing_units(self):
        boundary = PositionListBoundary([3, 7])
        # First unit was 3 tokens, now building second unit
        unit1 = [b"a", b"b", b"c"]
        buf = [b"d", b"e", b"f", b"g"]
        # prev_tokens=3, current_pos=3+4=7
        assert boundary([unit1], buf)

    def test_empty_positions_raises(self):
        with pytest.raises(ValueError):
            PositionListBoundary([])

    def test_sorts_positions(self):
        boundary = PositionListBoundary([10, 3, 7])
        assert boundary.positions == [3, 7, 10]


class TestFixedIntervalBoundary:
    def test_triggers_at_interval(self):
        boundary = FixedIntervalBoundary(5)
        assert not boundary([], [b"a"] * 4)
        assert boundary([], [b"a"] * 5)
        assert boundary([], [b"a"] * 6)

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            FixedIntervalBoundary(0)


class TestSMILESBoundary:
    def test_boundary_at_bracket_close(self):
        boundary = SMILESBoundary(min_tokens=1)
        assert boundary([], [b"[", b"N", b"H", b"]"])
        assert not boundary([], [b"[", b"N", b"H"])

    def test_boundary_at_ring_digit(self):
        boundary = SMILESBoundary(min_tokens=1)
        assert boundary([], [b"C", b"1"])

    def test_min_tokens_enforced(self):
        boundary = SMILESBoundary(min_tokens=3)
        assert not boundary([], [b"]"])  # only 1 token
        assert not boundary([], [b"a", b"]"])  # only 2 tokens
        assert boundary([], [b"a", b"b", b"]"])  # 3 tokens, ends with ]


class TestSQLClauseBoundary:
    def test_boundary_at_from_keyword(self):
        boundary = SQLClauseBoundary(min_tokens=2)
        buf = [b"s", b"e", b"l", b"e", b"c", b"t", b" ", b"*", b" ", b"F", b"R", b"O", b"M"]
        # Text ends with "FROM" and len > len("FROM")
        assert boundary([], buf)

    def test_no_boundary_at_start(self):
        boundary = SQLClauseBoundary(min_tokens=2)
        # Just "SELECT" alone shouldn't trigger (nothing before it)
        buf = [b"S", b"E", b"L", b"E", b"C", b"T"]
        # Text IS just "SELECT", len(text) == len("SELECT"), so len(text) > len(kw) is False
        assert not boundary([], buf)

    def test_min_tokens(self):
        boundary = SQLClauseBoundary(min_tokens=2)
        assert not boundary([], [b"F"])  # only 1 token
