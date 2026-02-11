"""Domain-specific BoundaryPredicate implementations.

Components:
    PositionListBoundary — boundary at EM-learned token positions
    SMILESBoundary       — boundary at SMILES atom/group completions
    SQLClauseBoundary    — boundary at SQL clause keywords
"""

from genlm.control.sampler.unit import BoundaryPredicate, flatten_units


class PositionListBoundary(BoundaryPredicate):
    """Boundary predicate triggering at specified cumulative token positions.

    Used for EM-learned boundaries: the EM loop discovers optimal positions,
    and this predicate enforces them during Block SMC.

    Args:
        positions: Sorted list of cumulative token positions where boundaries occur.
            E.g., [5, 12, 20] means boundaries after the 5th, 12th, and 20th token.
    """

    def __init__(self, positions: list[int]):
        if not positions:
            raise ValueError("positions must be non-empty")
        self.positions = sorted(positions)
        self._position_set = set(self.positions)

    def __call__(self, unit_context: list, subunit_buffer: list) -> bool:
        """Check if the cumulative token count has reached a boundary position."""
        # Count tokens already in completed units
        prev_tokens = sum(
            len(u) if isinstance(u, list) else 1 for u in unit_context
        )
        # Current position = previous tokens + tokens in current buffer
        current_pos = prev_tokens + len(subunit_buffer)
        return current_pos in self._position_set

    def __repr__(self) -> str:
        return f"PositionListBoundary({self.positions})"


class FixedIntervalBoundary(BoundaryPredicate):
    """Boundary predicate triggering every K tokens.

    Simple baseline: evenly-spaced boundaries.

    Args:
        interval: Number of tokens per block.
    """

    def __init__(self, interval: int):
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        self.interval = interval

    def __call__(self, unit_context: list, subunit_buffer: list) -> bool:
        return len(subunit_buffer) >= self.interval

    def __repr__(self) -> str:
        return f"FixedIntervalBoundary({self.interval})"


class SMILESBoundary(BoundaryPredicate):
    """Boundary predicate for SMILES molecular strings.

    Triggers at atom/group completion boundaries in SMILES notation.
    A boundary fires when the buffer ends with a complete atom or group:
    - After closing brackets: ']'
    - After closing parentheses: ')'
    - After organic atoms followed by bond/digit/branch/atom: when the next
      character would start a new atom

    For simplicity, we detect boundaries at structural delimiters:
    closing brackets, parentheses, and ring-closure digits.

    Args:
        min_tokens: Minimum tokens per block. Default 3.
    """

    # Characters that mark the end of a SMILES structural unit
    BOUNDARY_CHARS = {b"]", b")", b".", b"/", b"\\"}
    # Ring closure digits/% markers
    RING_CHARS = {str(i).encode() for i in range(10)}

    def __init__(self, min_tokens: int = 3):
        self.min_tokens = min_tokens

    def __call__(self, unit_context: list, subunit_buffer: list) -> bool:
        if len(subunit_buffer) < self.min_tokens:
            return False
        last = subunit_buffer[-1]
        if not isinstance(last, bytes):
            return False
        return last in self.BOUNDARY_CHARS or last in self.RING_CHARS

    def __repr__(self) -> str:
        return f"SMILESBoundary(min_tokens={self.min_tokens})"


class SQLClauseBoundary(BoundaryPredicate):
    """Boundary predicate for SQL queries.

    Triggers when the token buffer contains a SQL clause keyword,
    indicating the start of a new clause (and thus the end of the previous block).

    Keywords: SELECT, FROM, WHERE, GROUP BY, ORDER BY, HAVING, LIMIT, JOIN, ON

    Args:
        min_tokens: Minimum tokens per block before checking. Default 2.
    """

    CLAUSE_KEYWORDS = {
        b"select", b"SELECT",
        b"from", b"FROM",
        b"where", b"WHERE",
        b"group", b"GROUP",
        b"order", b"ORDER",
        b"having", b"HAVING",
        b"limit", b"LIMIT",
        b"join", b"JOIN",
        b"on", b"ON",
    }

    def __init__(self, min_tokens: int = 2):
        self.min_tokens = min_tokens

    def __call__(self, unit_context: list, subunit_buffer: list) -> bool:
        if len(subunit_buffer) < self.min_tokens:
            return False
        # Check if the last token starts a new clause keyword
        last = subunit_buffer[-1]
        if not isinstance(last, bytes):
            return False
        # Boundary fires when we see a clause keyword — the keyword
        # starts a new block, so the boundary is BEFORE it.
        # We use a look-back: boundary fires when the accumulated text
        # contains a keyword at a word boundary.
        text = b"".join(
            t for t in subunit_buffer if isinstance(t, bytes)
        ).strip()
        text_upper = text.upper()
        for kw in [b"SELECT", b"FROM", b"WHERE", b"GROUP BY", b"ORDER BY",
                    b"HAVING", b"LIMIT", b"JOIN", b"ON"]:
            # Check if text ends with the keyword (new clause starting)
            if text_upper.endswith(kw) and len(text) > len(kw):
                return True
        return False

    def finalize_unit(self, subunit_buffer: list) -> list:
        """Remove the clause keyword from the end of the block.

        The keyword that triggered the boundary belongs to the NEXT block.
        """
        # Find how many trailing tokens form the keyword
        # For simplicity, keep the full buffer (keyword included in this block)
        # TODO: split the keyword token to the next block if needed
        return subunit_buffer

    def __repr__(self) -> str:
        return f"SQLClauseBoundary(min_tokens={self.min_tokens})"
