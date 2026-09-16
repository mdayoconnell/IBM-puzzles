"""Weighted node scoring for Jordan/ruler move sequences."""

from __future__ import annotations

from jordanchain import compute_jordan_chain


def move_score(move: int, width: int) -> int:
    """Sum one-based node labels selected by ``move``.

    Node 1 is the most-significant bit and node ``width`` is the
    least-significant bit.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    if not 0 <= move < (1 << width):
        raise ValueError(f"move must fit in {width} bits")

    score = 0
    remaining = move
    while remaining:
        low_bit = remaining & -remaining
        score += width - (low_bit.bit_length() - 1)
        remaining ^= low_bit
    return score


def jordan_total_score(width: int) -> int:
    """Return the score of all ``2**width - 1`` ruler-function moves."""
    chain = compute_jordan_chain(width)
    return sum(
        move_score(move, width) << (width - level - 1)
        for level, move in enumerate(chain)
    )
