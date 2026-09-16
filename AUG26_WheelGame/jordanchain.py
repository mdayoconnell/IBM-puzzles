"""Jordan-chain construction and the optimal ruler-function strategy.

All public state and move values are physical eight-bit integers.  Functions
whose names mention ``coordinate`` use the Jordan-basis coordinate integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from util import MAX_INT256, WIDTH, int256_to_str


ALL_ONES: Final = MAX_INT256
NUM_STATES: Final = 1 << WIDTH


def rotate_left(value: int, amount: int = 1, width: int = WIDTH) -> int:
    """Cyclically rotate a ``width``-bit integer to the left."""
    if width <= 0:
        raise ValueError("width must be positive")
    mask = (1 << width) - 1
    if not 0 <= value <= mask:
        raise ValueError(f"value must fit in {width} bits")
    amount %= width
    if amount == 0:
        return value
    return ((value << amount) | (value >> (width - amount))) & mask


def rotate_right(value: int, amount: int = 1, width: int = WIDTH) -> int:
    """Cyclically rotate a ``width``-bit integer to the right."""
    return rotate_left(value, -amount, width)


def nilpotent_step(value: int, width: int = WIDTH) -> int:
    """Apply N = R^-1 + I over GF(2), using one right rotation."""
    return rotate_right(value, 1, width) ^ value


def compute_jordan_chain(width: int = WIDTH) -> tuple[int, ...]:
    """Return ``(a_0, ..., a_(width-1))`` for a power-of-two wheel.

    The returned vectors obey ``N(a_k) == a_(k-1)`` for every ``k > 0``.
    """
    if width <= 0 or width & (width - 1):
        raise ValueError("the single-chain construction needs power-of-two width")

    # With node costs increasing from the most-significant bit to the least,
    # the cheapest cyclic seed is node 1 (the most-significant bit).  Iterating
    # right-rotation + identity gives the minimum-score Jordan-adapted basis.
    descending = [1 << (width - 1)]  # Start at a_(width-1).
    for _ in range(width - 1):
        descending.append(nilpotent_step(descending[-1], width))
    chain = tuple(reversed(descending))

    if len(set(chain)) != width or nilpotent_step(chain[0], width) != 0:
        raise AssertionError("failed to construct a full Jordan chain")
    return chain


JORDAN_CHAIN: Final = compute_jordan_chain()


def ruler(index: int) -> int:
    """Return v2(index), the exponent of its largest power-of-two divisor."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("ruler index must be an integer")
    if index <= 0:
        raise ValueError("ruler index must be positive")
    return (index & -index).bit_length() - 1


def jordan_move(index: int) -> int:
    """Map a one-based strategy index in 1..255 to its physical move."""
    if not 1 <= index <= MAX_INT256:
        raise ValueError("move index must be between 1 and 255")
    return JORDAN_CHAIN[ruler(index)]


def move_sequence(count: int = MAX_INT256) -> tuple[int, ...]:
    """Return the first ``count`` moves of the optimal 255-move sequence."""
    if not 0 <= count <= MAX_INT256:
        raise ValueError("count must be between 0 and 255")
    return tuple(jordan_move(index) for index in range(1, count + 1))


def jordan_to_physical(coordinate: int) -> int:
    """Expand an eight-bit Jordan coordinate in the physical bit basis."""
    if not 0 <= coordinate <= MAX_INT256:
        raise ValueError("coordinate must be between 0 and 255")
    physical = 0
    for bit, basis_vector in enumerate(JORDAN_CHAIN):
        if coordinate & (1 << bit):
            physical ^= basis_vector
    return physical


@lru_cache(maxsize=1)
def _physical_to_jordan_table() -> tuple[int, ...]:
    table = [-1] * NUM_STATES
    for coordinate in range(NUM_STATES):
        table[jordan_to_physical(coordinate)] = coordinate
    if any(coordinate < 0 for coordinate in table):
        raise AssertionError("Jordan vectors do not form a basis")
    return tuple(table)


def physical_to_jordan(physical: int) -> int:
    """Convert a physical state/move to its Jordan coordinate integer."""
    if not 0 <= physical <= MAX_INT256:
        raise ValueError("physical value must be between 0 and 255")
    return _physical_to_jordan_table()[physical]


def coordinate_belief(size: int) -> frozenset[int]:
    """Return F_size, the canonical rotation-invariant belief set.

    C_0 is {0}; for k > 0, C_k is [2**k, 2**(k+1)).  F_size is
    the union of C_k for the set bits k of ``size`` and has exactly ``size``
    elements.
    """
    if not 0 <= size <= MAX_INT256:
        raise ValueError("belief size must be between 0 and 255")
    result: set[int] = set()
    for level in range(WIDTH):
        if not size & (1 << level):
            continue
        if level == 0:
            result.add(0)
        else:
            result.update(range(1 << level, 1 << (level + 1)))
    return frozenset(result)


def physical_belief(size: int) -> frozenset[int]:
    """Return F_size converted from Jordan coordinates to physical states."""
    return frozenset(jordan_to_physical(c) for c in coordinate_belief(size))


def rotation_closure(states: frozenset[int] | set[int]) -> frozenset[int]:
    """Close physical states under all eight cyclic rotations."""
    return frozenset(
        rotate_left(state, amount)
        for state in states
        for amount in range(WIDTH)
    )


@dataclass(frozen=True)
class Transition:
    """Auditable data for one non-winning belief transition."""

    index: int
    move: int
    ruler_level: int
    before: frozenset[int]
    killed_pre_move: int
    survivors_before_rotation: frozenset[int]
    after_rotation: frozenset[int]
    expected_after: frozenset[int]

    @property
    def rotation_added(self) -> frozenset[int]:
        return self.after_rotation - self.survivors_before_rotation

    @property
    def is_exact(self) -> bool:
        return (
            len(self.before) - len(self.after_rotation) == 1
            and not self.rotation_added
            and self.after_rotation == self.expected_after
        )


def strategy_transition(index: int) -> Transition:
    """Compute move ``index`` from F_(256-index) to F_(255-index)."""
    if not 1 <= index <= MAX_INT256:
        raise ValueError("move index must be between 1 and 255")

    before_size = NUM_STATES - index
    before = physical_belief(before_size)
    move = jordan_move(index)
    killed = ALL_ONES ^ move
    survivors = frozenset(
        state ^ move for state in before if (state ^ move) != ALL_ONES
    )
    closed = rotation_closure(survivors)
    expected = physical_belief(before_size - 1)
    return Transition(
        index=index,
        move=move,
        ruler_level=ruler(index),
        before=before,
        killed_pre_move=killed,
        survivors_before_rotation=survivors,
        after_rotation=closed,
        expected_after=expected,
    )


def verify_strategy() -> bool:
    """Verify all 255 exact transitions, including rotation closure."""
    return all(strategy_transition(index).is_exact for index in range(1, 256))


def _print_summary() -> None:
    print("Jordan chain (a_0 .. a_7):")
    for index, vector in enumerate(JORDAN_CHAIN):
        print(f"a_{index} = {int256_to_str(vector)} ({vector})")
    print(f"\nAll 255 transitions verified: {verify_strategy()}")


if __name__ == "__main__":
    _print_summary()
