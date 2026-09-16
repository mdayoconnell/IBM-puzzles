"""Conversions for the eight-bit wheel.

Python does not have an ``int256`` scalar type.  In this project the name means
an ordinary Python ``int`` whose value is one of the 256 values 0 through 255.
Bit strings are always written most-significant bit first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
WIDTH = 8
MIN_INT256 = 0
MAX_INT256 = (1 << WIDTH) - 1

BoolVector = tuple[bool, ...]


def _check_int256(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("int256 values must be integers")
    if not MIN_INT256 <= value <= MAX_INT256:
        raise ValueError("int256 values must be between 0 and 255")
    return value


def _check_bitstr(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("bit string must be a str")
    if len(value) != WIDTH or set(value) - {"0", "1"}:
        raise ValueError("bit string must contain exactly eight 0/1 characters")
    return value


def _check_byte(value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError("byte value must be bytes")
    if len(value) != 1:
        raise ValueError("byte value must contain exactly one byte")
    return value


def _check_bools(value: Iterable[bool]) -> BoolVector:
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError("boolean vector must be iterable") from exc
    if len(result) != WIDTH or any(type(bit) is not bool for bit in result):
        raise ValueError("boolean vector must contain exactly eight bool values")
    return result


def int256_to_str(value: int) -> str:
    """Convert 0..255 to an eight-character binary string."""
    return format(_check_int256(value), "08b")


def str_to_int256(value: str) -> int:
    """Convert an eight-character binary string to an integer in 0..255."""
    return int(_check_bitstr(value), 2)


def int256_to_bytes(value: int) -> bytes:
    """Convert 0..255 to a single raw byte."""
    return bytes((_check_int256(value),))


def bytes_to_int256(value: bytes) -> int:
    """Convert one raw byte to an integer in 0..255."""
    return _check_byte(value)[0]


def int256_to_bools(value: int) -> BoolVector:
    """Convert 0..255 to eight booleans, most-significant bit first."""
    return tuple(character == "1" for character in int256_to_str(value))


def bools_to_int256(value: Iterable[bool]) -> int:
    """Pack eight booleans, most-significant bit first, into 0..255."""
    result = 0
    for bit in _check_bools(value):
        result = (result << 1) | bit
    return result


def str_to_bytes(value: str) -> bytes:
    return int256_to_bytes(str_to_int256(value))


def bytes_to_str(value: bytes) -> str:
    return int256_to_str(bytes_to_int256(value))


def str_to_bools(value: str) -> BoolVector:
    return int256_to_bools(str_to_int256(value))


def bools_to_str(value: Iterable[bool]) -> str:
    return int256_to_str(bools_to_int256(value))


def bytes_to_bools(value: bytes) -> BoolVector:
    return int256_to_bools(bytes_to_int256(value))


def bools_to_bytes(value: Iterable[bool]) -> bytes:
    return int256_to_bytes(bools_to_int256(value))


def xor_bools(left: Sequence[bool], right: Sequence[bool]) -> BoolVector:
    """Apply a wheel move by XORing two eight-boolean vectors."""
    checked_left = _check_bools(left)
    checked_right = _check_bools(right)
    return tuple(a ^ b for a, b in zip(checked_left, checked_right))


__all__ = [
    "BoolVector",
    "MAX_INT256",
    "WIDTH",
    "bools_to_bytes",
    "bools_to_int256",
    "bools_to_str",
    "bytes_to_bools",
    "bytes_to_int256",
    "bytes_to_str",
    "int256_to_bools",
    "int256_to_bytes",
    "int256_to_str",
    "str_to_bools",
    "str_to_bytes",
    "str_to_int256",
    "xor_bools",
]
