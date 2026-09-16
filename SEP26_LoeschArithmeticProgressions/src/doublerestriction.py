"""Find minimum-endpoint Loesch APs using difference and start restrictions.

Preserves the endpoint-based search and CSV format of restrict_differences.py.
Only primitive progressions need be considered: dividing a Loesch AP by the
gcd of its terms preserves membership and decreases its endpoint. Primitive
APs of length at least four have every term congruent to 1 modulo 6.
"""

import argparse
import csv
from math import gcd, isqrt
from pathlib import Path
from time import perf_counter


def allowable_difference_multiple(length):
    """Return the necessary step divisor for a progression of this length."""
    multiple = 3 if length >= 3 else 1
    for p in range(2, length // 2 + 1):
        if p % 3 == 2 and all(p % q for q in range(2, isqrt(p) + 1)):
            multiple *= p
    return multiple


def find_loeschian_aps(limit):
    """Return (length, start, gap, terms, pairs) for each attainable length.

    Minimize the final term, then the start. Each term has one representation
    with 0 <= x <= y, choosing the smallest x. The singleton has gap zero.
    These filters preserve minimum-endpoint records, not every scaled AP.
    """
    if limit < 0:
        raise ValueError("limit must be nonnegative")

    # Keep the same norm generation and representation choices.
    representations = {}
    for x in range(isqrt(limit // 3) + 1):
        x_squared = x * x
        max_y = (isqrt(4 * limit - 3 * x_squared) - x) // 2
        for y in range(x, max_y + 1):
            value = x_squared + x * y + y * y
            representations.setdefault(value, (x, y))

    rows = [(1, 0, 0, [0], [representations[0]])]
    length = 2
    multiple = allowable_difference_multiple(length)
    for end in sorted(representations):
        # For length >= 4, gap is divisible by 6, so start == end (mod 6).
        # Reject the entire endpoint before trying any candidate gaps.
        if length >= 4 and end % 6 != 1:
            continue

        # Since multiple divides gap, this is a cheap endpoint-level part
        # of gcd(start, gap) == gcd(end, gap) == 1. It also covers mod 3
        # for the length-3 search, while preserving lengths 1 and 2.
        if gcd(end, multiple) != 1:
            continue

        # Only the next missing length can first occur at this endpoint.
        max_gap = end // (length - 1)
        max_gap -= max_gap % multiple
        for gap in range(max_gap, 0, -multiple):
            # Descending gaps preserve the smallest-start tie-breaker.
            start = end - (length - 1) * gap

            # Check the start before walking backward through the interior.
            if start not in representations:
                continue
            if gcd(start, gap) != 1:
                continue

            term = end - gap
            while term >= start and term in representations:
                term -= gap
            if term < start:
                terms = list(range(start, end + 1, gap))
                pairs = [representations[value] for value in terms]
                rows.append((length, start, gap, terms, pairs))
                length += 1
                multiple = allowable_difference_multiple(length)
                break

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100000000, help="largest allowed term")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("doublerestriction_aps.csv"),
        help="output CSV path (list columns use Python literal syntax)",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    started = perf_counter()
    rows = find_loeschian_aps(args.limit)
    elapsed = perf_counter() - started
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["length", "start", "gap", "numbers", "xy_pairs"])
        writer.writerows(rows)

    length, _, _, sequence, _ = rows[-1]
    print(f"Double-restricted AP Length: {length}")
    print(f"Sequence (Ending at {sequence[-1]}): {sequence}")
    print(f"Generation and search time: {elapsed:.6f} s")
    print(f"Saved AP lengths 1 through {length} to {args.output}")


if __name__ == "__main__":
    main()
