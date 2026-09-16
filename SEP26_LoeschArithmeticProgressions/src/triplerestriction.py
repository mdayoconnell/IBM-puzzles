"""Loesch APs: forced differences, primitive starts, and CRT.

By default, exact mode finds the minimum endpoint within the inclusive limit.
Dividing out the common gcd preserves Loesch membership, so this minimum
also applies to unrestricted APs. Lengths must be at least four.
With --min-needed false, stop at the first AP for each length instead.

The CRT class budget controls table size, NOT search coverage: any prime
that does not fit the wheel is checked on the generated starts instead.
Record mode optionally restricts differences to extra multiples. Result mode
does the same but always stops at the first AP for each target length. Its results
certify only that restricted family, never a global minimum or impossibility.

Uses the standard-library membership generator in loesch_explore.py.
"""

import argparse
import csv
import json
from functools import lru_cache
from math import gcd, isqrt, lcm
from pathlib import Path
from time import perf_counter

from loesch_explore import forced_step_multiple, loesch_flags, primes_through


@lru_cache(maxsize=256)
def allowed_residues(length, prime):
    """Allowed t mod p² for a = -d*t, when p does not divide d.

Reject valuation exactly one at any term. Higher odd valuations are left
to the final membership check. For p > length, retain the no-hit classes
as well as classes whose unique multiple of p is a multiple of p².
"""
    if length >= 2 * prime:
        return ()
    if length >= prime:
        return tuple(range(length - prime, prime))
    modulus = prime * prime
    allowed = bytearray(b"\x01") * modulus
    for i in range(length):
        for multiple in range(1, prime):
            allowed[(i + multiple * prime) % modulus] = 0
    return tuple(i for i, value in enumerate(allowed) if value)


@lru_cache(maxsize=128)
def crt_wheel(length, primes):
    """Combine all allowed normalized residues; never truncate classes."""
    modulus, residues = 1, (0,)
    for prime in primes:
        square = prime * prime
        inverse = pow(modulus, -1, square)
        residues = tuple(
            old + modulus * ((new - old) * inverse % square)
            for old in residues for new in allowed_residues(length, prime)
        )
        modulus *= square
    return modulus, residues


def crt_plan(length, difference, max_start, primes, class_budget):
    """Build a bounded wheel and exact filters for its remaining primes."""
    selected, remaining = [], []
    modulus, classes = 1, 1
    for prime in primes:
        if difference % prime == 0:
            # Primitive gcd(a,d)=1 already guarantees p divides no term.
            continue
        count = len(allowed_residues(length, prime))
        if classes * count <= class_budget and 6 * modulus <= max_start:
            selected.append(prime)
            classes *= count
            modulus *= prime * prime
        else:
            square = prime * prime
            remaining.append((square, (-pow(difference, -1, square)) % square,
                              frozenset(allowed_residues(length, prime))))
    modulus, residues = crt_wheel(length, tuple(selected))
    # Merge a = -d*t mod modulus with a = 1 mod 6.
    starts = []
    inverse_six = pow(modulus, -1, 6)
    for residue in residues:
        a = (-difference * residue) % modulus
        starts.append(a + modulus * ((1 - a) * inverse_six % 6))
    return 6 * modulus, sorted(starts), remaining


def search_length(flags, length, *, min_needed=True, mode="exact", extra_step_multiple=1,
                  crt_prime_limit=None, class_budget=4096, progress=None,
                  progress_seconds=10):
    """Return (end,start,difference) and counters, minimizing unless disabled."""
    if length < 4 or class_budget < 1 or extra_step_multiple < 1:
        raise ValueError("length >= 4, class_budget >= 1, extra_step_multiple >= 1 required")
    if mode not in ("exact", "record", "result"):
        raise ValueError("mode must be exact, record, or result")
    if mode == "result":
        min_needed = False
    if mode == "exact" and extra_step_multiple != 1:
        raise ValueError("extra difference restrictions require record or result mode")
    limit = len(flags) - 1
    forced = forced_step_multiple(length)
    step = lcm(forced, extra_step_multiple)
    prime_limit = length if crt_prime_limit is None else crt_prime_limit
    primes = [p for p in primes_through(prime_limit)
              if p % 3 == 2 and p > length // 2]
    primes.sort(key=lambda p: len(allowed_residues(length, p)) / (p * p))
    maximum_difference = (limit - 1) // (length - 1)
    stats = dict(length=length, limit=limit, mode=mode, min_needed=min_needed,
                 forced_step_multiple=forced, searched_step_multiple=step,
                 crt_primes=primes, crt_class_budget=class_budget,
                 differences_available=max(0, maximum_difference // step),
                 differences_examined=0, wheel_starts=0, crt_starts=0,
                 eligible_starts=0)
    best = None
    started = last_progress = perf_counter()
    for difference in range(step, maximum_difference + 1, step):
        endpoint_bound = limit if best is None else best[0]
        span = (length - 1) * difference
        max_start = endpoint_bound - span
        if max_start < 1:
            break  # All later differences have strictly larger endpoints.
        stats["differences_examined"] += 1
        period, residues, filters = crt_plan(
            length, difference, max_start, primes, class_budget)
        for residue in residues:
            # CRT includes 1 mod 6, so residue is strictly positive.
            stop = max_start if best is None else min(max_start, best[0] - span)
            for start in range(residue, stop + 1, period):
                stats["wheel_starts"] += 1
                if any((start * inverse) % square not in allowed
                       for square, inverse, allowed in filters):
                    continue
                stats["crt_starts"] += 1
                if not flags[start] or gcd(start, difference) != 1:
                    continue
                stats["eligible_starts"] += 1
                end = start + span
                if not flags[end]:
                    continue
                if all(flags[term] for term in range(start + difference, end, difference)):
                    candidate = (end, start, difference)
                    if best is None or candidate < best:
                        best = candidate
                        if progress:
                            action = "checking minimum" if min_needed else "stopping at first AP"
                            progress(f"FOUND length={length} start={start} difference={difference} "
                                     f"endpoint={end} ({action})")
                    break  # Later starts in this class cannot improve this hit.
            if best is not None and not min_needed:
                break
        if best is not None and not min_needed:
            break
        now = perf_counter()
        if progress and now - last_progress >= progress_seconds:
            progress(f"SEARCHING length={length} differences={stats['differences_examined']}/"
                     f"{stats['differences_available']} eligible_starts={stats['eligible_starts']} "
                     f"elapsed={now - started:.3f}s")
            last_progress = now
    ap = None if best is None else dict(endpoint=best[0], start=best[1], difference=best[2])
    if best is None:
        status = "EXHAUSTED_NO_AP" if mode == "exact" else "RESTRICTED_SEARCH_EXHAUSTED"
    else:
        status = "MINIMUM_FOUND" if min_needed else "AP_FOUND"
        if mode != "exact":
            status = "RESTRICTED_" + status
    stats.update(search_seconds=perf_counter() - started,
                 no_allowable_differences=stats["differences_available"] == 0,
                 no_crt_starting_points=stats["crt_starts"] == 0,
                 no_starting_points=stats["eligible_starts"] == 0,
                 ap=ap, minimum=ap if min_needed else None,
                 global_minimum_certified=best is not None and mode == "exact" and min_needed,
                 exhaustive_impossibility_below_limit=best is None and mode == "exact",
                 status=status)
    return best, stats


def representation(value):
    """Construct an independent integer norm witness for an output term."""
    for x in range(isqrt(value // 3) + 1):
        discriminant = 4 * value - 3 * x * x
        root = isqrt(discriminant)
        if root * root == discriminant and (root - x) % 2 == 0:
            y = (root - x) // 2
            if y >= x and x * x + x * y + y * y == value:
                return x, y
    raise ValueError(f"No Loesch witness for {value}")


def parse_bool(value):
    """Parse explicit CLI booleans without treating the string 'false' as true."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("expected true or false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1_000_000_000, help="inclusive endpoint bound")
    parser.add_argument("--start-length", type=int, default=32, help="first target length (default: 32)")
    parser.add_argument("--max-length", type=int, help="last target; default: continue until the first failure")
    parser.add_argument("--min-needed", "--min_needed", type=parse_bool, default=True,
                        metavar="{true,false}",
                        help="minimize each endpoint (default: true); false stops at the first AP per length")
    parser.add_argument("--mode", choices=("exact", "record", "result"), default="exact",
                        help="result allows restricted differences and always stops at the first AP per length")
    parser.add_argument("--extra-step-multiple", type=int, default=1,
                        help="require this additional difference divisor; record or result mode only")
    parser.add_argument("--crt-prime-limit", type=int,
                        help="use inert primes through this value; default: target length")
    parser.add_argument("--crt-class-budget", type=int, default=4096,
                        help="maximum combined CRT classes; remaining primes are postfilters")
    parser.add_argument("--progress-seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, default=Path("triplerestriction_aps.csv"))
    parser.add_argument("--stats-output", type=Path, default=Path("triplerestriction_stats.jsonl"))
    args = parser.parse_args()
    if args.limit < 1 or args.start_length < 4:
        parser.error("--limit must be positive and --start-length at least 4")
    if args.max_length is not None and args.max_length < args.start_length:
        parser.error("--max-length must be at least --start-length")
    if args.extra_step_multiple < 1 or args.crt_class_budget < 1 or args.progress_seconds <= 0:
        parser.error("step multiple, class budget, and progress interval must be positive")
    if args.crt_prime_limit is not None and args.crt_prime_limit < 2:
        parser.error("--crt-prime-limit must be at least 2")
    if args.mode == "exact" and args.extra_step_multiple != 1:
        parser.error("--extra-step-multiple requires --mode record or result")
    if args.mode == "result":
        args.min_needed = False
    if args.output.resolve() == args.stats_output.resolve():
        parser.error("--output and --stats-output must be different files")

    def report(message):
        print(message, flush=True)

    started = perf_counter()
    report(f"Generating Loesch membership through {args.limit:,}; "
           f"mode={args.mode}, min_needed={args.min_needed}.")
    try:
        flags = loesch_flags(args.limit)
    except KeyboardInterrupt:
        report("\nInterrupted during membership generation; output files were not changed.")
        return 130
    generation_seconds = perf_counter() - started
    report(f"Generation time: {generation_seconds:.3f}s")
    if args.mode != "exact":
        report("RESTRICTED SEARCH: results apply only to the selected primitive difference family.")
    interrupted = False
    rows, stats_lines = [], []
    with args.output.open("w", newline="") as output, args.stats_output.open("w") as stats_output:
        writer = csv.writer(output)
        header = ["length", "start", "gap", "numbers", "xy_pairs"]

        def save_stats(stats):
            line = json.dumps(stats) + "\n"
            stats_lines.append(line)
            stats_output.write(line)
            stats_output.flush()

        length = args.start_length
        try:
            writer.writerow(header)
            output.flush()
            save_stats(dict(event="precompute", limit=args.limit,
                            generation_seconds=generation_seconds))
            while args.max_length is None or length <= args.max_length:
                report(f"SEARCHING length={length} endpoint_limit={args.limit:,}")
                best, stats = search_length(
                    flags, length, min_needed=args.min_needed, mode=args.mode,
                    extra_step_multiple=args.extra_step_multiple,
                    crt_prime_limit=args.crt_prime_limit, class_budget=args.crt_class_budget,
                    progress=report, progress_seconds=args.progress_seconds)
                if best is not None:
                    end, start, difference = best
                    terms = list(range(start, end + 1, difference))
                    row = [length, start, difference, terms,
                           [representation(term) for term in terms]]
                    rows.append(row)
                    writer.writerow(row)
                    output.flush()
                    report(f"{stats['status']} length={length} start={start} difference={difference} "
                           f"endpoint={end} search_time={stats['search_seconds']:.3f}s")
                else:
                    for key in ("no_allowable_differences", "no_crt_starting_points", "no_starting_points"):
                        if stats[key]:
                            report(f"{key.upper()} length={length}")
                    report(f"{stats['status']} length={length} endpoint_limit={args.limit:,} "
                           f"search_time={stats['search_seconds']:.3f}s")
                    if args.mode == "exact":
                        report(f"Exhaustively impossible at length {length} or longer with endpoint <= {args.limit:,}.")
                    else:
                        report("No AP in the selected family; this does not rule out APs with other differences.")
                save_stats(stats)
                report(f"Candidates: wheel={stats['wheel_starts']:,}, CRT={stats['crt_starts']:,}, "
                       f"Loesch and coprime={stats['eligible_starts']:,}")
                if best is None:
                    break
                length += 1
            total_seconds = perf_counter() - started
            save_stats(dict(event="complete", total_seconds=total_seconds))
        except KeyboardInterrupt:
            interrupted = True
            total_seconds = perf_counter() - started
            # Repair either file if the interrupt arrived during a write.
            output.seek(0)
            output.truncate()
            writer.writerow(header)
            writer.writerows(rows)
            output.flush()
            stats_output.seek(0)
            stats_output.truncate()
            stats_output.writelines(stats_lines)
            stats_output.flush()
            save_stats(dict(event="interrupted", length=length, mode=args.mode, min_needed=args.min_needed,
                            completed_lengths=[row[0] for row in rows],
                            total_seconds=total_seconds))
    if interrupted:
        report(f"\nInterrupted; saved {len(rows)} completed AP records. "
               "Unfinished searches are not certified minima.")
    report(f"Total runtime: {total_seconds:.3f}s. APs: {args.output}; statistics: {args.stats_output}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
