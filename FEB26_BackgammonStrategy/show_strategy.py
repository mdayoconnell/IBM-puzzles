from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Dict, Tuple, List, Optional
from itertools import product
import random

State = Tuple[int, ...]        # number of men per position
Action = Tuple[int, ...]       # action[p] = men moved from source position p at turn start


@dataclass(frozen=True)
class Outcome:
    name: str
    p: float
    die_uses: Tuple[int, ...]   # e.g. (1,2) or (2,2,2,2)
    k_min: int = 1
    k_max: int = 4


def _canonical_outcome_key(o: Outcome) -> Tuple[Tuple[int, ...], int, int]:
    # Canonicalize die order so 12 and 21 share cache
    return (tuple(sorted(o.die_uses)), o.k_min, o.k_max)


def validate_outcomes(outcomes: List[Outcome], n_men: int, p_max: int) -> None:
    if not outcomes:
        raise ValueError("outcomes must be non-empty.")

    prob_sum = sum(o.p for o in outcomes)
    if abs(prob_sum - 1.0) > 1e-12:
        raise ValueError(f"Outcome probabilities must sum to 1. Got {prob_sum}.")

    for o in outcomes:
        if o.p < 0:
            raise ValueError(f"Negative probability in outcome {o.name}.")
        if any(d < 0 for d in o.die_uses):
            raise ValueError(f"die_uses must be nonnegative in outcome {o.name}.")
        if o.k_min < 0 or o.k_max < 0:
            raise ValueError(f"k_min/k_max must be >= 0 in outcome {o.name}.")
        if o.k_min > o.k_max:
            raise ValueError(f"k_min > k_max in outcome {o.name}.")
        if o.k_max > n_men:
            raise ValueError(f"k_max cannot exceed n_men in outcome {o.name}.")

        if len(o.die_uses) == 0:
            if not (o.k_min == 0 and o.k_max == 0):
                raise ValueError(f"Empty die_uses requires k_min=k_max=0 in outcome {o.name}.")
        else:
            if o.k_max == 0:
                raise ValueError(f"Non-empty die_uses requires k_max>=1 in outcome {o.name}.")


def is_blot(state: State) -> bool:
    # occupied point with exactly one man
    return any(c == 1 for c in state if c > 0)


def _apply_turn_sequence(
    state: State,
    label_assignment: Tuple[int, ...],
    src_tuple: Tuple[int, ...],
    die_uses: Tuple[int, ...],
) -> Tuple[State, Action]:
    """
    Apply a turn sequentially with checker identities.

    - There are k checker labels 0..k-1.
    - src_tuple[label] gives that checker's starting source position.
    - label_assignment[j] tells which checker gets die_uses[j].

    Returns (next_state, action_out_counts_by_source_position).
    If move is infeasible, returns (tuple([-1]), tuple()).
    """
    p_max = len(state) - 1

    checker_pos = list(src_tuple)

    # Move assigned checker labels in die-use order
    for d, lbl in zip(die_uses, label_assignment):
        cur = checker_pos[lbl]
        nxt = cur + d
        if nxt > p_max:
            return tuple([-1]), tuple()
        checker_pos[lbl] = nxt

    # Reconstruct next state from untouched men + moved checkers
    arr = list(state)

    for src in src_tuple:
        arr[src] -= 1
        if arr[src] < 0:
            return tuple([-1]), tuple()

    for pos in checker_pos:
        arr[pos] += 1

    action = [0] * len(state)
    for src in src_tuple:
        action[src] += 1

    return tuple(arr), tuple(action)


def valid_turn_actions(state: State, outcome: Outcome) -> List[Tuple[Action, State]]:
    """
    Generate legal (action, next_state) for a realized outcome.
    """
    die_uses = outcome.die_uses
    m = len(die_uses)

    if m == 0:
        return [(tuple(0 for _ in state), state)]

    occupied_positions = [i for i, c in enumerate(state) if c > 0]
    if not occupied_positions:
        return []

    out: List[Tuple[Action, State]] = []
    seen = set()

    k_lo = max(1, outcome.k_min)
    k_hi = min(outcome.k_max, m, sum(state))
    if k_lo > k_hi:
        return []

    for k in range(k_lo, k_hi + 1):
        # assign each die-use to man label in {0..k-1}, surjectively
        for label_assignment in product(range(k), repeat=m):
            if len(set(label_assignment)) != k:
                continue

            # choose starting source position for each of the k men
            for src_tuple in product(occupied_positions, repeat=k):
                men_per_source: Dict[int, int] = {}
                feasible_sources = True
                for src in src_tuple:
                    men_per_source[src] = men_per_source.get(src, 0) + 1
                    if men_per_source[src] > state[src]:
                        feasible_sources = False
                        break
                if not feasible_sources:
                    continue

                nxt, a = _apply_turn_sequence(state, label_assignment, src_tuple, die_uses)
                if len(nxt) == 1 and nxt[0] == -1:
                    continue

                key = (a, nxt)
                if key in seen:
                    continue
                seen.add(key)
                out.append((a, nxt))

    return out


def make_turn_action_cache(outcomes: List[Outcome]):
    outcome_keys = [_canonical_outcome_key(o) for o in outcomes]
    key_to_rep_outcome: Dict[Tuple[Tuple[int, ...], int, int], Outcome] = {}

    for o, key in zip(outcomes, outcome_keys):
        if key not in key_to_rep_outcome:
            die_uses, k_min, k_max = key
            key_to_rep_outcome[key] = Outcome(
                name=f"cached:{o.name}",
                p=0.0,
                die_uses=die_uses,
                k_min=k_min,
                k_max=k_max,
            )

    @lru_cache(maxsize=None)
    def turn_actions_cached(state: State, outcome_idx: int) -> Tuple[Tuple[Action, State], ...]:
        key = outcome_keys[outcome_idx]
        rep = key_to_rep_outcome[key]
        return tuple(valid_turn_actions(state, rep))

    return turn_actions_cached


def make_dp_solver(
    n_men: int = 5,
    p_max: int = 20,
    outcomes: List[Outcome] = None,
):
    validate_outcomes(outcomes, n_men=n_men, p_max=p_max)
    turn_actions_cached = make_turn_action_cache(outcomes)

    @lru_cache(maxsize=None)
    def V(state: State) -> float:
        if is_blot(state):
            return 0.0

        total = 0.0
        for oi, o in enumerate(outcomes):
            turn_actions = turn_actions_cached(state, oi)

            if not turn_actions:
                best = 0.0
            else:
                best = float("-inf")
                for _a, nxt in turn_actions:
                    cand = 0.0 if is_blot(nxt) else (1.0 + V(nxt))
                    if cand > best:
                        best = cand

            total += o.p * best

        return total

    def best_transition(state: State, outcome_idx: int):
        """
        Return (best_action, best_next_state_or_None, best_q)
        where q = 0 if blot else 1 + V(next).
        """
        turn_actions = turn_actions_cached(state, outcome_idx)

        if not turn_actions:
            return tuple(0 for _ in range(len(state))), None, 0.0

        best_a, best_nxt = turn_actions[0]
        best_val = 0.0 if is_blot(best_nxt) else (1.0 + V(best_nxt))

        for a, nxt in turn_actions[1:]:
            val = 0.0 if is_blot(nxt) else (1.0 + V(nxt))
            if val > best_val:
                best_val = val
                best_a = a
                best_nxt = nxt

        return best_a, best_nxt, best_val

    init_state = tuple([n_men] + [0] * p_max)
    return V, best_transition, init_state, outcomes


def _is_double_from_name(name: str) -> bool:
    return len(name) == 2 and name[0] == name[1]


def _sample_outcome_index(outcomes: List[Outcome], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, o in enumerate(outcomes):
        acc += o.p
        if r <= acc:
            return i
    return len(outcomes) - 1


def show_non_double_survivals(
    n_episodes: int,
    max_steps_per_episode: int,
    init_state: State,
    outcomes: List[Outcome],
    best_transition,
    seed: Optional[int] = 0,
) -> None:
    """
    Simulate episodes under optimal transition choice and print every event where
    a NON-DOUBLE outcome leads to a non-blot next state.
    """
    rng = random.Random(seed)

    total_steps = 0
    total_non_double_rolls = 0
    total_non_double_survivals = 0

    print("\n=== Non-double survival tracer ===")
    print(f"episodes={n_episodes}, max_steps_per_episode={max_steps_per_episode}, seed={seed}")

    for ep in range(1, n_episodes + 1):
        s = init_state
        for t in range(1, max_steps_per_episode + 1):
            if is_blot(s):
                break

            oi = _sample_outcome_index(outcomes, rng)
            o = outcomes[oi]
            a, nxt, q = best_transition(s, oi)

            total_steps += 1
            non_double = not _is_double_from_name(o.name)
            if non_double:
                total_non_double_rolls += 1

            if nxt is None:
                break

            survived = not is_blot(nxt)
            if non_double and survived:
                total_non_double_survivals += 1
                print(
                    f"[EP {ep:03d} | step {t:03d}] NON-DOUBLE SURVIVAL: "
                    f"roll={o.name}, state={s}, action={a}, next={nxt}, q={q:.6f}"
                )

            if not survived:
                break

            s = nxt

    rate = (total_non_double_survivals / total_non_double_rolls) if total_non_double_rolls else 0.0
    print("--- Summary ---")
    print(f"total_steps_simulated={total_steps}")
    print(f"non_double_rolls={total_non_double_rolls}")
    print(f"non_double_survivals={total_non_double_survivals}")
    print(f"non_double_survival_rate={rate:.6f}")


if __name__ == "__main__":
    start = time.perf_counter()

    # Choose d2 or d6 outcomes here.
    USE_D6 = False

    if USE_D6:
        outcomes = [
            Outcome("11", 1/36, (1, 1, 1, 1), 1, 4),
            Outcome("12", 2/36, (1, 2), 1, 2),
            Outcome("13", 2/36, (1, 3), 1, 2),
            Outcome("14", 2/36, (1, 4), 1, 2),
            Outcome("15", 2/36, (1, 5), 1, 2),
            Outcome("16", 2/36, (1, 6), 1, 2),
            Outcome("22", 1/36, (2, 2, 2, 2), 1, 4),
            Outcome("23", 2/36, (2, 3), 1, 2),
            Outcome("24", 2/36, (2, 4), 1, 2),
            Outcome("25", 2/36, (2, 5), 1, 2),
            Outcome("26", 2/36, (2, 6), 1, 2),
            Outcome("33", 1/36, (3, 3, 3, 3), 1, 4),
            Outcome("34", 2/36, (3, 4), 1, 2),
            Outcome("35", 2/36, (3, 5), 1, 2),
            Outcome("36", 2/36, (3, 6), 1, 2),
            Outcome("44", 1/36, (4, 4, 4, 4), 1, 4),
            Outcome("45", 2/36, (4, 5), 1, 2),
            Outcome("46", 2/36, (4, 6), 1, 2),
            Outcome("55", 1/36, (5, 5, 5, 5), 1, 4),
            Outcome("56", 2/36, (5, 6), 1, 2),
            Outcome("66", 1/36, (6, 6, 6, 6), 1, 4),
        ]
        p_max = 200
    else:
        # d2 puzzle
        outcomes = [
            Outcome("11", 1/4, (1, 1, 1, 1), 1, 4),
            Outcome("12", 1/4, (1, 2), 1, 2),
            Outcome("21", 1/4, (2, 1), 1, 2),
            Outcome("22", 1/4, (2, 2, 2, 2), 1, 4),
        ]
        p_max = 200

    V, best_transition, s0, used_outcomes = make_dp_solver(
        n_men=5,
        p_max=p_max,
        outcomes=outcomes,
    )

    print("Exact EV from initial state:", V(s0))

    show_non_double_survivals(
        n_episodes=100,
        max_steps_per_episode=300,
        init_state=s0,
        outcomes=used_outcomes,
        best_transition=best_transition,
        seed=371,
    )

    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.3f}s")