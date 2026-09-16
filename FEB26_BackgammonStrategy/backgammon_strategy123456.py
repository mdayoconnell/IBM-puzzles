from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Dict, Tuple, List
from itertools import product


State = Tuple[int, ...]        # number of men per position
Action = Tuple[int, ...]       # action[p] = men moved p -> p+1


@dataclass(frozen=True)
class Outcome:
    name: str
    p: float
    die_uses: Tuple[int, ...]          # e.g. (1, 2) or (2,2,2,2)
    k_min: int = 1                      # min distinct men/stacks used this turn
    k_max: int = 4                      # max distinct men/stacks used this turn


def _canonical_outcome_key(o: Outcome) -> Tuple[Tuple[int, ...], int, int]:
    """
    Canonical key for action-generation cache.
    Outcomes that differ only by die-use ordering (e.g. 12 vs 21) share cache entries.
    """
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

        # If there are no die uses, no stacks should be required/allowed.
        if len(o.die_uses) == 0:
            if not (o.k_min == 0 and o.k_max == 0):
                raise ValueError(f"Empty die_uses requires k_min=k_max=0 in outcome {o.name}.")
        else:
            if o.k_max == 0:
                raise ValueError(f"Non-empty die_uses requires k_max>=1 in outcome {o.name}.")


def is_blot(state: State) -> bool:
    # occupied point with exactly one man
    return any(c == 1 for c in state if c > 0)


def apply_action(state: State, action: Action) -> State:
    arr = list(state)
    p_max = len(arr) - 1
    for p, k in enumerate(action):
        if k == 0:
            continue
        arr[p] -= k
        arr[p + 1] += k  # only valid if p < p_max; ensured by action validity
    return tuple(arr)


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

    # Build checker positions by identity
    checker_pos = list(src_tuple)

    # Simulate die uses in order (same checker can be moved multiple times)
    for d, lbl in zip(die_uses, label_assignment):
        cur = checker_pos[lbl]
        nxt = cur + d
        if nxt > p_max:
            return tuple([-1]), tuple()
        checker_pos[lbl] = nxt

    # Reconstruct resulting state from untouched men + moved checkers
    arr = list(state)
    for src in src_tuple:
        arr[src] -= 1
        if arr[src] < 0:
            return tuple([-1]), tuple()

    for pos in checker_pos:
        arr[pos] += 1

    # Build action vector: number of men chosen from each source at turn start
    action = [0] * len(state)
    for src in src_tuple:
        action[src] += 1

    return tuple(arr), tuple(action)


def valid_turn_actions(state: State, outcome: Outcome) -> List[Tuple[Action, State]]:
    """
    Generate legal turn actions for a realized outcome.

    Interpretation:
      - die_uses are assigned to distinct MEN (checker identities),
      - k_min/k_max constrain number of distinct men used,
      - multiple chosen men may start from the same source position,
      - blot is checked only at end of turn.

    Returns list of (aggregated_action_vector, next_state).
    """
    p_max = len(state) - 1
    die_uses = outcome.die_uses
    m = len(die_uses)

    if m == 0:
        return [(tuple(0 for _ in state), state)]

    occupied_positions = [i for i, c in enumerate(state) if c > 0]
    if not occupied_positions:
        return []

    out: List[Tuple[Action, State]] = []
    seen = set()

    # Number of distinct men used this turn.
    k_lo = max(1, outcome.k_min)
    k_hi = min(outcome.k_max, m, sum(state))
    if k_lo > k_hi:
        return []

    # Step 1: assign each die-use to a man-label in {0,...,k-1}, surjectively.
    # This permits "same man moved multiple times" by reusing labels.
    for k in range(k_lo, k_hi + 1):
        for label_assignment in product(range(k), repeat=m):
            if len(set(label_assignment)) != k:  # require all k men actually used
                continue

            # Step 2: choose source positions for those k men.
            # Multiple men can share the same source (critical for initial all-on-0 state).
            for src_tuple in product(occupied_positions, repeat=k):
                # capacity check at sources: how many distinct men are drawn from each source
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

                # Dedup by full identity plan effect on counts (a,nxt).
                key = (a, nxt)
                if key in seen:
                    continue
                seen.add(key)
                out.append((a, nxt))

    return out


def make_turn_action_cache(outcomes: List[Outcome]):
    """
    Returns cached function: (state, outcome_idx) -> tuple[(action, next_state), ...]
    Uses canonicalized outcome keys to share work across symmetric outcomes.
    """
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


# --- Consistency check utility ---
def debug_outcome_actions(state: State, outcomes: List[Outcome], max_examples: int = 5) -> None:
    """
    Print a consistency summary of legal turn-actions for each outcome at a given state.
    Useful to verify rule semantics before running full DP.
    """
    print("\nConsistency check at state:", state)
    for idx, o in enumerate(outcomes):
        turn_actions = valid_turn_actions(state, o)
        print(
            f"  Outcome[{idx}] {o.name}: p={o.p}, die_uses={o.die_uses}, "
            f"k_range=[{o.k_min},{o.k_max}] -> legal_actions={len(turn_actions)}"
        )

        if not turn_actions:
            continue

        # Show up to max_examples unique next states with one representative action each
        shown = 0
        seen_next = set()
        for a, nxt in turn_actions:
            if nxt in seen_next:
                continue
            seen_next.add(nxt)
            print(f"    ex {shown+1}: action={a} -> next={nxt}, blot={is_blot(nxt)}")
            shown += 1
            if shown >= max_examples:
                break


def debug_initial_best_actions(
    state: State,
    outcomes: List[Outcome],
    best_transition,
) -> None:
    """
    For each outcome at a given state, print the policy-chosen action,
    resulting next state, and whether that next state is a blot.

    This is a targeted sanity check to verify which outcomes are survivable
    under the current optimal policy.
    """
    print("\nBest-action sanity check at state:", state)
    for i, o in enumerate(outcomes):
        a, nxt, val = best_transition(state, i)
        if nxt is None:
            print(
                f"  Outcome[{i}] {o.name}: no legal actions; chosen action={a}, value=0.0"
            )
            continue

        print(
            f"  Outcome[{i}] {o.name}: action={a} -> next={nxt}, "
            f"blot={is_blot(nxt)}, q={val:.6f}"
        )


def make_dp_solver(
    n_men: int = 5,
    p_max: int = 10,
    outcomes: List[Outcome] = None,
    timing: bool = False,
):
    """
    Returns:
      V(state): exact optimal expected successful rolls before blot (failing roll NOT counted)
      best_action(state, outcome_idx): optimal action for realized outcome index
      init_state: (n_men, 0, ..., 0)
      outcomes: validated outcomes list (same order as used by solver)
    """

    validate_outcomes(outcomes, n_men=n_men, p_max=p_max)

    turn_actions_cached = make_turn_action_cache(outcomes)

    timing_stats = {
        "turn_action_calls": 0,
        "turn_action_wall": 0.0,
        "bellman_updates": 0,
        "bellman_wall": 0.0,
    }

    @lru_cache(maxsize=None)
    def V(state: State) -> float:
        if is_blot(state):
            return 0.0

        _v_t0 = time.perf_counter() if timing else 0.0

        total = 0.0
        for oi, o in enumerate(outcomes):
            _ta_t0 = time.perf_counter() if timing else 0.0
            turn_actions = turn_actions_cached(state, oi)
            if timing:
                timing_stats["turn_action_calls"] += 1
                timing_stats["turn_action_wall"] += time.perf_counter() - _ta_t0

            if not turn_actions:
                # No legal way to play this outcome -> immediate fail on this roll.
                # Convention: do NOT count failing roll.
                best = 0.0
            else:
                best = float("-inf")
                for _a, nxt in turn_actions:
                    # Blot checked only at end of turn.
                    cand = 0.0 if is_blot(nxt) else (1.0 + V(nxt))
                    if cand > best:
                        best = cand

            total += o.p * best

        if timing:
            timing_stats["bellman_updates"] += 1
            timing_stats["bellman_wall"] += time.perf_counter() - _v_t0

        return total

    def best_transition(state: State, outcome_idx: int):
        """
        returns the optimal transition tuple for a realized outcome:
          (best_action_vector, best_next_state_or_None, best_q_value)
        where q-value is 0 if next state is blot else 1 + V(next_state).
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

    def best_action(state: State, outcome_idx: int) -> Action:
        a, _nxt, _val = best_transition(state, outcome_idx)
        return a

    init_state = tuple([n_men] + [0] * p_max)
    return V, best_action, best_transition, init_state, outcomes, timing_stats


if __name__ == "__main__":
    start = time.perf_counter()

    SHOW_SANITY_CHECK = False
    SHOW_BEST_ACTION_SANITY = False
    SHOW_TIMING = True

    outcomes = [
        Outcome("11", 1/36, (1, 1, 1, 1), 1, 4),  # stated rules, can move 1-2 men on nondoubles, 1-4 men on doubles
        Outcome("12", 2/36, (1, 2), 1, 2),        # 
        Outcome("13", 2/36, (1, 3), 1, 2),
        Outcome("14", 2/36, (1, 4), 1, 2),
        Outcome("15", 2/36, (1, 5), 1, 2),        # 
        Outcome("16", 2/36, (1, 6), 1, 2),
        Outcome("22", 1/36, (2, 2, 2, 2), 1, 4),
        Outcome("23", 2/36, (2, 3), 1, 2),        # 
        Outcome("24", 2/36, (2, 4), 1, 2),
        Outcome("25", 2/36, (2, 5), 1, 2),
        Outcome("26", 2/36, (2, 6), 1, 2),        # 
        Outcome("33", 1/36, (3, 3, 3, 3), 1, 4),
        Outcome("34", 2/36, (3, 4), 1, 2),
        Outcome("35", 2/36, (3, 5), 1, 2),        # 
        Outcome("36", 2/36, (3, 6), 1, 2),
        Outcome("44", 1/36, (4, 4, 4, 4), 1, 4),
        Outcome("45", 2/36, (4, 5), 1, 2),        # 
        Outcome("46", 2/36, (4, 6), 1, 2),
        Outcome("55", 1/36, (5, 5, 5, 5), 1, 4),
        Outcome("56", 2/36, (5, 6), 1, 2),
        Outcome("66", 1/36, (6, 6, 6, 6), 1, 4),
        
    ]

    if SHOW_SANITY_CHECK:       # Sanity check applies to initial position
        s0_preview = tuple([5] + [0] * 12)
        debug_outcome_actions(s0_preview, outcomes, max_examples=5)     # Shows the best policy for each roll

    V, policy, best_transition, s0, used_outcomes, timing_stats = make_dp_solver(
        n_men=5,
        p_max=20,
        outcomes=outcomes,
        timing=SHOW_TIMING,
    )

    if SHOW_BEST_ACTION_SANITY:
        debug_initial_best_actions(s0, used_outcomes, best_transition)

    #print("Outcomes:")
    #for i, o in enumerate(used_outcomes):
    #    print(f"  [{i}] {o.name}: p={o.p}, die_uses={o.die_uses}, k_range=[{o.k_min},{o.k_max}]")

    print("Exact EV from initial state:", V(s0))
    if SHOW_TIMING:
        print("Timing stats:")
        print(f"  bellman_updates={timing_stats['bellman_updates']}")
        print(f"  bellman_wall={timing_stats['bellman_wall']:.3f}s")
        print(f"  turn_action_calls={timing_stats['turn_action_calls']}")
        print(f"  turn_action_wall={timing_stats['turn_action_wall']:.3f}s")

    for p in [200]:
        t0 = time.perf_counter()
        Vp, policyp, best_transition_p, s0p, _, stats_p = make_dp_solver(
            n_men=5,
            p_max=p,
            outcomes=outcomes,
            timing=SHOW_TIMING,
        )
        val = Vp(s0p)
        dt = time.perf_counter() - t0
        if SHOW_TIMING:
            print(
                f"p_max={p}: EV={val:.15f}, time={dt:.3f}s, "
                f"bellman={stats_p['bellman_wall']:.3f}s, turn_actions={stats_p['turn_action_wall']:.3f}s"
            )
        else:
            print(f"p_max={p}: EV={val:.15f}, time={dt:.3f}s")
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.3f}s")