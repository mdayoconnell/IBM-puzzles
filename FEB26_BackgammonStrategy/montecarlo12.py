from __future__ import annotations

import argparse
import bisect
import math
import random
import statistics
import time
from functools import lru_cache
from typing import Callable, Iterable, List, Sequence, Tuple

from backgammon_strategy12 import (
    Action,
    Outcome,
    State,
    is_blot,
    make_dp_solver,
    validate_outcomes,
    valid_turn_actions,
)

DEFAULT_N_MEN = 5
DEFAULT_P_MAX = 10


def make_default_outcomes(n_men: int) -> List[Outcome]:
    return [
        Outcome("11", 0.25, (1, 1, 1, 1), 1, 4),
        Outcome("12", 0.25, (1, 2), 1, 2),
        Outcome("21", 0.25, (2, 1), 1, 2),
        Outcome("22", 0.25, (2, 2, 2, 2), 1, 4),
    ]


TurnAction = Tuple[Action, State]
TurnActions = Tuple[TurnAction, ...]
PolicyFn = Callable[[State, int, TurnActions], TurnAction]


def build_cdf(outcomes: Sequence[Outcome]) -> List[float]:
    total = 0.0
    cdf: List[float] = []
    for o in outcomes:
        total += o.p
        cdf.append(total)
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"Outcome probabilities must sum to 1. Got {total}.")
    return cdf


def sample_outcome_index(rng: random.Random, cdf: Sequence[float]) -> int:
    r = rng.random()
    idx = bisect.bisect_left(cdf, r)
    if idx >= len(cdf):
        idx = len(cdf) - 1
    return idx


def make_turn_action_cache(outcomes: Sequence[Outcome]):
    @lru_cache(maxsize=None)
    def cached(state: State, outcome_idx: int) -> TurnActions:
        return tuple(valid_turn_actions(state, outcomes[outcome_idx]))

    return cached


def make_random_policy(rng: random.Random) -> PolicyFn:
    def choose(_state: State, _outcome_idx: int, actions: TurnActions) -> TurnAction:
        return actions[rng.randrange(len(actions))]

    return choose


def make_optimal_policy(
    outcomes: Sequence[Outcome],
    n_men: int,
    p_max: int,
    turn_actions_cached,
) -> PolicyFn:
    V, _best_action, _s0, _used_outcomes = make_dp_solver(
        n_men=n_men,
        p_max=p_max,
        outcomes=list(outcomes),
    )

    @lru_cache(maxsize=None)
    def best_transition(state: State, outcome_idx: int) -> TurnAction:
        actions = turn_actions_cached(state, outcome_idx)
        if not actions:
            return tuple(0 for _ in range(len(state))), tuple(state)

        best_a, best_nxt = actions[0]
        best_val = 0.0 if is_blot(best_nxt) else (1.0 + V(best_nxt))
        for a, nxt in actions[1:]:
            val = 0.0 if is_blot(nxt) else (1.0 + V(nxt))
            if val > best_val:
                best_val = val
                best_a = a
                best_nxt = nxt
        return best_a, best_nxt

    def choose(state: State, outcome_idx: int, _actions: TurnActions) -> TurnAction:
        return best_transition(state, outcome_idx)

    return choose


def simulate_game(
    rng: random.Random,
    cdf: Sequence[float],
    turn_actions_cached,
    policy: PolicyFn,
    init_state: State,
    max_rolls: int = 1_000_000,
) -> int:
    rolls = 0
    state = init_state
    for _ in range(max_rolls):
        oi = sample_outcome_index(rng, cdf)
        actions = turn_actions_cached(state, oi)
        if not actions:
            return rolls

        _a, nxt = policy(state, oi, actions)
        if is_blot(nxt):
            return rolls

        rolls += 1
        state = nxt
    return rolls


def estimate_ev(
    num_trials: int,
    seed: int,
    outcomes: Sequence[Outcome],
    n_men: int,
    p_max: int,
    policy_name: str,
) -> Tuple[float, float, float, float]:
    validate_outcomes(list(outcomes), n_men=n_men, p_max=p_max)
    turn_actions_cached = make_turn_action_cache(outcomes)
    rng = random.Random(seed)

    if policy_name == "optimal":
        policy = make_optimal_policy(outcomes, n_men, p_max, turn_actions_cached)
    elif policy_name == "random":
        policy = make_random_policy(rng)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    init_state = tuple([n_men] + [0] * p_max)
    cdf = build_cdf(outcomes)

    results: List[int] = []
    for _ in range(num_trials):
        results.append(
            simulate_game(
                rng=rng,
                cdf=cdf,
                turn_actions_cached=turn_actions_cached,
                policy=policy,
                init_state=init_state,
            )
        )

    mean = float(sum(results)) / num_trials if num_trials else 0.0
    stdev = statistics.pstdev(results) if num_trials > 1 else 0.0
    sem = stdev / math.sqrt(num_trials) if num_trials > 1 else 0.0
    ci95 = 1.96 * sem
    return mean, stdev, sem, ci95


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo solver for the d2 game.")
    parser.add_argument("--trials", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n-men", type=int, default=DEFAULT_N_MEN)
    parser.add_argument("--p-max", type=int, default=DEFAULT_P_MAX)
    parser.add_argument("--policy", choices=["optimal", "random"], default="optimal")
    args = parser.parse_args()

    outcomes = make_default_outcomes(args.n_men)

    t0 = time.perf_counter()
    mean, stdev, sem, ci95 = estimate_ev(
        num_trials=args.trials,
        seed=args.seed,
        outcomes=outcomes,
        n_men=args.n_men,
        p_max=args.p_max,
        policy_name=args.policy,
    )
    dt = time.perf_counter() - t0

    print("Monte Carlo (d2)")
    print(f"  policy={args.policy}")
    print(f"  trials={args.trials}, seed={args.seed}")
    print(f"  EV≈{mean:.6f}  stdev={stdev:.6f}  sem={sem:.6f}  95% CI ±{ci95:.6f}")
    print(f"  elapsed={dt:.3f}s")


if __name__ == "__main__":
    main()
