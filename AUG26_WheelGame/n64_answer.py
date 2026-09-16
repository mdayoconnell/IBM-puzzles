"""Compute the minimum-score 64-node Jordan chain and total move score."""

from jordanchain import compute_jordan_chain
from scoring import jordan_total_score


WIDTH = 64


def main() -> None:
    chain = compute_jordan_chain(WIDTH)

    print("Jordan chain (a_0 .. a_63):")
    for level, move in enumerate(chain):
        print(f"a_{level} = {move:064b} ({move})")

    total_score = jordan_total_score(WIDTH)
    print(f"\nn=64 total score over {2**WIDTH - 1} moves: {total_score}")


if __name__ == "__main__":
    main()
