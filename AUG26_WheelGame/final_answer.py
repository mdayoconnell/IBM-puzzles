"""Print the minimum score for the 255-move, eight-node strategy."""

from scoring import jordan_total_score


def main() -> None:
    total = jordan_total_score(8)
    print(f"n=8 total score over 255 moves: {total}")


if __name__ == "__main__":
    main()
