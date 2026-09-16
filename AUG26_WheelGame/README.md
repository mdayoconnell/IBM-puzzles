# Eight-bit wheel game

This folder contains the explicit 255-move solution for the blind eight-node
wheel search.

- `util.py` converts between an 8-character bit string, one raw byte, a Python
  integer in `0..255`, and a tuple of eight booleans. `xor_bools` applies a move.
- `jordanchain.py` computes the minimum-score Jordan chain from
  `N(x) = rotate_right(x) XOR x`, implements the ruler function, and verifies
  every belief transition. Node 1 is the most-significant bit and node costs
  increase from 1 through 8 (or through 64).
- `wheel_demo.py` visualizes the 255 transitions and audits that arbitrary
  rotation adds no candidates.
- `scoring.py` implements the shared node-weight score and computes the total
  without materializing the exponentially long move sequence.
- `final_answer.py` prints the score for `n=8`; `n64_answer.py` prints the
  chain and score for `n=64`.

Run the verification:

```sh
python3 -m unittest -v
```

Print the two scored answers:

```sh
python3 final_answer.py
python3 n64_answer.py
```

Open the interactive demonstration:

```sh
/usr/bin/python3 wheel_demo.py
```

On this Mac, the Homebrew `python3` does not include Tkinter; Apple’s system
Python does. On another machine, use whichever Python installation can
successfully run `python3 -m tkinter`.

`"01000001"` is decimal `65` and the raw byte `b"A"`. Decimal `127` is the bit
string `"01111111"`.
