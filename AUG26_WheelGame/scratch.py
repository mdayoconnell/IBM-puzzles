import numpy as np
import sympy as sp

v = np.ones((1, 8), dtype=np.uint8)

N = np.zeros((8, 8), dtype=np.uint8)
for i in range(7):
    N[i, i] = 1
    N[i, i + 1] = 1
N[7, 7] = 1




def as_bit(vector):
    """
    Pack an 8-element binary vector into a uint8.
    Example:
        [1,0,1,0,0,1,1,0] -> 166
    """
    bits = np.asarray(vector, dtype=np.uint8).ravel()

    if bits.size != 8:
        raise ValueError("Expected exactly 8 bits.")

    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Entries must be 0 or 1.")

    return np.packbits(bits, bitorder="big")[0]


def from_bit(x):
    """
    Inverse of as_bit().
    Example:
        166 -> array([1,0,1,0,0,1,1,0], dtype=uint8)
    """
    return np.unpackbits(np.array([x], dtype=np.uint8), bitorder="big")


def jordan_basis(A):
    """
    Return P, J satisfying

        J = P^{-1} A P
    """
    A = sp.Matrix(np.asarray(A, dtype=int))
    P, J = A.jordan_form()
    return P, J


# Example
P, J = jordan_basis(N)

# print("Jordan basis P:")
# print(P)

# print("\nJordan form J:")
# print(J)

# print("\nBit example:")
# print(as_bit([1, 0, 1, 0, 0, 1, 1, 0]))
# print(from_bit(166))

def clean_bit(vector):
    vec_as_string = str(vector)
    remove_chars = " []"
    return(vec_as_string.translate(str.maketrans("", "", remove_chars)))

# for i in range(0,9):
#     print("-"*80)
#     print("v N^{}".format(i))
#     print(clean_bit(v @ np.linalg.matrix_power(N,i) % 2))
#     print("as bit {}".format(as_bit(v @ np.linalg.matrix_power(N,i) % 2)))

print(sp.Matrix(N.astype(int)).eigenvects())
