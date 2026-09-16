import unittest

import jordanchain as jc
import scoring
import util


class ConversionTests(unittest.TestCase):
    def test_known_value_in_every_representation(self):
        self.assertEqual(util.str_to_int256("01000001"), 65)
        self.assertEqual(util.str_to_bytes("01000001"), b"A")
        self.assertEqual(
            util.str_to_bools("01000001"),
            (False, True, False, False, False, False, False, True),
        )

    def test_every_value_round_trips(self):
        for value in range(256):
            self.assertEqual(util.str_to_int256(util.int256_to_str(value)), value)
            self.assertEqual(util.bytes_to_int256(util.int256_to_bytes(value)), value)
            self.assertEqual(util.bools_to_int256(util.int256_to_bools(value)), value)

    def test_boolean_xor_flips_selected_nodes(self):
        state = util.str_to_bools("01000001")
        move = util.str_to_bools("00000001")
        self.assertEqual(util.bools_to_str(util.xor_bools(state, move)), "01000000")

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            util.int256_to_str(256)
        with self.assertRaises(ValueError):
            util.str_to_int256("101")
        with self.assertRaises(ValueError):
            util.bytes_to_int256(b"AB")


class JordanChainTests(unittest.TestCase):
    def test_computed_chain(self):
        self.assertEqual(jc.JORDAN_CHAIN, (255, 170, 204, 136, 240, 160, 192, 128))
        self.assertEqual(jc.nilpotent_step(jc.JORDAN_CHAIN[0]), 0)
        for level in range(1, 8):
            self.assertEqual(
                jc.nilpotent_step(jc.JORDAN_CHAIN[level]),
                jc.JORDAN_CHAIN[level - 1],
            )

    def test_ruler_moves(self):
        expected = (255, 170, 255, 204, 255, 170, 255, 136)
        self.assertEqual(jc.move_sequence(8), expected)

    def test_right_rotation_chain_relation(self):
        for level in range(1, 8):
            self.assertEqual(
                jc.rotate_right(jc.JORDAN_CHAIN[level])
                ^ jc.JORDAN_CHAIN[level],
                jc.JORDAN_CHAIN[level - 1],
            )

    def test_basis_conversion_round_trip(self):
        for value in range(256):
            self.assertEqual(
                jc.jordan_to_physical(jc.physical_to_jordan(value)), value
            )

    def test_all_beliefs_have_the_named_size(self):
        for size in range(256):
            self.assertEqual(len(jc.coordinate_belief(size)), size)
            self.assertEqual(len(jc.physical_belief(size)), size)

    def test_all_255_transitions_are_exact(self):
        for index in range(1, 256):
            transition = jc.strategy_transition(index)
            self.assertTrue(transition.is_exact, f"failed at move {index}")
            self.assertEqual(len(transition.rotation_added), 0)
        self.assertTrue(jc.verify_strategy())


class ScoringTests(unittest.TestCase):
    def test_example_move_score(self):
        self.assertEqual(scoring.move_score(0b01111111, 8), 35)

    def test_eight_node_answer(self):
        self.assertEqual(scoring.jordan_total_score(8), 6279)

    def test_sixty_four_node_answer_and_chain(self):
        chain = jc.compute_jordan_chain(64)
        self.assertEqual(chain[-1], 1 << 63)
        for level in range(1, 64):
            self.assertEqual(
                jc.rotate_right(chain[level], width=64) ^ chain[level],
                chain[level - 1],
            )
        self.assertEqual(
            scoring.jordan_total_score(64),
            27636190239652591799943,
        )


if __name__ == "__main__":
    unittest.main()
