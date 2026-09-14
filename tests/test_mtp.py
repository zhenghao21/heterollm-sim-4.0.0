import unittest
from types import SimpleNamespace

from heterollm_sim.mtp import (
    MTPRequestCursor,
    expected_draft_prefix_tokens,
    expected_prefix_tokens,
    round_accepted_prefix,
)


class MTPAcceptanceTests(unittest.TestCase):
    def test_round_half_up_is_clamped_to_committable_prefix(self):
        self.assertEqual(round_accepted_prefix(0, 10.0), 0)
        self.assertEqual(round_accepted_prefix(4, 0.0), 1)
        self.assertEqual(round_accepted_prefix(4, 2.5), 3)
        self.assertEqual(round_accepted_prefix(4, 10.0), 4)

    def test_expected_prefix_uses_main_token_plus_geometric_candidates(self):
        self.assertAlmostEqual(
            expected_prefix_tokens(4, 0.8),
            1.0 + 0.8 + 0.8**2 + 0.8**3,
        )

    def test_draft_only_prefix_excludes_the_guaranteed_main_token(self):
        self.assertAlmostEqual(
            expected_draft_prefix_tokens(4, 0.8),
            0.8 + 0.8**2 + 0.8**3 + 0.8**4,
        )

    def test_request_cursor_accumulates_instead_of_rounding_each_round(self):
        policy = SimpleNamespace(
            candidate_tokens=1,
            min_draft_tokens=0,
            proposal_length_model="max",
            expected_draft_tokens_per_round=None,
            draft_length_trace=(),
            acceptance_model="expected",
            acceptance_rate=0.4,
            acceptance_trace=(),
        )
        cursor = MTPRequestCursor(policy)
        rounds = [cursor.next_round(99) for _ in range(5)]

        self.assertEqual(
            [item.verifier_tokens for item in rounds],
            [2, 2, 2, 2, 2],
        )
        self.assertEqual(
            [item.accepted_draft_tokens for item in rounds],
            [0, 1, 0, 1, 0],
        )
        self.assertEqual(
            [item.committed_tokens for item in rounds],
            [1, 2, 1, 2, 1],
        )

    def test_expected_mean_proposal_length_uses_its_own_cumulative_cursor(self):
        policy = SimpleNamespace(
            candidate_tokens=2,
            min_draft_tokens=0,
            proposal_length_model="expected_mean",
            expected_draft_tokens_per_round=1.5,
            draft_length_trace=(),
            acceptance_model="expected",
            acceptance_rate=1.0,
            acceptance_trace=(),
        )
        cursor = MTPRequestCursor(policy)
        self.assertEqual(
            [cursor.next_round(99).draft_tokens for _ in range(4)],
            [2, 1, 2, 1],
        )


if __name__ == "__main__":
    unittest.main()
