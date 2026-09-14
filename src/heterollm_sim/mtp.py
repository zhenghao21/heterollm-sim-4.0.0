"""Shared deterministic multi-token-prediction request semantics.

``candidate_tokens`` in the workload IR is the maximum number of *draft*
tokens.  A verifier round therefore contains one guaranteed main-head token
plus zero or more drafts.  Static planning, online serving, and cheap run
estimation all use :class:`MTPRequestCursor` so fractional expectations are
rounded only after they have accumulated for the request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


@dataclass
class CumulativeExpectedCursor:
    """Emit integer tokens from a cumulative (possibly fractional) mean.

    Rounding each round independently introduces a systematic bias.  This
    cursor instead rounds the request-wide cumulative expectation and emits
    only the newly due integer tokens.  ``maximum_tokens`` clamps a particular
    round without discarding any unserved cumulative expectation.
    """

    expected_total: float = 0.0
    emitted_total: int = 0

    def advance(self, expected_tokens: float, maximum_tokens: int) -> int:
        expected = float(expected_tokens)
        maximum = max(0, int(maximum_tokens))
        if not math.isfinite(expected) or expected < 0.0:
            raise ValueError("expected_tokens must be finite and non-negative")
        self.expected_total += expected
        target = max(0, _round_half_up(self.expected_total))
        emitted = max(0, min(maximum, target - self.emitted_total))
        self.emitted_total += emitted
        return emitted


def expected_draft_prefix_tokens(
    draft_tokens: int, acceptance_rate: float
) -> float:
    """Return the expected accepted *draft-only* prefix length.

    Draft position zero is accepted with probability ``rate``; each later
    draft requires every preceding draft to have continued the prefix.
    """

    drafts = max(0, int(draft_tokens))
    rate = max(0.0, min(1.0, float(acceptance_rate)))
    return sum(rate**position for position in range(1, drafts + 1))


def expected_prefix_tokens(proposed_tokens: int, acceptance_rate: float) -> float:
    """Return the legacy main-plus-draft expected prefix length.

    ``proposed_tokens`` here is a verifier width, not the workload
    ``candidate_tokens`` field.  New MTP execution code should use
    :func:`expected_draft_prefix_tokens` through :class:`MTPRequestCursor`.
    """

    proposed = max(0, int(proposed_tokens))
    if proposed == 0:
        return 0.0
    return 1.0 + expected_draft_prefix_tokens(proposed - 1, acceptance_rate)


def round_accepted_prefix(proposed_tokens: int, expected_tokens: float) -> int:
    """Round one expected prefix half-up and clamp it to the proposal."""

    proposed = max(0, int(proposed_tokens))
    if proposed == 0:
        return 0
    rounded = _round_half_up(expected_tokens)
    return max(1, min(proposed, rounded))


def _acceptance_rate(policy: Any, round_index: int) -> float:
    model = str(getattr(policy, "acceptance_model", "")).strip().lower()
    if model in {"expected", "expected_prefix"}:
        raw_rate = getattr(policy, "acceptance_rate", None)
        rate = 0.0 if raw_rate is None else float(raw_rate)
    elif model == "trace":
        trace = tuple(getattr(policy, "acceptance_trace", ()) or ())
        if not trace:
            raise ValueError("MTP acceptance_model='trace' requires acceptance_trace")
        rate = float(trace[int(round_index) % len(trace)])
    else:
        raise ValueError(
            "unsupported MTP acceptance_model {}; expected expected, "
            "expected_prefix, or trace".format(model or "<empty>")
        )
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("MTP acceptance value must be in [0, 1]")
    return rate


def _proposal_draft_tokens(
    policy: Any,
    round_index: int,
    expected_cursor: CumulativeExpectedCursor,
) -> int:
    maximum = max(0, int(getattr(policy, "candidate_tokens", 0)))
    minimum = max(0, int(getattr(policy, "min_draft_tokens", 0)))
    minimum = min(minimum, maximum)
    model = str(getattr(policy, "proposal_length_model", "max")).strip().lower()
    if model == "max":
        return maximum
    if model == "trace":
        trace = tuple(getattr(policy, "draft_length_trace", ()) or ())
        if not trace:
            raise ValueError(
                "MTP proposal_length_model='trace' requires draft_length_trace"
            )
        return max(minimum, min(maximum, int(trace[int(round_index) % len(trace)])))
    if model == "expected_mean":
        raw_expected = getattr(policy, "expected_draft_tokens_per_round", None)
        if raw_expected is None:
            raise ValueError(
                "MTP proposal_length_model='expected_mean' requires "
                "expected_draft_tokens_per_round"
            )
        expected = float(raw_expected)
        if not math.isfinite(expected) or not minimum <= expected <= maximum:
            raise ValueError(
                "MTP expected_draft_tokens_per_round must be between "
                "min_draft_tokens and candidate_tokens"
            )
        return minimum + expected_cursor.advance(
            expected - minimum, maximum - minimum
        )
    raise ValueError(
        "unsupported MTP proposal_length_model {}; expected max, "
        "expected_mean, or trace".format(model or "<empty>")
    )


@dataclass(frozen=True)
class MTPRound:
    round_index: int
    main_tokens: int
    draft_tokens: int
    verifier_tokens: int
    expected_accepted_draft_tokens: float
    accepted_draft_tokens: int
    committed_tokens: int
    rejected_draft_tokens: int


class MTPRequestCursor:
    """Per-request deterministic MTP proposal and acceptance state."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.round_index = 0
        self._proposal_cursor = CumulativeExpectedCursor()
        self._acceptance_cursor = CumulativeExpectedCursor()

    @property
    def expected_draft_total(self) -> float:
        return self._acceptance_cursor.expected_total

    @property
    def accepted_draft_total(self) -> int:
        return self._acceptance_cursor.emitted_total

    def clone(self) -> "MTPRequestCursor":
        result = MTPRequestCursor(self.policy)
        result.round_index = self.round_index
        result._proposal_cursor = CumulativeExpectedCursor(
            self._proposal_cursor.expected_total,
            self._proposal_cursor.emitted_total,
        )
        result._acceptance_cursor = CumulativeExpectedCursor(
            self._acceptance_cursor.expected_total,
            self._acceptance_cursor.emitted_total,
        )
        return result

    def next_round(self, remaining_tokens: int) -> MTPRound:
        remaining = max(0, int(remaining_tokens))
        if remaining <= 0:
            raise ValueError("MTP remaining_tokens must be positive")
        round_index = self.round_index
        planned_drafts = _proposal_draft_tokens(
            self.policy, round_index, self._proposal_cursor
        )
        drafts = min(max(0, remaining - 1), planned_drafts)
        expected_drafts = expected_draft_prefix_tokens(
            drafts, _acceptance_rate(self.policy, round_index)
        )
        accepted_drafts = self._acceptance_cursor.advance(
            expected_drafts, drafts
        )
        committed = min(remaining, 1 + accepted_drafts)
        verifier = 1 + drafts
        self.round_index += 1
        return MTPRound(
            round_index=round_index,
            main_tokens=1,
            draft_tokens=drafts,
            verifier_tokens=verifier,
            expected_accepted_draft_tokens=expected_drafts,
            accepted_draft_tokens=accepted_drafts,
            committed_tokens=committed,
            rejected_draft_tokens=drafts - accepted_drafts,
        )


def mtp_rounds_for_output(
    output_tokens: int, policy: Any
) -> Tuple[MTPRound, ...]:
    """Return request rounds for an already-prefilled output suffix."""

    remaining = max(0, int(output_tokens))
    cursor = MTPRequestCursor(policy)
    rounds = []
    while remaining > 0:
        item = cursor.next_round(remaining)
        rounds.append(item)
        remaining -= item.committed_tokens
    return tuple(rounds)


__all__ = [
    "CumulativeExpectedCursor",
    "MTPRequestCursor",
    "MTPRound",
    "expected_draft_prefix_tokens",
    "expected_prefix_tokens",
    "mtp_rounds_for_output",
    "round_accepted_prefix",
]
