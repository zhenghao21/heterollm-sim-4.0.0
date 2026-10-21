"""Small explicit slot/phase oracles for dynamic paged KV capacity."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heterollm_sim.paper_kv_capacity import staggered_kv_token_series


def _oracle(batch, ep, prompt, output, chunk):
    life = output - 1
    full = [[0] * life for _ in range(ep)]
    chunks = [[0] * life for _ in range(ep)]
    for slot in range(batch):
        rank, initial_age = slot % ep, slot * life // batch
        for phase in range(life):
            age = (initial_age + phase) % life
            full[rank][phase] += prompt + age + 1
            if chunk:
                chunks[rank][phase] += (prompt + age) % chunk + 1
    return tuple(map(tuple, full)), tuple(map(tuple, chunks))


@pytest.mark.parametrize("batch,ep", [(1, 1), (2, 1), (3, 2), (4, 4), (9, 2), (19, 3), (2, 5)])
@pytest.mark.parametrize("prompt,output,chunk", [(1, 2, 0), (1, 5, 0), (3, 8, 4),
                                                (4, 7, 4), (7, 11, 3), (10, 7, 1),
                                                (16, 3, 8), (9, 4, 8), (7, 2, 4)])
def test_histogram_recurrence_matches_each_slot_at_every_phase(batch, ep, prompt, output, chunk):
    values = staggered_kv_token_series(batch, ep, prompt, output, chunk)
    full, chunks = _oracle(batch, ep, prompt, output, chunk)
    assert values.full_by_ep_phase == full
    assert values.chunk_by_ep_phase == chunks
    assert values.life_steps == output - 1
    assert values.requests_per_ep == tuple(sum(slot % ep == rank for slot in range(batch)) for rank in range(ep))


def test_chunk_full_page_exists_before_eviction_and_lifetime_wrap_is_separate():
    # P=3, L=6 and C=4 crosses a normal chunk boundary at age zero,
    # crosses another at age four, then wraps age five to age zero.
    values = staggered_kv_token_series(1, 1, 3, 7, 4)
    assert values.full_by_ep_phase == ((4, 5, 6, 7, 8, 9),)
    assert values.chunk_by_ep_phase == ((4, 1, 2, 3, 4, 1),)
    assert max(values.chunk_by_ep_phase[0]) == 4
    # Here the lifetime wrap is itself a chunk boundary. It must not receive
    # both the normal chunk correction and the lifetime-wrap correction.
    both = staggered_kv_token_series(2, 1, 1, 4, 4)
    assert (both.full_by_ep_phase, both.chunk_by_ep_phase) == _oracle(2, 1, 1, 4, 4)


def test_stage_peak_is_maximum_of_weighted_same_phase_not_independent_maxima():
    values = staggered_kv_token_series(2, 1, 1, 5, 4)
    full, chunk = values.full_by_ep_phase[0], values.chunk_by_ep_phase[0]
    assert full == (6, 8, 6, 8)
    assert chunk == (6, 4, 6, 4)
    actual_stage_peak = max(f + c for f, c in zip(full, chunk))
    assert actual_stage_peak == 12
    assert max(full) + max(chunk) == 14


def test_per_ep_private_peaks_are_not_a_pooled_average():
    values = staggered_kv_token_series(6, 2, 2, 7, 4)
    assert values.full_by_ep_phase[0] != values.full_by_ep_phase[1]
    assert values.full_by_ep_phase == _oracle(6, 2, 2, 7, 4)[0]
    for phase in range(6):
        assert sum(row[phase] for row in values.full_by_ep_phase) == sum(
            2 + ((slot + phase) % 6) + 1 for slot in range(6))


@pytest.mark.parametrize("prompt,output", [(1660, 373), (5900, 499), (103500, 1100)])
def test_paper_lifetimes_shorter_than_8192_chunks_are_exact(prompt, output):
    values = staggered_kv_token_series(11, 3, prompt, output, 8192)
    full, chunk = _oracle(11, 3, prompt, output, 8192)
    assert values.full_by_ep_phase == full and values.chunk_by_ep_phase == chunk
    # None of these three declared mean workloads crosses a chunk boundary
    # inside one request's decode lifetime; only request completion resets age.
    assert prompt % 8192 + output - 2 < 8192
    offset = prompt - prompt % 8192
    for rank in range(3):
        assert all(f - c == values.requests_per_ep[rank] * offset
                   for f, c in zip(full[rank], chunk[rank]))


def test_no_chunk_geometry_does_not_allocate_fictitious_cache_tokens():
    values = staggered_kv_token_series(5, 2, 20, 6)
    assert values.chunk_by_ep_phase == ((0, 0, 0, 0, 0),) * 2


def test_result_is_immutable_and_cached_with_bounded_entries():
    staggered_kv_token_series.cache_clear()
    values = staggered_kv_token_series(2, 1, 20, 6, 4)
    assert values is staggered_kv_token_series(2, 1, 20, 6, 4)
    with pytest.raises(FrozenInstanceError):
        values.life_steps = 20
    with pytest.raises(TypeError):
        values.full_by_ep_phase[0][0] = 0
    assert staggered_kv_token_series.cache_info().maxsize == 128
    for prompt in range(1, 145):
        staggered_kv_token_series(1, 1, prompt, 2)
    assert staggered_kv_token_series.cache_info().currsize == 128


@pytest.mark.parametrize("position,value", [(0, 0), (0, -1), (0, True), (0, 1.0),
    (1, 0), (1, False), (1, 2.5), (2, 0), (2, True), (2, "10"),
    (3, 1), (3, 0), (3, False), (3, 3.0), (4, -1), (4, True), (4, 4.0)])
def test_invalid_inputs_do_not_round_or_bypass_cached_validation(position, value):
    staggered_kv_token_series(1, 1, 1, 2, 0)
    arguments = [1, 1, 1, 2, 0]
    arguments[position] = value
    with pytest.raises(ValueError):
        staggered_kv_token_series(*arguments)
