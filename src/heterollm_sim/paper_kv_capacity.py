"""Exact staggered KV occupancy, before releasing completed decode requests.

The full-batch controller has ``output_tokens - 1`` decode iterations per
request: the first output was produced by the separated prefill node. Each
stable slot returns to age zero immediately after completing its lifetime.
This module counts cached tokens after the current decode write and before
completed slots are released. Prepared successor KV is a separate pool and
must not be subtracted from these active-request counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


def _integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class StaggeredKVTokenSeries:
    """Private EP-rank token sums, indexed ``[expert_rank][phase]``.

    Full/chunk series share the SAME phase, so stage capacity must maximize
    their weighted sum, not add their independent maxima. The output has no
    TP factor: multiply each token sum by that rank's actual GQA K+V bytes.
    When ``chunk_size == 0`` there is no chunked cache, and chunk sums are zero.
    Uneven ``batch_size / ep`` is represented exactly; an inference backend
    may separately require divisible batches for balanced compute.
    """

    batch_size: int
    ep: int
    input_tokens: int
    output_tokens: int
    chunk_size: int
    life_steps: int
    requests_per_ep: tuple[int, ...]
    full_by_ep_phase: tuple[tuple[int, ...], ...]
    chunk_by_ep_phase: tuple[tuple[int, ...], ...]


@lru_cache(maxsize=128, typed=True)
def staggered_kv_token_series(batch_size: int, ep: int, input_tokens: int,
                              output_tokens: int, chunk_size: int = 0) -> StaggeredKVTokenSeries:
    """Return all phases in O(B + EP*L*(1 + ceil(L/C))) integer work.

    ``L = output_tokens - 1``; starting age of stable slot ``s`` is
    ``floor(s*L/B)`` and its EP owner is ``s % EP``. At phase ``t`` its age is
    ``(starting_age + t) % L``. Full cache tokens after writing are P+age+1;
    aligned-chunk tokens are (P+age)%C+1, INCLUDING a complete C-token chunk
    just before eviction. No B-by-L matrix or individual-token simulation is
    built. The age histogram is cyclically indexed without rotating arrays.
    """
    _integer("batch_size", batch_size, minimum=1)
    _integer("ep", ep, minimum=1)
    _integer("input_tokens", input_tokens, minimum=1)
    _integer("output_tokens", output_tokens, minimum=2)
    _integer("chunk_size", chunk_size, minimum=0)
    life = output_tokens - 1
    histograms = [[0] * life for _ in range(ep)]
    initial_full = [0] * ep
    initial_chunk = [0] * ep
    local_counts = [0] * ep
    for slot in range(batch_size):
        rank, age = slot % ep, slot * life // batch_size
        histograms[rank][age] += 1
        local_counts[rank] += 1
        initial_full[rank] += input_tokens + age + 1
        if chunk_size:
            initial_chunk[rank] += (input_tokens + age) % chunk_size + 1

    # A normal age increment advances chunk occupancy by +1, except the
    # C-1 -> 0 chunk boundaries. The lifetime wrap L-1 -> 0 has its own exact
    # delta, so exclude age L-1 from the normal chunk-boundary correction.
    boundaries = ()
    chunk_wrap_correction = 0
    if chunk_size:
        first_boundary = (chunk_size - 1 - input_tokens) % chunk_size
        boundaries = tuple(range(first_boundary, life - 1, chunk_size))
        first_value = input_tokens % chunk_size + 1
        last_value = (input_tokens + life - 1) % chunk_size + 1
        chunk_wrap_correction = first_value - last_value - 1

    full_series, chunk_series = [], []
    for rank, histogram in enumerate(histograms):
        full, chunk = initial_full[rank], initial_chunk[rank]
        full_values, chunk_values = [full], [chunk]
        count = local_counts[rank]
        for phase in range(1, life):
            # Previous phase t=phase-1 has age L-1 in histogram bucket
            # (L-1-t)%L = L-phase. Those slots wrap to age zero now.
            wraps = histogram[life - phase]
            full += count - life * wraps
            if chunk_size:
                crossings = sum(histogram[(boundary - phase + 1) % life] for boundary in boundaries)
                chunk += count + chunk_wrap_correction * wraps - chunk_size * crossings
            full_values.append(full)
            chunk_values.append(chunk)
        full_series.append(tuple(full_values))
        chunk_series.append(tuple(chunk_values))
    return StaggeredKVTokenSeries(batch_size, ep, input_tokens, output_tokens, chunk_size, life,
                                  tuple(local_counts), tuple(full_series), tuple(chunk_series))
