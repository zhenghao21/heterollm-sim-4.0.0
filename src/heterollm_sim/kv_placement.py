"""Host placement of NAND-backed KV blocks; OCP HBF 13.3.2 leaves it to the host.

The physical bank is the low address bits and a bank log gives any program the
bank's next page number, so a whole-request block striped over consecutive banks
writes and reads on the banks its address names: a tail's bank is its block's
first bank plus the pages it holds.

Every live tail gains one token per step, so all tails rotate round the banks
together and their relative positions never change. A block chosen against the
tails that will be live when a request joins decoding stays clear for the
request's whole life, with no length known and no program moved. The host starts
an import as a slot frees and fixes the step it will join, so it knows that
moment when it picks the block the import writes to.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import math
import random

import numpy as np

# Free blocks a pool keeps beyond its peak, so a pick has real choice: each live
# tail rules out a few positions of the ring, up to about 2/3 of them near the
# batches the SRAM admits.
KV_POOL_SPARE_BLOCKS = 32
# Stationary refill cycles replayed before a stationary workload's seeds take
# their blocks; one already leaves no two tails on a bank.
KV_SETTLE_GENERATIONS = 2


@lru_cache(maxsize=1024)
def kv_stripe_layout(prompt, output, per_page, banks, blocks, chunk=0):
    """Stride, K/V phase and pool blocks of NAND-backed KV, from the block size alone.

    The stride is the first from a block's pages that is coprime with the banks
    and spreads the pool's blocks round them, since each layer reads them all
    and a pick needs free blocks near every position. An import writes K and V
    of each layer in turn, so V's pool starts the history's pages after K's:
    the next pool starts where this one ended and the import rolls round the
    banks, one program at a time on each.
    """
    pool = blocks + max(KV_POOL_SPARE_BLOCKS, blocks // 8)
    pages = -(-(prompt + output - 1) // per_page)
    widest = 2 * -(-banks // pool)

    def spreads(stride):
        first = np.sort(np.arange(pool, dtype=np.int64) * stride % banks)
        return max(int(np.diff(first).max(initial=0)), int(first[0] + banks - first[-1])) <= widest

    stride = next((stride for stride in range(pages, pages + banks)
                   if math.gcd(stride, banks) == 1 and spreads(stride)), None)
    if stride is None:
        raise ValueError("no KV stripe stride spreads the pool over the banks")
    history = -(-min(prompt - 1, chunk or prompt - 1) // per_page) % banks or 1
    return {"stride_pages": stride, "phase_pages": (history - pool * stride) % banks, "pool_blocks": pool}


class KVTailRing:
    """The tails of one KV pool, as token positions on a ring of banks * per_page.

    X = block * stride * per_page + rows, with ``rows`` taken at one fixed step
    (a request joining later counts as younger, so below ``prompt``). Two tails
    can share a bank at some step iff their X differ by less than per_page; they
    fill a page on one bank in one step iff their X are equal. V's tail sits
    ``shift`` pages after K's, so each pair is checked at 0 and at +-shift. A
    chunked layer's tail wraps at ``held`` rows, so also a chunk away.
    """
    __slots__ = ("banks", "per_page", "length", "stride", "shift", "held", "tails", "centres")

    def __init__(self, banks, per_page, stride, shift, held):
        self.banks, self.per_page = banks, per_page
        self.length = banks * per_page
        self.stride = stride * per_page % self.length
        self.shift = shift
        self.held = held
        self.tails = np.zeros(self.length, dtype=np.int64)
        offset = shift * per_page % self.length
        # K against K and V against V, K against V, V against K.
        self.centres = [(0, 2), (offset, 1), (-offset, 1)]
        if held:
            self.centres = [(centre + jump, weight) for centre, weight in self.centres for jump in (0, held, -held)]

    def copy(self):
        ring = KVTailRing.__new__(KVTailRing)
        ring.banks, ring.per_page, ring.length = self.banks, self.per_page, self.length
        ring.stride, ring.shift, ring.held, ring.centres = self.stride, self.shift, self.held, self.centres
        ring.tails = self.tails.copy()
        return ring

    def position(self, block, rows):
        return (block * self.stride + (rows % self.held if self.held else rows)) % self.length

    def add(self, position, sign=1):
        self.tails[position] += sign

    def pick(self, options, rows):
        """Index in ``options`` of the block with the fewest same-step page fills, then the fewest
        shared banks; the earliest on a tie."""
        positions = self.position(np.asarray(options, dtype=np.int64), rows)
        tails, length, reach = self.tails, self.length, self.per_page - 1
        total = np.concatenate(([0], np.cumsum(np.tile(tails, 3))))
        fills, shared = np.zeros(len(options), dtype=np.int64), np.zeros(len(options), dtype=np.int64)
        for centre, weight in self.centres:
            at = (positions + centre) % length
            fills += weight * tails[at]
            shared += weight * (total[at + length + reach + 1] - total[at + length - reach])
        return int(np.lexsort((shared, fills))[0])

    def sharing(self):
        """(most K and V tails on one bank, tails on a shared bank, tails) at the fixed step."""
        held = np.flatnonzero(self.tails)
        keys = np.bincount(held // self.per_page % self.banks, weights=self.tails[held], minlength=self.banks).astype(np.int64)
        banks = keys + np.roll(keys, self.shift)
        return int(banks.max()), int(banks[banks > 1].sum()), int(banks.sum())


@lru_cache(maxsize=256)
def settled_seed_blocks(ages, prompt, output, blocks, geometry, prepared):
    """The free list the host's rule leaves in a long stationary run: seeds' blocks in ``ages``
    order (youngest first), then the rest.

    Requests finish oldest first. Each freed block goes to the list's tail, and
    the import that starts then picks from what is free for the slot it will
    land in, ``prepared`` landings later, with the tails of the running requests
    and the other imports in view.
    """
    ring = KVTailRing(*geometry)
    order = list(range(blocks))
    # A long run's free list is scattered, whatever order the pool started in.
    random.Random(0).shuffle(order)
    free = deque(order)
    slots = deque(prompt + age + KV_SETTLE_GENERATIONS * output for age in reversed(ages))
    held = deque()
    for rows in slots:
        block = free.popleft()
        position = ring.position(block, rows)
        ring.add(position)
        held.append((block, position))

    def admit():
        rows = slots.popleft() - output
        slots.append(rows)
        options = np.fromiter(free, dtype=np.int64, count=len(free))
        pick = ring.pick(options, rows)
        block = free[pick]
        del free[pick]
        position = ring.position(block, rows)
        ring.add(position)
        held.append((block, position))

    for _ in range(min(prepared, blocks - len(ages))):
        admit()
    for _ in range(KV_SETTLE_GENERATIONS * len(ages)):
        block, position = held.popleft()
        ring.add(position, -1)
        free.append(block)
        admit()
    seeds = [held.popleft()[0] for _ in ages]
    # Imports in flight here re-pick at the run's first boundary.
    return (*reversed(seeds), *free, *(block for block, _position in held))
