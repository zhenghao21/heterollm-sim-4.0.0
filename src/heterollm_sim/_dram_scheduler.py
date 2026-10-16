"""Bounded, online request scheduling above the DRAM command controller.

This schedules whole AccessRequests, not individual burst commands. Only
arrived requests in a finite FIFO window compete. No request is executed to
score a candidate, and no future arrival is preferred over ready work.
"""
from dataclasses import replace

from .memory_mapping import map_dram_address
from .memory_types import Operation


class _RangeClock:
    """Sparse range maxima; O(log capacity) update/query without page scans."""
    def __init__(self, size):
        self.size = size
        self.nodes = {}

    def update(self, left, right, value):
        def visit(node, lo, hi):
            if right <= lo or hi <= left:
                return
            own, total = self.nodes.get(node, (0.0, 0.0))
            if left <= lo and hi <= right:
                own = max(own, value)
            else:
                mid = (lo + hi) // 2
                visit(node * 2, lo, mid)
                visit(node * 2 + 1, mid, hi)
            total = max(own, self.nodes.get(node * 2, (0, 0))[1],
                        self.nodes.get(node * 2 + 1, (0, 0))[1])
            self.nodes[node] = (own, total)
        visit(1, 0, self.size)

    def query(self, left, right):
        def visit(node, lo, hi):
            if right <= lo or hi <= left or node not in self.nodes:
                return 0.0
            own, total = self.nodes[node]
            if left <= lo and hi <= right:
                return total
            mid = (lo + hi) // 2
            return max(own, visit(node * 2, lo, mid), visit(node * 2 + 1, mid, hi))
        return visit(1, 0, self.size)


def schedule_requests(core, requests):
    """Return input-ordered results after bounded legal issue reordering."""
    c = core.config
    burst = c.burst_bytes
    reads_and_writes = _RangeClock(c.effective_capacity_bytes // burst)
    writes = _RangeClock(c.effective_capacity_bytes // burst)
    results = [None] * len(requests)
    pending = []
    cursor = 0
    now = core._acceptance_ns
    while pending or cursor < len(requests):
        while cursor < len(requests) and len(pending) < c.scheduler_window:
            request = requests[cursor]
            pending.append([cursor, request, request.address // burst,
                (request.address + request.byte_count - 1) // burst + 1, 0])
            cursor += 1
        candidates = []
        for position, entry in enumerate(pending):
            index, request, left, right, _age = entry
            if any(other[2] < right and left < other[3]
                   and (request.operation is Operation.WRITE or other[1].operation is Operation.WRITE)
                   for other in pending[:position]):
                continue
            ledger = reads_and_writes if request.operation is Operation.WRITE else writes
            ready = core._admission_time(request, max(now, ledger.query(left, right)))
            candidates.append((entry, ready))
        # The oldest pending request always has no pending predecessor.
        now = max(now, min(ready for _entry, ready in candidates))
        available = [entry for entry, ready in candidates if ready <= now]
        aged = [entry for entry in available if entry[4] >= c.scheduler_window]
        if aged:
            selected = aged[0]
        elif c.scheduler == "row_hit_first":
            def hit(entry):
                mapping = map_dram_address(c, entry[1].address)
                row = (mapping.row if c.control_mode != "jedec_hbm" or c.activated_row_msb is None
                       else mapping.row & ~(1 << c.activated_row_msb))
                bank = core._banks.get(core._bank_id(mapping))
                return bank is not None and bank.open_row == row
            selected = next((entry for entry in available if hit(entry)), available[0])
        else:
            same = [entry for entry in available if entry[1].operation.value == core._scheduler_direction]
            opposite = [entry for entry in available if entry[1].operation.value != core._scheduler_direction]
            chosen = (opposite if core._scheduler_streak >= c.scheduler_window and opposite
                      else same or available)
            selected = chosen[0]
        index, request, left, right, _age = selected
        reordered = int(selected is not pending[0])
        for entry in pending:
            if entry[0] < index:
                entry[4] += 1
        pending.remove(selected)
        result = core.submit(replace(request, arrival_ns=max(request.arrival_ns, now)))
        result = replace(result, arrival_ns=request.arrival_ns, counters={
            **result.counters,
            "queue_wait_ns": result.queue_wait_ns + max(0.0, now - request.arrival_ns),
            "scheduler_reordered_requests": reordered,
        })
        results[index] = result
        reads_and_writes.update(left, right, result.completion_ns)
        if request.operation is Operation.WRITE:
            writes.update(left, right, result.completion_ns)
        now = max(now, core._acceptance_ns)
    return tuple(results)
