"""Sequential host placement of NAND-backed KV blocks.

OCP HBF v0.7 leaves placement to the host.  The host allocates whole KV blocks
in cursor order and records arithmetic page-table runs.  The shared NAND
mapper turns each logical page into its channel, local DLU, plane/bank, page
and block without a stripe phase or a synthetic block-open rate term.
"""
from __future__ import annotations

from functools import lru_cache


def sequential_page_table_runs(page_count: int, channels: int):
    """Return one compact arithmetic run for sequential channel placement.

    The logical page cursor advances by one for every 4 KiB page.  The host
    channel is ``cursor % channels`` and its local DLU is ``cursor //
    channels``; storing the arithmetic progression avoids a per-page table.
    The tuple is ``(logical_start, count, channel_start, dlu_start, stride)``.
    """
    if type(page_count) is not int or page_count < 0:
        raise ValueError("page_count must be a non-negative integer")
    if type(channels) is not int or channels <= 0:
        raise ValueError("channels must be a positive integer")
    if not page_count:
        return ()
    return ((0, page_count, 0, 0, channels),)


@lru_cache(maxsize=1024)
def kv_sequential_layout(prompt, output, per_page, channels, planes, blocks,
                         pages_per_block=256):
    """Describe the OCP host's sequential KV placement and compact page map.

    No bank stripe, phase or block-open rate term is introduced.  A pool owns
    whole logical blocks; all pages are allocated in cursor order and decoded
    by the shared NAND mapping rule.  ``page_table_runs`` is sufficient to
    reconstruct every page's channel/local-DLU pair during replay.
    """
    values = (prompt, output, per_page, channels, planes, blocks, pages_per_block)
    if any(type(value) is not int for value in values):
        raise ValueError("sequential KV layout arguments must be integers")
    if prompt < 0 or output < 0 or per_page <= 0 or channels <= 0 or planes <= 0 or blocks < 0 or pages_per_block <= 0:
        raise ValueError("invalid sequential KV layout dimensions")
    pages = (prompt + output + per_page - 1) // per_page
    pool_pages = blocks * pages_per_block
    return {
        "placement_mode": "sequential_channel",
        "pool_blocks": blocks,
        "pool_pages": pool_pages,
        "pages_per_block": pages_per_block,
        "channels": channels,
        "planes": planes,
        "request_pages": pages,
        "page_table_runs": sequential_page_table_runs(pool_pages, channels),
    }
