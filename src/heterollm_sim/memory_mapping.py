"""Deterministic address mapping and request splitting."""
from __future__ import annotations

from typing import Iterable, List

from .memory_types import AccessRequest, AddressMapping, DramConfig, NandConfig, Operation, Segment


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def _check_address(address: int, capacity: int) -> None:
    if address < 0 or address >= capacity:
        raise ValueError(f"address {address} is outside modeled capacity {capacity}")


def map_dram_address(config: DramConfig, address: int) -> AddressMapping:
    """Map a byte address to a DRAM lane, bank, row and column."""
    if not isinstance(config, DramConfig):
        raise TypeError("config must be DramConfig")
    if not isinstance(address, int) or isinstance(address, bool):
        raise TypeError("address must be an integer")
    _check_address(address, config.effective_capacity_bytes)
    interleave = config.interleave_bytes or config.burst_bytes
    per_die = config.lane_count * config.bank_count * config.rows_per_bank * config.row_bytes
    per_stack = per_die * config.dies_per_stack
    stack, stack_local = divmod(address, per_stack)
    die, local = divmod(stack_local, per_die)
    interleave_bursts = interleave // config.burst_bytes
    interleave_block, inner_burst = divmod(
        local // config.burst_bytes, interleave_bursts
    )
    lane = interleave_block % config.lane_count
    bank_linear = interleave_block // config.lane_count
    bank = bank_linear % config.bank_count
    row_column = bank_linear * interleave_bursts + inner_burst
    bursts_per_row = config.row_bytes // config.burst_bytes
    column = row_column % bursts_per_row
    row = (row_column // bursts_per_row) % config.rows_per_bank
    # Decode the flat lane while allowing data_lanes to describe an arbitrary
    # package connection.  Coordinates are descriptive; lane remains the
    # authoritative data resource id.
    rem = lane
    pseudo = rem % config.pseudo_channels_per_channel
    rem //= config.pseudo_channels_per_channel
    subchannel = rem % config.subchannels_per_channel
    rem //= config.subchannels_per_channel
    channel = rem % config.channels
    # ``stack`` was decoded above from capacity; lane coordinates remain
    # interface coordinates and therefore do not multiply with stack count.
    rank = bank // (config.bank_groups_per_rank * config.banks_per_group)
    bank_in_rank = bank % (config.bank_groups_per_rank * config.banks_per_group)
    bank_group = bank_in_rank // config.banks_per_group
    bank_id = bank_in_rank % config.banks_per_group
    return AddressMapping(
        address=address, stack=stack, die=die, lane=lane, channel=channel, subchannel=subchannel,
        pseudo_channel=pseudo, rank=rank, bank_group=bank_group,
        bank=bank_id, row=row, column=column,
    )


def map_nand_address(config: NandConfig, address: int) -> AddressMapping:
    """Map a byte address to a NAND array unit and page coordinates."""
    if not isinstance(config, NandConfig):
        raise TypeError("config must be NandConfig")
    if not isinstance(address, int) or isinstance(address, bool):
        raise TypeError("address must be an integer")
    _check_address(address, config.effective_capacity_bytes)
    page_index, page_offset = divmod(address, config.page_bytes)
    base_units = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
    base_unit = page_index % base_units
    if config.planes_independent:
        plane = (page_index // base_units) % config.planes_per_lun
        unit = base_unit + plane * base_units
        physical_page = page_index // (base_units * config.planes_per_lun)
    else:
        unit = base_unit
        physical_page = page_index // base_units
        plane = physical_page % config.planes_per_lun
        physical_page //= config.planes_per_lun
    page = physical_page % config.pages_per_block
    block = physical_page // config.pages_per_block
    # A flat index is decoded for reporting.  If planes are shared, ``unit``
    # intentionally omits the plane and therefore those planes contend.
    base = base_unit
    lun = base % config.luns_per_die
    base //= config.luns_per_die
    die = base % config.dies_per_target
    base //= config.dies_per_target
    target = base % config.targets_per_channel
    channel = base // config.targets_per_channel
    return AddressMapping(
        address=address, channel=channel, target=target, die=die, lun=lun,
        plane=plane, block=block, page=page, page_offset=page_offset,
        unit=unit,
    )


def split_dram_request(request: AccessRequest, config: DramConfig) -> List[Segment]:
    if request.operation is Operation.ERASE:
        raise ValueError("DRAM does not support erase")
    start, end = request.address, request.address + request.byte_count
    _check_address(start, config.effective_capacity_bytes)
    if end > config.effective_capacity_bytes:
        raise ValueError("request extends beyond modeled DRAM capacity")
    first = start // config.burst_bytes
    last = (end - 1) // config.burst_bytes
    if last - first + 1 > config.max_expanded_segments:
        raise ValueError("DRAM request exceeds max_expanded_segments; split it upstream")
    segments: List[Segment] = []
    for burst in range(first, last + 1):
        burst_start = burst * config.burst_bytes
        logical_start = max(start, burst_start)
        logical_end = min(end, burst_start + config.burst_bytes)
        logical = logical_end - logical_start
        mapping = map_dram_address(config, burst_start)
        segments.append(Segment(
            request_id=request.request_id, operation=request.operation,
            address=burst_start, logical_bytes=logical,
            transfer_bytes=config.burst_bytes, mapping=mapping,
        ))
    return segments


def split_nand_request(request: AccessRequest, config: NandConfig) -> List[Segment]:
    start, end = request.address, request.address + request.byte_count
    _check_address(start, config.effective_capacity_bytes)
    if end > config.effective_capacity_bytes:
        raise ValueError("request extends beyond modeled NAND capacity")
    if request.operation is Operation.ERASE:
        # Erase follows the same striped physical mapping as page access.
        # A byte range can touch one block in each interleaved LUN/plane, so
        # deduplicate by the full physical block coordinate rather than by a
        # linear logical block number.
        page = config.page_bytes
        first_page = start // page
        last_page = (end - 1) // page
        if last_page - first_page + 1 > config.max_expanded_segments:
            raise ValueError("NAND erase exceeds max_expanded_segments; split it upstream")
        blocks = {}
        for page_index in range(first_page, last_page + 1):
            address = page_index * page
            mapping = map_nand_address(config, address)
            key = (mapping.channel, mapping.target, mapping.die,
                   mapping.lun, mapping.plane, mapping.block)
            blocks.setdefault(key, (address, mapping))
        return [
            Segment(
                request_id=request.request_id,
                operation=request.operation,
                address=address,
                logical_bytes=0,
                transfer_bytes=0,
                mapping=mapping,
            )
            for address, mapping in blocks.values()
        ]
    first = start // config.page_bytes
    last = (end - 1) // config.page_bytes
    if last - first + 1 > config.max_expanded_segments:
        raise ValueError("NAND request exceeds max_expanded_segments; split it upstream")
    segments = []
    for page_index in range(first, last + 1):
        page_start = page_index * config.page_bytes
        logical_start = max(start, page_start)
        logical_end = min(end, page_start + config.page_bytes)
        logical = logical_end - logical_start
        if request.operation is Operation.WRITE and logical != config.page_bytes and config.partial_page_policy == "reject":
            raise ValueError("partial NAND page write rejected by configuration")
        host_bytes = _round_up(logical, config.host_granularity_bytes)
        segments.append(Segment(
            request_id=request.request_id, operation=request.operation,
            address=page_start, logical_bytes=logical,
            transfer_bytes=config.transfer_page_bytes,
            mapping=map_nand_address(config, page_start),
            host_transfer_bytes=host_bytes,
            internal_transfer_bytes=config.transfer_page_bytes,
        ))
    return segments


def map_address(config: DramConfig | NandConfig, address: int) -> AddressMapping:
    """Dispatch to the medium-specific deterministic mapping."""
    return map_dram_address(config, address) if isinstance(config, DramConfig) else map_nand_address(config, address)


def split_request(request: AccessRequest, config: DramConfig | NandConfig) -> List[Segment]:
    """Split at DRAM burst or NAND page/block boundaries."""
    if isinstance(config, DramConfig):
        return split_dram_request(request, config)
    if isinstance(config, NandConfig):
        return split_nand_request(request, config)
    raise TypeError("config must be DramConfig or NandConfig")


__all__ = [
    "map_address", "map_dram_address", "map_nand_address", "split_request",
    "split_dram_request", "split_nand_request",
]
