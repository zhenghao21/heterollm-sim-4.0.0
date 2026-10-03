"""Small analytical cold-page model for opt-in NAND media endpoints."""
import math
from numbers import Real
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from .memory_service import realtime_memory_metrics

_REQ = {"version", "host_transaction_bytes", "host_max_request_bytes", "media_page_bytes", "command_queue_depth", "media_parallelism", "page_read_latency_ns", "page_program_latency_ns", "access_pattern"}
_ALLOWED = _REQ | {
    "physical_planes", "erase_block_bytes", "erase_latency_ns",
    "background_work", "physical_dies", "physical_channels",
    "program_order",
    "source", "parameter_evidence",
}
_PATTERNS = {"contiguous_page_aligned", "unknown_alignment_conservative"}
_EVIDENCE_STATUSES = {"parameterized", "sourced", "unknown"}
_PARAMETER_KEYS = _REQ | {
    "physical_planes", "erase_block_bytes", "erase_latency_ns",
    "background_work", "physical_dies", "physical_channels",
    "program_order",
}

def _num(value, name, positive=False, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or (positive and value <= 0)
            or (nonnegative and value < 0)):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return value

def _media_contract(component):
    metadata = getattr(component, "metadata", {})
    contract = metadata.get("nand_media")
    field_name = "nand_media"
    if contract is None:
        contract = metadata.get("hbf_media")
        field_name = "hbf_media"
    return contract, field_name


def _parameter_evidence(contract):
    """Return provenance without implying an unverified device measurement."""
    source = contract.get("source")
    supplied = contract.get("parameter_evidence", {})
    result = {}
    for name in sorted(_PARAMETER_KEYS):
        present = name in contract
        value = contract.get(name, "unknown")
        item = supplied.get(name, {})
        status = (
            item.get("status", "parameterized")
            if isinstance(item, Mapping) and item
            else ("parameterized" if present else "unknown")
        )
        entry = {"value": value, "status": status}
        item_source = item.get("source") if isinstance(item, Mapping) else None
        if item_source is not None:
            entry["source"] = item_source
        elif source is not None:
            entry["source"] = source
        result[name] = entry
    return result


def validate_nand_media(component):
    contract, field_name = _media_contract(component)
    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("{} must be a mapping".format(field_name))
    unknown = set(contract) - _ALLOWED
    if unknown:
        raise ValueError(f"unknown {field_name} keys: {sorted(unknown)}")
    missing = _REQ - set(contract)
    if missing:
        raise ValueError(f"missing {field_name} keys: {sorted(missing)}")
    version = contract["version"]
    if version not in {"cold_page_v1", "nand_media_v1"}:
        raise ValueError("unsupported NAND media contract version")
    if (version == "cold_page_v1"
            and (contract["host_transaction_bytes"] != 64
                 or contract["media_page_bytes"] != 4096)):
        raise ValueError("unsupported cold_page_v1 granularity")
    for key in ("host_transaction_bytes", "host_max_request_bytes", "media_page_bytes", "command_queue_depth", "media_parallelism"):
        if isinstance(contract[key], bool) or not isinstance(contract[key], int) or contract[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (contract["host_max_request_bytes"] < contract["host_transaction_bytes"]
            or contract["host_max_request_bytes"] > contract["media_page_bytes"]
            or contract["host_max_request_bytes"] % contract["host_transaction_bytes"]):
        raise ValueError("host_max_request_bytes must be a host-transaction multiple no larger than one media page")
    qd = contract["command_queue_depth"]
    if version == "cold_page_v1" and (not 256 <= qd <= 16384 or qd & (qd - 1)):
        raise ValueError("command_queue_depth must be a power of two in [256, 16384]")
    for key in ("page_read_latency_ns", "page_program_latency_ns"):
        _num(contract[key], key, True)
    if contract["access_pattern"] not in _PATTERNS:
        raise ValueError("unsupported access_pattern")
    if "physical_planes" in contract and (isinstance(contract["physical_planes"], bool) or not isinstance(contract["physical_planes"], int) or contract["physical_planes"] <= 0):
        raise ValueError("physical_planes must be a positive integer")
    if "physical_dies" in contract and (isinstance(contract["physical_dies"], bool) or not isinstance(contract["physical_dies"], int) or contract["physical_dies"] <= 0):
        raise ValueError("physical_dies must be a positive integer")
    if "physical_channels" in contract and (isinstance(contract["physical_channels"], bool) or not isinstance(contract["physical_channels"], int) or contract["physical_channels"] <= 0):
        raise ValueError("physical_channels must be a positive integer")
    has_erase_geometry = "erase_block_bytes" in contract or "erase_latency_ns" in contract
    if has_erase_geometry and not {"erase_block_bytes", "erase_latency_ns"}.issubset(contract):
        raise ValueError("erase_block_bytes and erase_latency_ns must be supplied together")
    if "erase_block_bytes" in contract and (isinstance(contract["erase_block_bytes"], bool) or not isinstance(contract["erase_block_bytes"], int) or contract["erase_block_bytes"] <= 0):
        raise ValueError("erase_block_bytes must be a positive integer")
    if "erase_latency_ns" in contract:
        _num(contract["erase_latency_ns"], "erase_latency_ns", True)
    if "background_work" in contract and not isinstance(contract["background_work"], Mapping):
        raise ValueError("background_work must be a mapping when supplied")
    if "program_order" in contract and contract["program_order"] not in {
        "sequential", "unspecified", "unknown",
    }:
        raise ValueError("program_order must be sequential, unspecified or unknown")
    if "source" in contract and (
        not isinstance(contract["source"], str) or not contract["source"].strip()
    ):
        raise ValueError("source must be non-empty text when supplied")
    evidence = contract.get("parameter_evidence", {})
    if not isinstance(evidence, Mapping):
        raise ValueError("parameter_evidence must be a mapping when supplied")
    unknown_evidence = set(evidence) - (_PARAMETER_KEYS | {"memory_access_offset_bytes"})
    if unknown_evidence:
        raise ValueError(
            "unknown parameter_evidence keys: {}".format(sorted(unknown_evidence))
        )
    for name, item in evidence.items():
        if not isinstance(item, Mapping):
            raise ValueError("parameter_evidence[{}] must be a mapping".format(name))
        status = item.get("status", "parameterized")
        if status not in _EVIDENCE_STATUSES:
            raise ValueError(
                "parameter_evidence[{}].status must be parameterized, sourced or unknown".format(name)
            )
        if name in _PARAMETER_KEYS and name not in contract and status != "unknown":
            raise ValueError(
                "parameter_evidence[{}] cannot assert an omitted parameter".format(name)
            )
        item_source = item.get("source")
        if item_source is not None and (
            not isinstance(item_source, str) or not item_source.strip()
        ):
            raise ValueError("parameter_evidence[{}].source must be non-empty text".format(name))
        if status == "sourced" and not (item_source or contract.get("source")):
            raise ValueError(
                "parameter_evidence[{}] sourced status requires source".format(name)
            )
    if "erase_block_bytes" in contract and contract["erase_block_bytes"] % contract["media_page_bytes"]:
        raise ValueError("erase_block_bytes must be a media_page_bytes multiple")


def validate_hbf_media(component):
    """Compatibility validator for the legacy HBF contract name."""

    return validate_nand_media(component)

def _ceil(value, unit):
    return (value + unit - 1) // unit


def _topology(c):
    """Return explicit NAND dimensions and their bounded media slots.

    ``media_parallelism`` remains the controller-side cap.  Explicit
    channel/die/plane counts add physical slots; omitted dimensions stay
    unknown and therefore do not invent an additional cap.
    """
    dimensions = {
        name: c.get(name)
        for name in ("physical_channels", "physical_dies", "physical_planes")
    }
    known = [value for value in dimensions.values() if value is not None]
    slots = 1
    for value in known:
        slots *= value
    return (
        dimensions,
        slots,
        "explicit_channel_die_plane_product"
        if known else "controller_parallelism_only_unknown_topology",
    )


def _address_mapping(c, page_offset_bytes, pages, page):
    dimensions, _slots, _basis = _topology(c)
    known = page_offset_bytes is not None
    start_page = page_offset_bytes // page if known else None
    if not known or not pages:
        return {
            "address_mapping": "unknown_offset" if not known else "empty",
            "first_page": start_page,
            "last_page": (start_page + pages - 1) if start_page is not None else None,
            "mapped_channels": "unknown" if dimensions["physical_channels"] is None else 0,
            "mapped_dies": "unknown" if dimensions["physical_dies"] is None else 0,
            "mapped_planes": "unknown" if dimensions["physical_planes"] is None else 0,
            "mapping_basis": "page_index_modulo_channel_die_plane" if known else "unknown_page_offset",
        }
    channels = dimensions["physical_channels"]
    dies = dimensions["physical_dies"]
    planes = dimensions["physical_planes"]
    mapped = {"channels": set(), "dies": set(), "planes": set()}
    for index in range(start_page, start_page + pages):
        if channels is not None:
            mapped["channels"].add(index % channels)
        if dies is not None:
            mapped["dies"].add((index // (channels or 1)) % dies)
        if planes is not None:
            mapped["planes"].add((index // ((channels or 1) * (dies or 1))) % planes)
    return {
        "address_mapping": "known_page_index",
        "first_page": start_page,
        "last_page": start_page + pages - 1,
        "mapped_channels": sorted(mapped["channels"]) if channels is not None else "unknown",
        "mapped_dies": sorted(mapped["dies"]) if dies is not None else "unknown",
        "mapped_planes": sorted(mapped["planes"]) if planes is not None else "unknown",
        "mapping_basis": "page_index_modulo_channel_die_plane",
    }


@dataclass(frozen=True)
class NANDQueueState:
    """Immutable cross-request readiness for explicit NAND media slots."""

    ready_ns: Mapping[str, float] = field(default_factory=dict)


def _queue_coordinate(c, page_index):
    channels = c["physical_channels"]
    dies = c["physical_dies"]
    planes = c["physical_planes"]
    channel = page_index % channels
    die = (page_index // channels) % dies
    plane = (page_index // (channels * dies)) % planes
    return "channel={};die={};plane={}".format(channel, die, plane)


def nand_media_service_batch(component, requests, *, start_ns=0.0, state=None):
    """Price a small NAND batch with cross-request slot queues.

    This is intentionally a queue envelope, not an FTL.  It requires explicit
    channel/die/plane dimensions and page offsets so an unknown physical route
    cannot be silently scheduled onto a made-up owner.
    """
    validate_nand_media(component)
    contract, _field_name = _media_contract(component)
    if any(
        contract.get(name) is None
        for name in ("physical_channels", "physical_dies", "physical_planes")
    ):
        raise ValueError(
            "nand queue scheduling requires physical_channels, physical_dies "
            "and physical_planes"
        )
    if isinstance(start_ns, bool) or not isinstance(start_ns, Real) or not math.isfinite(float(start_ns)) or start_ns < 0:
        raise ValueError("start_ns must be a finite non-negative number")
    if state is None:
        state = NANDQueueState()
    if not isinstance(state, NANDQueueState):
        raise TypeError("state must be NANDQueueState or None")
    if not isinstance(requests, (tuple, list)):
        raise TypeError("requests must be a tuple or list")
    ready = {str(key): float(value) for key, value in state.ready_ns.items()}
    for value in ready.values():
        if not math.isfinite(value) or value < 0:
            raise ValueError("NANDQueueState.ready_ns values must be finite non-negative numbers")
    cursor = float(start_ns)
    batch_end = cursor
    rows = []
    totals = {"logical_bytes": 0, "physical_bytes": 0, "service_ns": 0.0, "queue_wait_ns": 0.0}
    operation_counts = {}
    for request in requests:
        if not isinstance(request, Mapping):
            raise ValueError("NAND queue request must be a mapping")
        operation = str(request.get("operation", "")).lower()
        if operation not in {"read", "program", "erase"}:
            raise ValueError("NAND queue operation must be read, program or erase")
        count = request.get("byte_count")
        offset = request.get("page_offset_bytes")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("NAND queue byte_count must be a positive integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("NAND queue page_offset_bytes must be a non-negative integer")
        bill = nand_media_service(
            component,
            count,
            operation == "read",
            page_offset_bytes=offset,
            operation=operation,
        )
        page = contract["media_page_bytes"]
        if operation == "erase":
            block = contract.get("erase_block_bytes")
            if block is None:
                raise ValueError("NAND queue erase requires erase_block_bytes")
            first = offset // block
            slot_count = bill["erase_operations"]
            pages_per_block = block // page
            coordinates = [
                _queue_coordinate(contract, first * pages_per_block + index * pages_per_block)
                for index in range(slot_count)
            ]
        else:
            first = offset // page
            slot_count = bill["pages_touched"]
            coordinates = [_queue_coordinate(contract, first + index) for index in range(slot_count)]
        # A request touching several pages waits for the slowest selected slot.
        wait = max((ready.get(key, 0.0) - cursor for key in set(coordinates)), default=0.0)
        wait = max(0.0, wait)
        begin = cursor + wait
        end = begin + float(bill["service_ns"])
        for key in set(coordinates):
            ready[key] = end
        row = dict(bill)
        row.update({
            "queue_wait_ns": wait,
            "queue_start_ns": begin,
            "queue_end_ns": end,
            "queue_resources": tuple(sorted(set(coordinates))),
        })
        rows.append(row)
        totals["logical_bytes"] += count
        totals["physical_bytes"] += int(bill.get("physical_bytes", 0))
        totals["service_ns"] += float(bill["service_ns"])
        totals["queue_wait_ns"] += wait
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
        batch_end = max(batch_end, end)
    return {
        "model": "nand_media_queue_v1",
        "requests": tuple(rows),
        "request_count": len(rows),
        "logical_bytes": totals["logical_bytes"],
        "physical_bytes": totals["physical_bytes"],
        "service_ns": totals["service_ns"],
        "queue_wait_ns": totals["queue_wait_ns"],
        "end_ns": batch_end,
        "operation_counts": dict(sorted(operation_counts.items())),
        "queue_resources": tuple(sorted(ready)),
        "physical_owner": str(
            getattr(component, "metadata", {}).get("physical_owner")
            or getattr(component, "component_id", "unknown")
        ),
        "physical_resource_id": str(
            getattr(component, "metadata", {}).get("memory_resource_id")
            or getattr(component, "metadata", {}).get("physical_owner")
            or getattr(component, "component_id", "unknown")
        ),
        "queue_depth": contract["command_queue_depth"],
        "topology": {
            name: contract[name]
            for name in ("physical_channels", "physical_dies", "physical_planes")
        },
        "profile_evidence": "analytical/no hardware validated",
    }, NANDQueueState(ready_ns=dict(sorted(ready.items())))


def resolve_nand_task(task, states, *, start_ns=0.0):
    """Preview a queued endpoint without committing event-kernel state."""
    contract = task.metadata.get("nand_access")
    if not isinstance(contract, Mapping):
        return task, None, None
    component = contract.get("component")
    if not isinstance(component, Mapping):
        raise ValueError("nand_access.component must be a mapping")
    from types import SimpleNamespace
    component = SimpleNamespace(**component)
    resource_id = str(contract.get("resource_id") or "")
    state_key = str(contract.get("state_key") or "")
    if not resource_id or not state_key:
        raise ValueError("nand_access resource_id and state_key must be non-empty")
    if all(demand.resource_id != resource_id for demand in task.demands):
        raise ValueError("nand_access.resource_id must name a task demand")
    metrics, next_state = nand_media_service_batch(
        component,
        contract.get("requests"),
        start_ns=start_ns,
        state=states.get(state_key),
    )
    service_ns = max(0.0, metrics["end_ns"] - start_ns)
    return replace(
        task,
        demands=tuple(
            replace(demand, service_ns=service_ns)
            if demand.resource_id == resource_id else demand
            for demand in task.demands
        ),
        metadata={**task.metadata, "nand_execution": metrics},
    ), state_key, next_state


def nand_media_service(
    component,
    byte_count,
    read: bool = False,
    *,
    page_offset_bytes=None,
    operation=None,
    background_work=None,
) -> dict:
    validate_nand_media(component)
    c, field_name = _media_contract(component)
    if c is None:
        raise ValueError("nand_media or hbf_media contract is required")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError("byte_count must be a non-negative integer")
    if not isinstance(read, bool):
        raise ValueError("read must be bool")
    operation = ("read" if read else "program") if operation is None else str(operation).lower()
    if operation not in {"read", "program", "erase"}:
        raise ValueError("operation must be read, program or erase")
    if operation == "read" and not read:
        raise ValueError("read operation requires read=True")
    if operation == "program" and read:
        raise ValueError("program operation requires read=False")
    if page_offset_bytes is not None and (
        isinstance(page_offset_bytes, bool)
        or not isinstance(page_offset_bytes, int)
        or page_offset_bytes < 0
    ):
        raise ValueError("page_offset_bytes must be a non-negative integer")
    page, host_tx = c["media_page_bytes"], c["host_transaction_bytes"]
    if operation == "erase":
        block_bytes = c.get("erase_block_bytes")
        erase_latency = c.get("erase_latency_ns")
        if block_bytes is None or erase_latency is None:
            raise ValueError("erase requires erase_block_bytes and erase_latency_ns")
        known_offset = page_offset_bytes is not None
        block_offset = page_offset_bytes % block_bytes if known_offset else 0
        operations = _ceil(block_offset + byte_count, block_bytes) if byte_count else 0
        dimensions, topology_slots, topology_basis = _topology(c)
        declared_planes = dimensions["physical_planes"]
        parallel = min(
            c["media_parallelism"],
            topology_slots if topology_basis.startswith("explicit") else c["media_parallelism"],
            c["command_queue_depth"],
        )
        waves = _ceil(operations, parallel) if operations else 0
        physical = operations * block_bytes
        erase_ns = waves * erase_latency
        queue_wait_ns = max(0, waves - 1) * erase_latency
        marker = c.get("background_work") if background_work is None else background_work
        return {
            "version": c["version"], "contract_field": field_name,
            "operation": "erase",
            "address_scope": "known_block_offset" if known_offset else "block_parameterized",
            "block_offset_bytes": block_offset if known_offset else None,
            "host_transfer_bytes": 0, "physical_read_bytes": 0,
            "physical_write_bytes": physical, "physical_bytes": physical,
            "pages_touched": 0, "read_operations": 0, "program_operations": 0,
            "erase_operations": operations, "media_command_count": operations,
            "command_count": operations, "effective_parallelism": parallel,
            "media_waves": waves, "media_erase_service_ns": erase_ns,
            "queue_wait_ns": queue_wait_ns,
            "queue_saturated": bool(operations and operations > parallel),
            "service_ns": erase_ns, "background_work": marker or {},
            "physical_planes": declared_planes if declared_planes is not None else "unknown",
            "addressable_media_slots": topology_slots if topology_basis.startswith("explicit") else "unknown",
            "topology_basis": topology_basis,
            "parallelism_basis": (
                "declared_media_parallelism_and_physical_plane_cap"
                if declared_planes is not None
                else "declared_media_parallelism_without_physical_plane_cap"
            ),
            "physical_dies": c.get("physical_dies", "unknown"),
            "physical_channels": c.get("physical_channels", "unknown"),
            "physical_owner": str(
                getattr(component, "metadata", {}).get("physical_owner")
                or getattr(component, "component_id", "unknown")
            ),
            "physical_resource_id": str(
                getattr(component, "metadata", {}).get("memory_resource_id")
                or getattr(component, "metadata", {}).get("physical_owner")
                or getattr(component, "component_id", "unknown")
            ),
            "program_order": c.get("program_order", "unknown"),
            "write_completion": "block_erase_complete",
            "timing_diagnostics": "parameterized NAND block erase; no FTL/GC simulation",
            "profile_evidence": "analytical/no hardware validated",
            **({"source": c["source"]} if c.get("source") else {}),
            "parameter_evidence": {
                **_parameter_evidence(c),
                "memory_access_offset_bytes": {
                    "value": page_offset_bytes if known_offset else "unknown",
                    "status": "parameterized" if known_offset else "unknown",
                    **({"source": c["source"]} if c.get("source") else {}),
                },
            },
        }
    host_bytes = _ceil(byte_count, host_tx) * host_tx if byte_count else 0
    known_offset = page_offset_bytes is not None
    offset = (page_offset_bytes % page) if known_offset else 0
    nominal = _ceil(offset + byte_count, page) if byte_count else 0
    unknown = (
        c["access_pattern"] == "unknown_alignment_conservative"
        and not known_offset
    )
    # A 64B-aligned request can start at most 4096-64B before a page end.
    pages = (
        _ceil(byte_count + page - host_tx, page)
        if unknown and not known_offset and byte_count
        else nominal
    )
    if operation == "read" or not byte_count:
        rmw = 0
    elif known_offset:
        end_offset = offset + byte_count
        rmw = min(pages, int(offset > 0) + int(end_offset % page != 0))
    elif unknown:
        rmw = min(2, pages)
    else:
        rmw = int(bool(byte_count % page))
    physical_read = pages * page if read else rmw * page
    physical_write = 0 if read else pages * page
    read_ops, program_ops = (pages, 0) if read else (rmw, pages)
    host_command_count = _ceil(host_bytes, c["host_max_request_bytes"]) if host_bytes else 0
    if unknown and host_command_count:
        # Unknown alignment commands are page-bounded; never let one command
        # silently span the conservative page envelope.
        host_command_count = max(host_command_count, pages)
    media_command_count = read_ops + program_ops
    # Omitted planes means no additional physical bottleneck: the declared
    # media_parallelism remains effective. An explicit smaller plane count caps it.
    dimensions, topology_slots, topology_basis = _topology(c)
    declared_planes = dimensions["physical_planes"]
    parallel = min(
        c["media_parallelism"],
        topology_slots if topology_basis.startswith("explicit") else c["media_parallelism"],
        c["command_queue_depth"],
    )
    program_order = c.get("program_order", "unknown")
    program_parallel = 1 if program_order == "sequential" and program_ops else parallel
    rwaves = _ceil(read_ops, parallel) if read_ops else 0
    pwaves = _ceil(program_ops, program_parallel) if program_ops else 0
    direction = "read" if operation == "read" else "write"
    # A zero-byte request has no bandwidth requirement; otherwise each used
    # direction must have a finite positive physical-media cap.
    if host_bytes:
        host_bw = _num(getattr(component, f"{direction}_bandwidth_gbps", 0),
                       f"component.{direction}_bandwidth_gbps", True)
    else:
        host_bw = 1.0
    rbw = _num(getattr(component, "read_bandwidth_gbps", 0),
               "component.read_bandwidth_gbps", True) if physical_read else 1.0
    wbw = _num(getattr(component, "write_bandwidth_gbps", 0),
               "component.write_bandwidth_gbps", True) if physical_write else 1.0
    read_ns = rwaves * c["page_read_latency_ns"] + 8.0 * physical_read / rbw
    program_ns = pwaves * c["page_program_latency_ns"] + 8.0 * physical_write / wbw
    media_ns = read_ns if operation == "read" else read_ns + program_ns
    host_waves = _ceil(host_command_count, c["command_queue_depth"]) if host_command_count else 0
    host_ns = 8.0 * host_bytes / host_bw if host_bytes else 0.0
    host_queue_wait_ns = max(0, host_waves - 1) * (host_ns / host_waves if host_waves else 0.0)
    media_queue_wait_ns = (
        max(0, rwaves - 1) * c["page_read_latency_ns"]
        + max(0, pwaves - 1) * c["page_program_latency_ns"]
    )
    read_energy_rate = _num(getattr(component, "metadata", {}).get("read_energy_pj_per_byte", 0.0),
                             "read_energy_pj_per_byte", nonnegative=True)
    write_energy_rate = _num(getattr(component, "metadata", {}).get("write_energy_pj_per_byte", 0.0),
                              "write_energy_pj_per_byte", nonnegative=True)
    read_energy = physical_read * read_energy_rate
    write_energy = physical_write * write_energy_rate
    energy = read_energy + write_energy

    if media_ns > host_ns:
        bottleneck = "media"
    elif host_ns > media_ns:
        bottleneck = "host_endpoint"
    elif media_ns or host_ns:
        bottleneck = "host_and_media"
    else:
        bottleneck = "none"
    overall_metrics = realtime_memory_metrics(
        byte_count,
        media_ns + host_ns,
        physical_bytes=physical_read + physical_write,
        bandwidth_ceiling_gb_s=(
            host_bytes / host_ns if host_ns > 0 else 0.0
        ),
        request_window_utilization=(
            min(1.0, host_command_count / float(c["command_queue_depth"]))
            if host_command_count else 0.0
        ),
        bottleneck=bottleneck,
    )
    host_metrics = realtime_memory_metrics(
        host_bytes,
        host_ns,
        physical_bytes=host_bytes,
        bandwidth_ceiling_gb_s=(host_bytes / host_ns if host_ns > 0 else 0.0),
        request_window_utilization=(
            min(1.0, host_command_count / float(c["command_queue_depth"]))
            if host_command_count else 0.0
        ),
        bottleneck="host_endpoint" if host_ns else "none",
    )
    media_metrics = realtime_memory_metrics(
        byte_count,
        media_ns,
        physical_bytes=physical_read + physical_write,
        # Component fields are Gbit/s; the shared helper reports GB/s.
        bandwidth_ceiling_gb_s=(rbw if read else wbw) / 8.0,
        request_window_utilization=(
            min(1.0, media_command_count / float(parallel))
            if media_command_count else 0.0
        ),
        bottleneck="media" if media_ns else "none",
    )

    return {
        "version": c["version"],
        "contract_field": field_name,
        "host_transaction_bytes": c["host_transaction_bytes"],
        "host_max_request_bytes": c["host_max_request_bytes"],
        "media_page_bytes": c["media_page_bytes"],
        "page_offset_bytes": page_offset_bytes,
        "address_scope": "known_page_offset" if known_offset else "unknown",
        "pages_touched": pages,
        "command_queue_depth": c["command_queue_depth"],
        "media_parallelism": c["media_parallelism"],
        "physical_planes": declared_planes if declared_planes is not None else "unknown",
        "addressable_media_slots": topology_slots if topology_basis.startswith("explicit") else "unknown",
        "topology_basis": topology_basis,
        "parallelism_basis": (
            "declared_media_parallelism_and_physical_plane_cap"
            if declared_planes is not None
            else "declared_media_parallelism_without_physical_plane_cap"
        ),
        "page_read_latency_ns": c["page_read_latency_ns"],
        "page_program_latency_ns": c["page_program_latency_ns"],
        "access_pattern": c["access_pattern"],
        "operation": operation,
        "program_order": program_order,
        "program_parallelism": program_parallel,
        "program_order_constraint": (
            "one_page_at_a_time" if program_order == "sequential" else
            "not_declared" if program_order in {"unknown", "unspecified"} else "unknown"
        ),
        "erase_operations": 0,
        "background_work": background_work if background_work is not None else c.get("background_work", {}),
        "physical_dies": c.get("physical_dies", "unknown"),
        "physical_channels": c.get("physical_channels", "unknown"),
        "physical_owner": str(
            getattr(component, "metadata", {}).get("physical_owner")
            or getattr(component, "component_id", "unknown")
        ),
        "physical_resource_id": str(
            getattr(component, "metadata", {}).get("memory_resource_id")
            or getattr(component, "metadata", {}).get("physical_owner")
            or getattr(component, "component_id", "unknown")
        ),
        "host_transfer_bytes": host_bytes,
        "physical_read_bytes": physical_read,
        "physical_write_bytes": physical_write,
        "physical_bytes": physical_read + physical_write,
        "read_operations": read_ops,
        "program_operations": program_ops,
        "command_count": host_command_count,
        "media_command_count": media_command_count,
        "rmw_read_operations": rmw,
        "effective_parallelism": parallel,
        "media_waves": rwaves + pwaves,
        "read_waves": rwaves,
        "program_waves": pwaves,
        "host_waves": host_waves,
        "host_queue_wait_ns": host_queue_wait_ns,
        "media_queue_wait_ns": media_queue_wait_ns,
        "queue_wait_ns": host_queue_wait_ns + media_queue_wait_ns,
        "queue_saturated": bool(media_command_count > parallel or host_command_count > c["command_queue_depth"]),
        "media_read_service_ns": read_ns,
        "media_program_service_ns": program_ns,
        "host_service_ns": host_ns,
        # Deliberately serialized: no cache, GC, or early-ack overlap is assumed.
        "service_ns": media_ns + host_ns,
        **overall_metrics,
        "host_realtime_throughput_gb_s": host_metrics["realtime_throughput_gb_s"],
        "host_bandwidth_ceiling_gb_s": host_metrics["bandwidth_ceiling_gb_s"],
        "host_bandwidth_utilization": host_metrics["bandwidth_utilization"],
        "media_realtime_throughput_gb_s": media_metrics["physical_realtime_throughput_gb_s"],
        "media_bandwidth_utilization": media_metrics["bandwidth_utilization"],
        "media_bandwidth_ceiling_gb_s": media_metrics["bandwidth_ceiling_gb_s"],
        "read_energy_pj": read_energy,
        "write_energy_pj": write_energy,
        "energy_pj": energy,
        "timing_diagnostics": "analytical conservative model; serialized host transfer plus media waves and bandwidth",
        "profile_evidence": "analytical/no hardware validated",
        **({"source": c["source"]} if c.get("source") else {}),
        "parameter_evidence": {
            **_parameter_evidence(c),
            "memory_access_offset_bytes": {
                "value": page_offset_bytes if known_offset else "unknown",
                "status": "parameterized" if known_offset else "unknown",
                **({"source": c["source"]} if c.get("source") else {}),
            },
        },
        "cache_policy": "cold_no_persistent_cache",
        "write_completion": "media_program_complete",
        **_address_mapping(c, page_offset_bytes, pages, page),
    }


def hbf_media_service(
    component,
    byte_count,
    read: bool,
    *,
    page_offset_bytes=None,
    operation=None,
    background_work=None,
) -> dict:
    """Compatibility alias for the legacy HBF cold-page entrypoint."""

    return nand_media_service(
        component,
        byte_count,
        read,
        page_offset_bytes=page_offset_bytes,
        operation=operation,
        background_work=background_work,
    )
