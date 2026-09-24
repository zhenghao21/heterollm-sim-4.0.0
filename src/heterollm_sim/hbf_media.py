"""Small analytical cold-page model for opt-in HBF media endpoints."""
import math
from numbers import Real
from collections.abc import Mapping

_REQ = {"version", "host_transaction_bytes", "host_max_request_bytes", "media_page_bytes", "command_queue_depth", "media_parallelism", "page_read_latency_ns", "page_program_latency_ns", "access_pattern"}
_ALLOWED = _REQ | {"physical_planes"}
_PATTERNS = {"contiguous_page_aligned", "unknown_alignment_conservative"}

def _num(value, name, positive=False, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or (positive and value <= 0)
            or (nonnegative and value < 0)):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return value

def validate_hbf_media(component):
    contract = getattr(component, "metadata", {}).get("hbf_media")
    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("hbf_media must be a mapping")
    unknown = set(contract) - _ALLOWED
    if unknown:
        raise ValueError(f"unknown hbf_media keys: {sorted(unknown)}")
    missing = _REQ - set(contract)
    if missing:
        raise ValueError(f"missing hbf_media keys: {sorted(missing)}")
    if (contract["version"] != "cold_page_v1"
            or contract["host_transaction_bytes"] != 64
            or contract["media_page_bytes"] != 4096):
        raise ValueError("unsupported cold_page_v1 granularity")
    for key in ("host_transaction_bytes", "host_max_request_bytes", "media_page_bytes", "command_queue_depth", "media_parallelism"):
        if isinstance(contract[key], bool) or not isinstance(contract[key], int) or contract[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (contract["host_max_request_bytes"] < contract["host_transaction_bytes"]
            or contract["host_max_request_bytes"] > contract["media_page_bytes"]
            or contract["host_max_request_bytes"] % contract["host_transaction_bytes"]):
        raise ValueError("host_max_request_bytes must be a host-transaction multiple no larger than one media page")
    qd = contract["command_queue_depth"]
    if not 256 <= qd <= 16384 or qd & (qd - 1):
        raise ValueError("command_queue_depth must be a power of two in [256, 16384]")
    for key in ("page_read_latency_ns", "page_program_latency_ns"):
        _num(contract[key], key, True)
    if contract["access_pattern"] not in _PATTERNS:
        raise ValueError("unsupported access_pattern")
    if "physical_planes" in contract and (isinstance(contract["physical_planes"], bool) or not isinstance(contract["physical_planes"], int) or contract["physical_planes"] <= 0):
        raise ValueError("physical_planes must be a positive integer")

def _ceil(value, unit):
    return (value + unit - 1) // unit

def hbf_media_service(component, byte_count, read: bool) -> dict:
    validate_hbf_media(component)
    c = getattr(component, "metadata", {}).get("hbf_media")
    if c is None:
        raise ValueError("hbf_media contract is required")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError("byte_count must be a non-negative integer")
    if not isinstance(read, bool):
        raise ValueError("read must be bool")
    page, host_tx = c["media_page_bytes"], c["host_transaction_bytes"]
    host_bytes = _ceil(byte_count, host_tx) * host_tx if byte_count else 0
    nominal = _ceil(byte_count, page) if byte_count else 0
    unknown = c["access_pattern"] == "unknown_alignment_conservative"
    # A 64B-aligned request can start at most 4096-64B before a page end.
    pages = (_ceil(byte_count + page - host_tx, page) if unknown and byte_count else nominal)
    rmw = 0 if read else (min(2, pages) if unknown else int(bool(byte_count % page)))
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
    planes = c.get("physical_planes", c["media_parallelism"])
    parallel = min(c["media_parallelism"], planes, c["command_queue_depth"])
    rwaves = _ceil(read_ops, parallel) if read_ops else 0
    pwaves = _ceil(program_ops, parallel) if program_ops else 0
    direction = "read" if read else "write"
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
    media_ns = read_ns if read else read_ns + program_ns
    host_waves = _ceil(host_command_count, c["command_queue_depth"]) if host_command_count else 0
    host_ns = 8.0 * host_bytes / host_bw if host_bytes else 0.0
    read_energy_rate = _num(getattr(component, "metadata", {}).get("read_energy_pj_per_byte", 0.0),
                             "read_energy_pj_per_byte", nonnegative=True)
    write_energy_rate = _num(getattr(component, "metadata", {}).get("write_energy_pj_per_byte", 0.0),
                              "write_energy_pj_per_byte", nonnegative=True)
    read_energy = physical_read * read_energy_rate
    write_energy = physical_write * write_energy_rate
    energy = read_energy + write_energy

    return {
        "version": "cold_page_v1",
        "host_transaction_bytes": 64,
        "host_max_request_bytes": c["host_max_request_bytes"],
        "media_page_bytes": 4096,
        "command_queue_depth": c["command_queue_depth"],
        "media_parallelism": c["media_parallelism"],
        "physical_planes": planes,
        "page_read_latency_ns": c["page_read_latency_ns"],
        "page_program_latency_ns": c["page_program_latency_ns"],
        "access_pattern": c["access_pattern"],
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
        "host_waves": host_waves,
        "media_read_service_ns": read_ns,
        "media_program_service_ns": program_ns,
        "host_service_ns": host_ns,
        # Deliberately serialized: no cache, GC, or early-ack overlap is assumed.
        "service_ns": media_ns + host_ns,
        "read_energy_pj": read_energy,
        "write_energy_pj": write_energy,
        "energy_pj": energy,
        "timing_diagnostics": "analytical conservative model; serialized host transfer plus media waves and bandwidth",
        "profile_evidence": "analytical/no hardware validated",
        "cache_policy": "cold_no_persistent_cache",
        "write_completion": "media_program_complete",
    }
