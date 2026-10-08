"""Load timing-free GGML structural compilation inputs for CUDA lifecycle.

The diagnostic native backend emits complete node-property snapshots before
making its CUDA Graph decision. In dry mode it then skips CUDA execution.
Only dry snapshots can become predictor inputs. Live snapshots and observed
decisions can be used to check the compiler, never to replace its prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .cuda_graph_lifecycle import CudaGraphInvocation, SOURCE_REVISION


@dataclass(frozen=True)
class CudaGraphStructuralCall:
    label: str
    device: int
    context_id: str
    graph_key: str
    graph_uid: int
    node_properties: tuple[bytes, ...]
    compatible: bool
    dry_run: bool

    def invocation(self, *, enabled: bool, update_result: str | None = None):
        if not self.dry_run:
            raise ValueError("live CUDA execution records cannot be prediction inputs")
        return CudaGraphInvocation(
            context_id=self.context_id, graph_key=self.graph_key,
            graph_uid=self.graph_uid, node_properties=self.node_properties,
            compatible=self.compatible, compatibility_reason="source_operator_dispatch",
            enabled=enabled, update_result=update_result,
        )


@dataclass(frozen=True)
class CudaGraphStructuralProgram:
    calls: tuple[CudaGraphStructuralCall, ...]
    source_path: str

    def for_label(self, label: str) -> tuple[CudaGraphStructuralCall, ...]:
        calls = tuple(call for call in self.calls if call.label == label)
        if not calls:
            raise ValueError(f"no compiled CUDA backend invocation for {label}")
        return calls

    @property
    def prediction_input(self):
        return bool(self.calls) and all(call.dry_run for call in self.calls)


def _snapshot_bytes(value):
    if not isinstance(value, str) or not value or len(value) % 2:
        raise ValueError("CUDA node property snapshot must be complete hex bytes")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid CUDA node property hex snapshot") from exc


def load_cuda_graph_structure(path: str | Path, *, require_dry_run: bool = True):
    """Expand lossless property dictionaries, discarding all diagnostic events.

    No duration, capture decision, replay counter or native update result is
    read. Reusing such fields as a prediction input is intentionally impossible
    through the returned type. The producer revision is an applicability tag,
    not independent proof of the installed binary identity.
    """
    properties: dict[int, bytes] = {}
    calls = []
    widths = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"invalid structural record at line {line_number}")
            kind = row.get("kind")
            if kind == "node_property":
                key = row.get("id")
                if type(key) is not int or key < 0 or key in properties:
                    raise ValueError("duplicate or invalid CUDA node property dictionary ID")
                value = _snapshot_bytes(row.get("bytes"))
                properties[key] = value
                widths.add(len(value))
                continue
            if kind != "snapshot":
                continue
            if row.get("source_revision") != SOURCE_REVISION:
                raise ValueError("structural compiler source revision does not match lifecycle model")
            label = row.get("label")
            context = row.get("context_id")
            graph_key = row.get("graph_key")
            if any(not isinstance(v, str) or not v for v in (label, context, graph_key)):
                raise ValueError("structural snapshots require explicit call and pointer identities")
            uid, device, node_count = row.get("graph_uid"), row.get("device"), row.get("n_nodes")
            if (type(uid) is not int or uid < 0 or type(device) is not int or device < 0
                    or type(node_count) is not int or node_count < 1):
                raise ValueError("invalid structural graph uid, device or node count")
            dry, compatible = row.get("dry_run"), row.get("compatible")
            if type(dry) is not bool or type(compatible) is not bool:
                raise ValueError("structural execution and compatibility modes must be explicit")
            if require_dry_run and not dry:
                raise ValueError("live CUDA execution records cannot be prediction inputs")
            references = row.get("node_property_refs")
            if not isinstance(references, list) or len(references) != node_count:
                raise ValueError("structural snapshot does not cover every GGML node")
            if any(type(ref) is not int or ref not in properties for ref in references):
                raise ValueError("structural snapshot references undefined node properties")
            calls.append(CudaGraphStructuralCall(label, device, context, graph_key, uid,
                tuple(properties[ref] for ref in references), compatible, dry))
    if not calls or len(widths) != 1:
        raise ValueError("structural file must contain complete uniform-width GGML node snapshots")
    return CudaGraphStructuralProgram(tuple(calls), str(Path(path).resolve()))


def structural_lifecycle_signature(program: CudaGraphStructuralProgram):
    """Compare source decision predicates across address-randomized processes.

    Pointer values themselves cannot be equal in separate processes. Preserve
    context/key aliasing, nonzero-UID reuse, node counts, and the exact property
    equality predicate. No observed native Graph decision enters this signature.
    """
    contexts, keys, previous = {}, {}, {}
    result = []
    for call in program.calls:
        context = contexts.setdefault(call.context_id, len(contexts))
        key = keys.setdefault((context, call.graph_key), len(keys))
        previous_uid, previous_props = previous.get(key, (0, ()))
        uid_reused = call.graph_uid != 0 and call.graph_uid == previous_uid
        properties_changed = None if uid_reused else call.node_properties != previous_props
        result.append((call.label, call.device, context, key, len(call.node_properties),
                       call.compatible, uid_reused, properties_changed))
        if call.compatible and not uid_reused:
            previous[key] = (call.graph_uid, call.node_properties)
    return tuple(result)


def compare_cuda_graph_structure(dry: CudaGraphStructuralProgram,
                                 live: CudaGraphStructuralProgram):
    if not dry.prediction_input or any(call.dry_run for call in live.calls):
        raise ValueError("structural verification requires separate dry and live programs")
    left, right = structural_lifecycle_signature(dry), structural_lifecycle_signature(live)
    mismatches = tuple({"call_index": index, "dry": a, "live": b}
        for index, (a, b) in enumerate(zip(left, right)) if a != b)
    return {"qualified": len(left) == len(right) and not mismatches,
            "dry_call_count": len(left), "live_call_count": len(right),
            "mismatches": mismatches,
            "scope": "source_lifecycle_predicates_only_not_cuda_node_topology_or_timing",
            "native_decisions_used_as_prediction_inputs": False}
