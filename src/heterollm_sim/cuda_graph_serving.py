"""Runtime-owned CUDA Graph binding for the explicit source-assisted path.

No lifecycle state is stored in planner caches. This narrow adapter admits one
physical ubatch, one CUDA backend split and one sequence at a time. Every
native warmup call must be represented by an explicit simulated request.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path

from .cuda_graph_lifecycle import (
    CudaGraphRuntime, SOURCE_REVISION, bind_cuda_graph_tasks, _is_direct_device_memory_task,
)
from .cuda_graph_structure import load_cuda_graph_structure
from .cuda_graph_contract import (
    require_cuda_graph_source_contract, validate_cuda_graph_scenario_contract,
    INITIALIZATION_STAGES,
)
from .contracts import TaskCategory


_ACTIVE_CUDA_GRAPH_COHORT = ContextVar("active_cuda_graph_cohort", default=None)
CONFIG_KEY = "cuda_graph_structural_program"


def active_cuda_graph_cohort():
    return _ACTIVE_CUDA_GRAPH_COHORT.get()


def _chain_descriptor(row):
    """Qualify an actual captured pure-kernel chain; no topology guessing."""
    nodes, edges = row.get("nodes"), row.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("CUDA structural capture has no complete graph topology")
    edge_data = row.get("edge_data")
    if (not isinstance(edge_data, list) or len(edge_data) != len(edges)
            or any(value != "0000000000000000" for value in edge_data)):
        raise ValueError("CUDA topology has uncalibrated or missing dependency edge types")
    ids = tuple(node.get("id") for node in nodes)
    if len(ids) != len(set(ids)) or any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("CUDA topology node identities are invalid")
    if any(node.get("type") != 0 for node in nodes):
        raise ValueError("CUDA topology includes uncalibrated non-kernel nodes")
    predecessors, successors = {key: [] for key in ids}, {key: [] for key in ids}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or any(key not in predecessors for key in edge):
            raise ValueError("invalid CUDA topology edge")
        source, target = edge
        predecessors[target].append(source)
        successors[source].append(target)
    roots = [key for key in ids if not predecessors[key]]
    if len(edges) != len(ids) - 1 or len(roots) != 1 or any(
            len(predecessors[key]) > 1 or len(successors[key]) > 1 for key in ids):
        raise ValueError("CUDA structural topology is outside independently measured pure-kernel chain")
    order, key = [], roots[0]
    while key not in order:
        order.append(key)
        if not successors[key]:
            break
        key = successors[key][0]
    if len(order) != len(ids):
        raise ValueError("CUDA structural graph is not a connected acyclic chain")
    by_id = {node["id"]: node for node in nodes}
    signature = tuple(tuple((field, tuple(node[field]) if isinstance(node.get(field), list) else node.get(field))
                            for field in ("type", "function", "grid", "block", "shared_bytes"))
                      for node in (by_id[key] for key in order))
    if any(not by_id[key].get("function") for key in order):
        raise ValueError("CUDA kernel function identities are required for update compatibility")
    return {"topology": "chain", "node_count": len(ids), "signature": signature}


def cuda_graph_topology_descriptor(row):
    """Canonical typed chain recipe shared by predictor and synthetic probes.

    The key includes node type/copy geometry and dependency type, never model
    names, addresses, kernel function identities or measured LLM latency.
    Kernel attributes remain separately in the executable-update signature.
    """
    nodes, edges, edge_data = row.get("nodes"), row.get("edges"), row.get("edge_data")
    if (not isinstance(nodes, list) or not nodes or not isinstance(edges, list)
            or not isinstance(edge_data, list) or len(edge_data) != len(edges)):
        raise ValueError("CUDA capture requires complete typed nodes and edges")
    by_id = {node.get("id"): node for node in nodes}
    if len(by_id) != len(nodes) or any(not isinstance(key, str) or not key for key in by_id):
        raise ValueError("invalid CUDA node identity")
    pred, succ, incoming = {key: [] for key in by_id}, {key: [] for key in by_id}, {}
    for edge, data in zip(edges, edge_data):
        if (not isinstance(edge, list) or len(edge) != 2 or any(key not in by_id for key in edge)
                or data not in {"0000000000000000", "0100010000000000"}):
            raise ValueError("CUDA chain has unsupported dependency edge semantics")
        a, b = edge
        pred[b].append(a)
        succ[a].append(b)
        incoming[b] = data
    roots = [key for key in by_id if not pred[key]]
    if len(edges) != len(nodes) - 1 or len(roots) != 1 or any(
            len(pred[key]) > 1 or len(succ[key]) > 1 for key in by_id):
        raise ValueError("CUDA structural topology is not a simple typed chain")
    order, seen, key = [], set(), roots[0]
    while key not in seen:
        order.append(key)
        seen.add(key)
        if not succ[key]:
            break
        key = succ[key][0]
    if len(order) != len(nodes):
        raise ValueError("CUDA structural graph is not a connected acyclic chain")
    ordered, recipes, signatures = [], [], []
    update_coverage = True
    for key in order:
        node = by_id[key]
        kind = node.get("type")
        if kind == 0:
            if not node.get("function"):
                raise ValueError("CUDA structural kernel function identity is missing")
            update_coverage = update_coverage and type(node.get("cooperative")) is int
            recipe = ["kernel"]
            signature = tuple((field, tuple(node[field]) if isinstance(node.get(field), list) else node.get(field))
                              for field in ("function", "grid", "block", "shared_bytes", "cooperative"))
        elif kind == 1:
            copy = node.get("copy", node)
            fields = ("src_memory_type", "dst_memory_type", "width_bytes", "height", "depth", "src_pitch", "dst_pitch")
            if any(type(copy.get(field)) is not int for field in fields):
                raise ValueError("CUDA memcpy structural geometry is incomplete")
            if copy["height"] != 1 or copy["depth"] != 1 or copy["width_bytes"] <= 0:
                raise ValueError("CUDA synthetic runtime probes cover one-dimensional copies only")
            if copy["src_memory_type"] != 2 or copy["dst_memory_type"] != 2:
                raise ValueError("CUDA synthetic probes currently cover device-to-device copies only")
            if incoming.get(key, "0000000000000000") != "0000000000000000":
                raise ValueError("CUDA memcpy requires default dependency edge semantics")
            recipe = ["memcpy", *[copy[field] for field in fields]]
            update_coverage = update_coverage and all(
                isinstance(copy.get(field), str) and bool(copy[field])
                for field in ("src_context", "dst_context")) and all(
                    type(copy.get(field)) is int and copy[field] >= 0
                    for field in ("src_device_ordinal", "dst_device_ordinal"))
            signature = tuple((field, copy.get(field)) for field in (*fields,
                "src_context", "dst_context", "src_device_ordinal", "dst_device_ordinal"))
        else:
            raise ValueError("CUDA typed chain contains an unmeasured node kind")
        recipe.append(incoming.get(key, "root"))
        recipes.append(recipe)
        signatures.append((kind, signature, incoming.get(key, "root")))
        ordered.append(node)
    if all(recipe[0] == "kernel" and recipe[-1] in {"root", "0000000000000000"} for recipe in recipes):
        topology = "chain"
    else:
        runs = []
        for recipe in recipes:
            if runs and runs[-1][0] == recipe:
                runs[-1][1] += 1
            else:
                runs.append([recipe, 1])
        topology = "typed_chain/v1:" + json.dumps(runs, separators=(",", ":"))
    return {"topology": topology, "node_count": len(nodes), "signature": tuple(signatures),
            "ordered_nodes": tuple(ordered), "ordered_edge_data": tuple(incoming[key] for key in order[1:]),
            "update_compatibility_covered": update_coverage}


def typed_chain_topology_key(row):
    return cuda_graph_topology_descriptor(row)["topology"]


def _phase_structures(events, current, previous):
    def small(value):
        return {key: value[key] for key in ("topology", "node_count", "structure_id") if key in value}
    result = {}
    for event in events:
        if event in {"destroy_graph", "destroy_exec"}:
            if previous is None:
                raise ValueError("CUDA destruction requires the retained old graph structure")
            result[event] = small(previous)
        else:
            result[event] = small(current)
        if event in {"update", "update_failure"}:
            result[event].update(previous=small(current if previous is None else previous),
                                 current=small(current))
    return result


def _load_capture_topologies(path):
    """Only explicit capture-only records are legal predictor structure inputs."""
    labels, structures = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "snapshot":
            if row.get("capture_only") is not True or row.get("source_revision") != SOURCE_REVISION:
                raise ValueError("CUDA topology input must be capture-only; live execution is validation only")
            labels[row["call_id"]] = row["label"]
        elif row.get("kind") == "cuda_structure":
            if row.get("capture_only") is not True or row.get("source_revision") != SOURCE_REVISION:
                raise ValueError("CUDA topology must come from the supported capture-only compiler")
            call_id = row.get("call_id")
            if call_id not in labels or labels[call_id] in structures:
                raise ValueError("CUDA topology must identify exactly one backend split per label")
            structures[labels[call_id]] = cuda_graph_topology_descriptor(row)
    if not structures or len(structures) != len(labels):
        raise ValueError("capture-only topology file does not cover every backend invocation")
    return structures


@dataclass
class _PreparedCudaCohort:
    owner: "CudaGraphServingRuntime"
    cohort: object
    host_time_us: int
    label: str
    stage: str = "request"
    transition: object = None
    next_topology: object = None
    cost_audits: tuple = ()

    def register_costs(self, tasks):
        self.cost_audits = tuple(task.metadata["cuda_runtime_cost"] for task in tasks
                                if task.metadata.get("cuda_runtime_cost", {}).get("schema") == "cuda-runtime-cost/v1")

    def bind(self, tasks, groups):
        if self.transition is not None:
            raise ValueError("one CUDA cohort must be lowered once, not through a stale cost cache")
        if len(groups) != 1:
            raise ValueError("source-assisted CUDA binding currently requires one physical ubatch")
        group = groups[0]
        if len(tuple(group.get("request_ids", ()))) != 1:
            raise ValueError("source-assisted CUDA binding requires one sequence")
        if tuple(group["request_ids"]) != (self.cohort.items[0].request_id,):
            raise ValueError("CUDA physical invocation does not belong to the active request")
        calls = self.owner.program.for_label(self.label)
        if len(calls) != 1:
            raise ValueError("source-assisted CUDA binding requires one native CUDA backend split")
        call = calls[0]
        descriptor = self.owner.topologies.get(self.label)
        if descriptor is None:
            raise ValueError("CUDA invocation has no independent capture-only topology")
        old = self.owner.executable_topologies.get((call.context_id, call.graph_key))
        update_result = None
        try:
            transition = self.owner.lifecycle.prepare(call.invocation(
                enabled=self.owner.enabled), host_time_us=self.host_time_us)
        except ValueError as error:
            if "source-derived executable update compatibility" not in str(error):
                raise
            if old is None:
                raise ValueError("CUDA executable has no source-compiled update topology") from error
            if old["node_count"] != descriptor["node_count"]:
                update_result = "constraints_failure"
            elif (old["update_compatibility_covered"] and descriptor["update_compatibility_covered"]
                  and old["signature"] == descriptor["signature"]):
                update_result = "success"
            else:
                raise ValueError("CUDA executable update compatibility is outside the exact chain contract") from error
            transition = self.owner.lifecycle.prepare(call.invocation(
                enabled=self.owner.enabled, update_result=update_result), host_time_us=self.host_time_us)
        if transition.evicted:
            raise ValueError("CUDA Graph cache eviction requires independently priced destruction")
        launches = [task for task in tasks if task.metadata.get("phase") == "kernel_launch"]
        targets = {task.metadata.get("target_component") for task in launches}
        if len(targets) != 1 or None in targets:
            raise ValueError("source-assisted CUDA binding requires exactly one GPU target")
        target = next(iter(targets))
        group_id = group["group_id"]
        memory_targets = set()
        if any(task.metadata.get("direct_memory_component")
               or (task.metadata.get("event_kind") == "kv_read"
                   and task.metadata.get("resource_accounting") == "included_in_attention_kernel")
               for task in tasks):
            from .planner import _parallel_plan
            memory_targets = {rank.memory_component_id
                for rank in _parallel_plan(self.owner.scenario).ranks
                if rank.component_id == target}
        # Device work authored under one physical operator group is the source
        # split. CPU sampling, input uploads and output transfers stay outside.
        core = {task.task_id for task in tasks
                        if task.metadata.get("operator_invocation_group_id") == group_id
                        and (task.metadata.get("target_component") == target
                             or _is_direct_device_memory_task(task, target, memory_targets))
                        and (task.demands or task.metadata.get("phase") == "kernel_launch")
                        and not task.metadata.get("orchestration_stage")
                        and not task.metadata.get("transfer_kind")
                        and not task.metadata.get("serving_output_stage")}
        # Lowering emits cost-free access/join markers between actual GPU
        # tasks. Include only markers lying *between* device work; pulling a
        # leading marker into the graph could cross the CPU embedding split.
        children, parents = defaultdict(list), {}
        for task in tasks:
            parents[task.task_id] = task.dependencies
            for dependency in task.dependencies:
                children[dependency].append(task.task_id)
        def reachable(adjacency):
            seen, pending = set(core), list(core)
            while pending:
                for key in adjacency.get(pending.pop(), ()):
                    if key not in seen:
                        seen.add(key)
                        pending.append(key)
            return seen
        interior = reachable(children) & reachable(parents)
        device_markers = set()
        for task in tasks:
            if (task.task_id not in interior or task.task_id in core or task.demands
                    or task.metadata.get("operator_invocation_group_id") != group_id
                    or task.category not in {TaskCategory.COMMUNICATION, TaskCategory.SYNCHRONIZATION}
                    or task.metadata.get("orchestration_stage") or task.metadata.get("serving_output_stage")):
                continue
            marker_target = task.metadata.get("target_component")
            if marker_target in (None, target):
                device_markers.add(task.task_id)
            elif (task.metadata.get("event_kind") == "kv_read"
                  and task.metadata.get("resource_accounting") == "included_in_attention_kernel"):
                if (marker_target in memory_targets
                        and task.metadata.get("source_component") == marker_target):
                    device_markers.add(task.task_id)
        members = tuple(task.task_id for task in tasks if task.task_id in core | device_markers)
        if any(task.task_id not in members for task in launches):
            raise ValueError("source-assisted CUDA split does not cover all planner launches")
        result = bind_cuda_graph_tasks(tasks, transition, member_task_ids=members,
            topology=descriptor["topology"], node_count=descriptor["node_count"],
            node_count_basis="native_capture_only_structural_compiler",
            phase_structures=_phase_structures(transition.events, descriptor, old),
            device_marker_task_ids=device_markers, device_memory_component_ids=memory_targets,
            structure_id=descriptor.get("structure_id"))
        self.transition, self.next_topology = transition, descriptor
        return result


class CudaGraphServingRuntime:
    def __init__(self, scenario):
        self.scenario = scenario
        config = scenario.workload.metadata.get(CONFIG_KEY)
        self.config = config
        self.lifecycle = CudaGraphRuntime()
        self.executable_topologies = {}
        self.position = 0
        self.records = []
        self.structures_registry = {}
        self.cost_registry = {}
        self.enabled = False
        expected_contract = validate_cuda_graph_scenario_contract(scenario)
        if config is None:
            return
        self.enabled = config["graph_enabled"]
        self.stages = dict(config["request_stages"])
        require_cuda_graph_source_contract(config["dry_program_path"], expected_contract)
        require_cuda_graph_source_contract(config["capture_program_path"], expected_contract)
        self.program = load_cuda_graph_structure(config["dry_program_path"])
        self.topologies = _load_capture_topologies(config["capture_program_path"])
        structure_ids = {}
        for descriptor in self.topologies.values():
            key = (descriptor["topology"], descriptor["node_count"])
            if key not in structure_ids:
                structure_id = "cuda_structure:" + str(len(structure_ids))
                structure_ids[key] = structure_id
                self.structures_registry[structure_id] = {
                    "topology": descriptor["topology"], "node_count": descriptor["node_count"]}
            descriptor["structure_id"] = structure_ids[key]
        self.prefixes = config.get("request_prefixes")
        if not isinstance(self.prefixes, dict) or not self.prefixes:
            raise ValueError("CUDA source program requires explicit request-to-label mapping including warmup")
        self.labels = tuple(dict.fromkeys(call.label for call in self.program.calls))
        if set(self.topologies) != set(self.labels):
            raise ValueError("CUDA dry and capture-only structural invocation labels differ")

    @contextmanager
    def scope(self, cohort, now_ns):
        if self.config is None:
            yield None
            return
        items = tuple(cohort.items)
        if len(items) != 1 or items[0].phase not in {"prefill", "decode"}:
            raise ValueError("CUDA source program requires one ordinary request item")
        item = items[0]
        prefix = self.prefixes.get(item.request_id)
        if not prefix:
            raise ValueError("simulated request has no structural compiler label")
        stage = self.stages[item.request_id]
        if stage in INITIALIZATION_STAGES:
            if (item.phase != "prefill" or item.completion_cursor != 2
                    or item.token_count != 2 or item.context_tokens != 0):
                raise ValueError("CUDA initialization must be a single prefill invocation")
            label = prefix
        elif item.phase == "prefill":
            label = prefix + ":prefill"
        else:
            if type(item.completion_cursor) is not int or item.completion_cursor < 2:
                raise ValueError("CUDA decode requires explicit output completion cursor")
            label = prefix + ":decode:" + str(item.completion_cursor - 1)
        if self.position >= len(self.labels) or label != self.labels[self.position]:
            raise ValueError("CUDA structural program call order mismatch; warmup cannot be skipped or guessed")
        prepared = _PreparedCudaCohort(self, cohort, int(now_ns // 1000), label, stage)
        token = _ACTIVE_CUDA_GRAPH_COHORT.set(prepared)
        try:
            yield prepared
        finally:
            _ACTIVE_CUDA_GRAPH_COHORT.reset(token)

    def commit(self, prepared):
        if prepared is None:
            return
        if prepared.owner is not self or prepared.transition is None:
            raise ValueError("CUDA cohort was not bound to a realized task graph")
        self.lifecycle.commit(prepared.transition)
        if prepared.transition.use_graph:
            self.executable_topologies[(prepared.transition.context_id, prepared.transition.graph_key)] = prepared.next_topology
        self.position += 1
        self.records.append({"label": prepared.label, **prepared.transition.metadata()})
        for audit in prepared.cost_audits:
            identity = audit.get("audit_id")
            if not identity:
                identity = prepared.transition.metadata()["invocation_id"]
            self.cost_registry[identity] = audit

    def failed(self):
        if self.config is not None:
            self.lifecycle.failed()

    def assert_complete(self):
        if self.config is not None and self.position != len(self.labels):
            raise ValueError("CUDA structural program was not fully executed; missing compiled invocations")

    def summary(self):
        return {"status": "disabled" if self.config is None else "source_assisted",
                "structural_compiler_dependency": "pinned_native_GGUF_graph_construction",
                "native_latency_used": False, "transitions": tuple(self.records),
                "event_counts": dict(Counter(event for row in self.records for event in row["events"])),
                "structures_registry": self.structures_registry,
                "cuda_runtime_cost_registry": self.cost_registry,
                "remaining_compiled_invocations": 0 if self.config is None else len(self.labels) - self.position}
