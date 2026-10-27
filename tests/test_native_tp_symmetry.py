"""TP physical reuse keeps the original DAG, events and independent state."""
from heterollm_sim import native_layer_sampling as sampling
from heterollm_sim.execution_control import ExecutionControl
from test_native_layer_sampling import native_prepared
import pytest
from dataclasses import replace
from collections.abc import Mapping
from dataclasses import fields, is_dataclass


def exact(value):
    if type(value) is float:
        return float, value.hex()
    if isinstance(value, Mapping):
        return tuple((k, exact(value[k])) for k in sorted(value))
    if isinstance(value, (tuple, list)):
        return type(value), tuple(map(exact, value))
    if is_dataclass(value):
        return type(value), tuple((f.name, exact(getattr(value, f.name))) for f in fields(value))
    return value


@pytest.mark.parametrize("mode,with_import,delay", [("quiet", False, False),
    ("quiet", False, True), ("quiet", False, "offset"), ("all", False, False), ("all", True, False)])
def test_native_tp_events_and_window_are_bit_exact(native_prepared, monkeypatch, mode, with_import, delay):
    runtime = native_prepared["runtime"]
    source = runtime._execution_resource_kernel
    origin = max(runtime.now, native_prepared["ready"], source.makespan_ns)
    specs, _ = runtime._stage_task_specs(native_prepared["stages"],
        {stage.stage_id: origin for stage in native_prepared["stages"]}, "symmetry")
    wanted = [t.task_id for t in specs if t.metadata.get("layer_id") in ("dense0", "dense1")]
    if delay:
        from heterollm_sim.data_motion import is_physical_task
        changed = [t for t in specs if t.task_id in wanted and t.metadata.get("rank") == 1
                   and is_physical_task(t)][2]
        if delay == "offset":
            specs = tuple(replace(t, metadata={**t.metadata, "l2_execution": {
                "backing_arrival_offset_ns": 2.}}) if t is changed else t for t in specs)
        else:
            specs = tuple(replace(t, earliest_start_ns=origin + 1e6) if t is changed else t for t in specs)
    imports = None
    if with_import:
        # An empty live-import scope exercises the experimental dispatcher;
        # real phase interference is checked by the paired native candidates.
        imports = (runtime._external_executor, [])
        assert imports[0] is not None
    seen = []
    observe = sampling.LiveCohortMetrics.observe
    def record(self, event):
        seen.append(exact(event))
        observe(self, event)
    monkeypatch.setattr(sampling.LiveCohortMetrics, "observe", record)
    expected = sampling._execute_window(source, specs, wanted, origin_ns=origin,
        control=ExecutionControl(), sample_id="symmetry", concurrent_imports=imports)
    events = list(seen)
    seen.clear()
    actual = sampling._execute_window(source, specs, wanted, origin_ns=origin,
        control=ExecutionControl(), sample_id="symmetry",
        tp_symmetry=(runtime.plan.scenario, mode), concurrent_imports=imports)
    if delay:
        # Native first touches establish public key order before any reuse.
        # This early asymmetric transition can then consume every remaining
        # reuse opportunity; the contract here is fallback with exact events.
        assert actual["tp_symmetry"]["fallback_tasks"] > 0
    elif not with_import:
        assert actual["tp_symmetry"]["mirrored_tasks"] > 0
    assert seen == events
    def semantic(row):
        return exact({k: v for k, v in row.items() if k not in
                     ("wall_seconds", "wall_breakdown_seconds", "tp_symmetry")})
    assert semantic(actual) == semantic(expected)


@pytest.mark.parametrize("change", ["ep", "tp", "address", "alias_graph", "graph", "work", "contract", "offset", "empty_l2", "empty_l2_offset", "request"])
def test_structural_guards_exclude_unsupported_work(native_prepared, change):
    from heterollm_sim.data_motion import is_physical_task
    runtime = native_prepared["runtime"]
    source = runtime._execution_resource_kernel
    scenario = runtime.plan.scenario
    specs, _ = runtime._stage_task_specs(native_prepared["stages"],
        {stage.stage_id: runtime.now for stage in native_prepared["stages"]}, "guard")
    tasks = [t for t in specs if t.metadata.get("layer_id") == "dense0"]
    target = next(t for t in tasks if t.metadata.get("rank") == 1 and is_physical_task(t))
    if change in ("ep", "tp"):
        from types import SimpleNamespace
        p = scenario.placement.parallel
        parallel = SimpleNamespace(ep_degree=2 if change == "ep" else 1,
                                   tp_degree=1 if change == "tp" else p.tp_degree, pp_degree=1)
        scenario = SimpleNamespace(placement=SimpleNamespace(parallel=parallel))
    elif change == "address":
        owner = target.metadata["physical_owner"]
        source.physical_runtime.allocators[owner].allocate("asymmetric", 64)
    elif change == "alias_graph":
        owners = [next(t.metadata["physical_owner"] for t in tasks
                       if t.metadata.get("rank") == rank and is_physical_task(t)) for rank in (0, 1)]
        for rank, owner in enumerate(owners):
            allocator = source.physical_runtime.allocators[owner]
            allocator.allocate("root", 64)
            allocator.allocate("aliasA", 64, alias_of="root")
            allocator.allocate("aliasB", 64, alias_of="aliasA" if rank else "root")
    elif change == "graph":
        tasks = [replace(t, dependencies=()) if t is target else t for t in tasks]
    elif change == "request":
        tasks = [replace(t, request_id="different-request") if t is target else t for t in tasks]
    elif change in ("empty_l2", "empty_l2_offset"):
        key, value = ("stateful_l2", {}) if change == "empty_l2" else ("l2_execution", None)
        tasks = [replace(t, metadata={**t.metadata, key: value}) if t is target else t for t in tasks]
    elif change == "contract":
        tasks = [replace(t, metadata={**t.metadata, "phase_metadata": {"kernel_model": {
            "physical_memory_envelope": {"resource_suffix": ".other", "dependency_ns": 0.}}}})
                 if t is target else t for t in tasks]
    elif change == "offset":
        tasks = [replace(t, metadata={**t.metadata, "l2_execution": {"backing_arrival_offset_ns": 2.}})
                 if t is target else t for t in tasks]
    elif change == "work":
        tasks = [replace(t, demands=(replace(t.demands[0], service_ns=t.demands[0].service_ns+1),
                                    *t.demands[1:])) if t is target else t for t in tasks]
    mirror = sampling._TPRankMirror(source, tasks, scenario)
    assert target.task_id not in mirror.pairs
