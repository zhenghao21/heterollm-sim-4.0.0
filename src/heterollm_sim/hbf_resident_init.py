"""Initialize warm HBF weight storage before any measured request executes.

The ordinary dense planner supplies the storage declarations.  This module
never executes its inventory DAG, changes a request, or fabricates a timed
NAND write.  ``weights_resident`` means these complete physical blocks were
loaded before the request; their padding and GC reserve still consume space.
"""
from __future__ import annotations

def has_warm_active_hbf(scenario):
    return scenario.weights_resident and any(
        component.normalized_kind == "hbf" and component.is_active_memory
        for component in scenario.hardware.components)


def _inventory_tasks(scenario):
    from .ir import model_graph_execution_layers
    from .planner import _compilation_scope, _lower_serving_cohort
    from .serving import BatchCohort, BatchItem

    layers = model_graph_execution_layers(scenario.model.graph)
    if (not layers or any(layer.kind != "dense" or layer.is_moe
                          or layer.is_linear_attention for layer in layers)
            or scenario.workload.mtp is not None
            or scenario.llama_cpp_config is not None
            or scenario.model.graph.attributes.get("metadata", {}).get("gguf_architecture_id")
            or any(getattr(scenario.resolve_component_profile(component), "kernel_model", None)
                   is not None for component in scenario.hardware.components
                   if component.normalized_kind == "gpu")):
        raise ValueError("warm HBF resident initialization requires a complete ordinary dense analytic weight inventory")
    # In this qualified path resident matrix shapes do not depend on token
    # rows or context.  Include the output head explicitly without lowering
    # a long-context activation working set merely to discover weights.
    item = BatchItem("@hbf-resident-inventory", "prefill", 1, 0,
        kv_append_tokens=1, kv_materialized_tokens=1,
        main_tokens=0, draft_tokens=0, verifier_tokens=1,
        committed_tokens=0, logit_tokens=1, completion_cursor=1)
    with _compilation_scope(scenario):
        return _lower_serving_cohort(scenario, BatchCohort(
            "@hbf-resident-inventory", "prefill", 0., (item,))).schedule.tasks


def _inventory(scenario):
    from .contracts import TaskCategory, TaskSpec
    from .data_motion import lower_physical_allocations

    declarations, order, bindings, configs = {}, {}, {}, {}
    for task in _inventory_tasks(scenario):
        task = lower_physical_allocations(task)
        metadata = task.metadata
        for owner, binding in metadata.get("hbf_host_staging_by_owner", {}).items():
            # Task routes depend on the CPU/GPU initiator.  Pre-existing
            # data issues no transfer; only the physical HOST identity,
            # staging capacity and GC contract must agree here.
            identity_fields = ("host_owner", "host_component_id", "host_config", "capacity_bytes", "reserve_blocks")
            if owner in bindings and any(bindings[owner].get(name) != binding.get(name)
                                         for name in identity_fields):
                raise ValueError("conflicting warm HBF initialization binding: " + owner)
            bindings.setdefault(owner, binding)
            config = metadata.get("physical_memory_configs", {}).get(owner)
            if config is None and metadata.get("physical_owner") == owner:
                config = metadata.get("physical_memory_config")
            if config is None:
                raise ValueError("warm HBF initialization has no owner configuration: " + owner)
            if owner in configs and configs[owner] != config:
                raise ValueError("conflicting warm HBF initialization configuration: " + owner)
            configs[owner] = config
        for row in metadata.get("physical_allocations", ()):
            owner = str(row.get("physical_owner") or metadata.get("physical_owner") or "")
            key = (owner, row["buffer_id"], row.get("generation", 0))
            normalized = {**row, "physical_owner": owner}
            declarations.setdefault(key, []).append(normalized)
        for owner, identity, generation in metadata.get("hbf_preloaded_buffers", ()):
            order.setdefault((owner, identity, generation), None)
    if not bindings:
        return None, (), {}, {}

    selected, visiting = {}, set()
    def select(key):
        if key in selected:
            return
        if key in visiting or key not in declarations:
            raise ValueError("warm HBF resident allocation or alias root is missing: " + str(key))
        visiting.add(key)
        rows = declarations[key]
        row = dict(rows[0])
        for other in rows[1:]:
            facts = ("alias_of", "alias_generation", "alias_offset_bytes", "allocation_arena", "address")
            if any(row.get(name) != other.get(name) for name in facts):
                raise ValueError("conflicting warm HBF resident allocation: " + str(key))
            if row["size_bytes"] != other["size_bytes"] and not (row.get("inferred") and other.get("inferred")):
                raise ValueError("warm HBF resident extent is not fixed: " + str(key))
            row["size_bytes"] = max(row["size_bytes"], other["size_bytes"])
        if key[2] != 0 or row.get("allocation_arena") != "resident":
            raise ValueError("warm HBF preload must identify a generation-zero resident weight")
        if row.get("alias_of"):
            select((key[0], row["alias_of"], row.get("alias_generation", key[2])))
        selected[key] = row
        visiting.remove(key)
    for key in order:
        if key[0] not in bindings:
            raise ValueError("warm HBF resident allocation has no staging binding")
        select(key)
    owners = tuple(bindings)
    metadata = {"physical_owner": owners[0], "physical_memory_config": configs[owners[0]],
        "physical_memory_configs": configs,
        "physical_allocations": tuple(selected.values())}
    task = TaskSpec("@hbf-resident-registration", "@bootstrap",
                    "Register pre-existing HBF resident weights", TaskCategory.MEMORY,
                    metadata=metadata)
    return task, tuple(order), bindings, configs


def _expected_hbf_resident_bytes(scenario):
    """Check the inventory against the existing physical placement ledger."""
    from .ir import default_memory_resource_id
    components = {}
    for component in scenario.hardware.components:
        if component.normalized_kind != "hbf" or not component.is_active_memory:
            continue
        owner = str(component.metadata.get("memory_service", {}).get("physical_owner")
                    or default_memory_resource_id(component))
        if owner in components.values():
            raise ValueError("warm HBF initialization requires unique physical owners")
        components[component.component_id] = owner
    decision = scenario.placement.metadata.get("control_plane", {}).get("decision", {})
    replicas = {}
    shards = decision.get("rank_weight_shards")
    if shards is None:
        raise ValueError("warm HBF initialization requires a complete physical weight placement ledger")
    for tensor, rows in shards.items():
        for row in rows:
            component = row["storage_component_id"]
            if component not in components or row.get("residency") == "cold_cim_transient":
                continue
            key = (component, row.get("replica_id", (tensor, row["rank_id"])))
            size = row["physical_bytes"]
            if key in replicas and replicas[key] != size:
                raise ValueError("conflicting HBF resident placement replica size")
            replicas[key] = size
    expected = dict.fromkeys(components.values(), 0)
    for (component, _), size in replicas.items():
        expected[components[component]] += size
    return expected


def initialize_hbf_resident_weights(scenario, kernel):
    """Materialize all qualified warm HBF roots without timed physical IO."""
    if not has_warm_active_hbf(scenario):
        return
    from . import hbf_staging_runtime
    from .data_motion import register_physical_allocations

    task, order, bindings, configs = _inventory(scenario)
    expected = _expected_hbf_resident_bytes(scenario)
    missing = [owner for owner, size in expected.items() if size and owner not in bindings]
    if missing:
        raise ValueError("warm HBF weights have no HOST staging initialization binding: " + ", ".join(missing))
    if set(bindings) - set(expected):
        raise ValueError("warm HBF inventory references an unknown physical owner")
    if not bindings:
        return
    runtime = kernel.physical_runtime
    for owner in bindings:
        previous = runtime.hbf_staging.get(owner)
        if previous is not None and previous.get("resident_initialization_complete"):
            raise ValueError("warm HBF resident initialization may run only once")
    register_physical_allocations(task, runtime)
    for owner, binding in bindings.items():
        hbf_staging_runtime.configure(runtime, owner, runtime.parse_config(configs[owner]), binding)
    roots = {}
    for owner, identity, generation in order:
        allocator = runtime.allocators[owner]
        identity, generation, _ = allocator.canonical_range(identity, 0, 1, generation)
        roots.setdefault((owner, identity, generation), allocator.lookup(identity, generation))
    actual = dict.fromkeys(expected, 0)
    for (owner, _, _), allocation in roots.items():
        actual[owner] += allocation.size_bytes
    for owner, size in expected.items():
        if actual[owner] != size:
            raise ValueError("warm HBF resident inventory bytes disagree with physical placement for {}: {} != {}".format(
                owner, actual[owner], size))

    # Check every device before preloading any of them.  A real capacity
    # failure is discovered at initialization, with all block padding charged.
    for owner in bindings:
        controller = runtime.hbf_staging[owner]["controller"]
        if controller._pages or controller._used or controller._preloads:
            raise ValueError("warm HBF initialization must precede dynamic media use")
        required_blocks = sum((allocation.size_bytes + controller.config.block_bytes - 1)
                              // controller.config.block_bytes
                              for (device, _, _), allocation in roots.items() if device == owner)
        if (required_blocks > len(controller._free_blocks)
                or required_blocks * controller.config.block_bytes > controller.usable_capacity_bytes):
            raise MemoryError("insufficient HBF capacity for complete resident initialization and GC reserve")
    for owner, identity, generation in roots:
        hbf_staging_runtime.preload(runtime, owner, identity, generation, bindings[owner])
    for owner in bindings:
        state = runtime.hbf_staging[owner]
        state["resident_initialization_complete"] = True
        state["resident_initialization_scope"] = "ordinary_dense_analytic_weight_inventory"
        state["resident_initialization_bytes"] = sum(
            allocation.size_bytes for (device, _, _), allocation in roots.items() if device == owner)
