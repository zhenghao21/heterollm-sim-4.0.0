"""Stationary forecast over attention-DP groups, and its memory cuts.

Groups refill their own slots from their own pool, layers stay synchronous
across groups (so imports delay a step by the busiest group), unbalanced
groups are rejected, and the sampler path drops the per-request KV records of
layers no window runs while the Decode guard keeps its errors and value.
"""
from dataclasses import replace
from itertools import chain
from random import Random
from types import SimpleNamespace
import tracemalloc

import pytest

from heterollm_sim import native_layer_sampling as sampling
from heterollm_sim import native_stationary as stationary
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig
import native_toy

NativeLayerSamplingGuardError = sampling.NativeLayerSamplingGuardError


# ---------------------------------------------------------------- refill / rho

def _centered_ages(count, output):
    return [(2 * index + 1) * output // (2 * count) for index in range(count)]


@pytest.mark.parametrize("seed", range(8))
def test_one_group_is_the_plain_schedule_and_rho_is_exactly_one(seed):
    rng = Random(seed)
    output = rng.randrange(3, 12)
    ages = [rng.randrange(output) for _ in range(rng.randrange(1, 14))]
    pool, warmup, measured = rng.randrange(1, 5), rng.randrange(0, 6), rng.randrange(1, 25)
    plain = stationary._refill_schedule(ages, output, pool, warmup, measured)
    grouped = stationary._group_refill_schedule([ages], output, pool, warmup, measured)
    assert (grouped["waits"], grouped["most"], grouped["lag"]) == plain
    assert grouped["rho"] == 1.0 and type(grouped["rho"]) is float
    assert grouped["peak_group_landings"] == grouped["total_landings"]


@pytest.mark.parametrize("seed", range(8))
def test_landings_follow_the_same_slot_rules_as_the_schedule(seed):
    rng = Random(100 + seed)
    output = rng.randrange(3, 12)
    ages = [rng.randrange(output) for _ in range(rng.randrange(1, 14))]
    pool, warmup, measured = rng.randrange(1, 5), rng.randrange(0, 6), rng.randrange(1, 25)
    first, last = warmup + 2, warmup + measured + 1
    landed = stationary._refill_landings(ages, output, pool, last)
    waits, _most, _lag = stationary._refill_schedule(ages, output, pool, warmup, measured)
    # A request is complete in the window when it lands from step `first` on
    # and its last token (output steps later) is inside the window.
    complete = sum(count for step, count in landed.items() if step >= first and step - 1 + output <= last)
    assert complete == len(waits)


def test_two_symmetric_groups_halve_the_busiest_group_share():
    output, batch = 8, 8
    seeds = _centered_ages(batch, output)
    by_group = [seeds[0::2], seeds[1::2]]
    schedule = stationary._group_refill_schedule(by_group, output, 2, 5, 14)
    for ages, row in zip(by_group, schedule["groups"]):
        plain = stationary._refill_schedule(ages, output, 2, 5, 14)
        assert (row["most_admissions_per_step"], row["fewest_import_steps"]) == plain[1:]
        assert row["pool"] == 2 and row["requests"] == 4
    assert schedule["waits"] == sum((stationary._refill_schedule(ages, output, 2, 5, 14)[0]
                                     for ages in by_group), [])
    assert 0.5 <= schedule["rho"] <= 1.0
    assert schedule["peak_group_landings"] * 2 >= schedule["total_landings"]
    # Identical groups land together: the busiest group carries half of them.
    twin = stationary._group_refill_schedule([seeds[0::2], seeds[0::2]], output, 2, 5, 14)
    assert twin["rho"] == 0.5 and twin["total_landings"] == 2 * twin["peak_group_landings"]
    # Groups that never land together make every landing the step's maximum.
    apart = stationary._group_refill_schedule([[0, 0], [4, 4]], 8, 2, 0, 16)
    assert apart["rho"] == 1.0


# --------------------------------------------------------------- evaluate (fake)

def _evaluate_fixture(monkeypatch, *, degree, batch=8, output=8, pool=4, warmup=5, measured=14):
    monkeypatch.setattr(stationary, "_step_service", lambda prepared, samples: (100., 110., {}))
    monkeypatch.setattr(stationary, "_import_extra", lambda samples: (2., 3., 5., 6., []))
    ages = _centered_ages(batch, output)
    states = [SimpleNamespace(spec=SimpleNamespace(request_id=f"seed{index}", output_tokens=output - age,
              metadata={"warmup_seed": True})) for index, age in enumerate(ages)]
    states += [SimpleNamespace(spec=SimpleNamespace(request_id=f"fresh{index}", output_tokens=output,
               metadata={"warmup_seed": False})) for index in range(batch)]
    parallel = SimpleNamespace(tp_degree=2, pp_degree=1, ep_degree=degree)
    runtime = SimpleNamespace(
        plan=SimpleNamespace(scheduler=SimpleNamespace(max_num_seqs=batch),
                             scenario=SimpleNamespace(placement=SimpleNamespace(parallel=parallel))),
        _external_ingress={"max_inflight_requests": pool, "warmup_decode_cohorts": warmup,
                           "measurement_decode_cohorts": measured},
        _state_values=states)
    group_of = {f"seed{index}": index % degree for index in range(batch)}
    prepared = {"context": {f"seed{index}": 20 + age for index, age in enumerate(ages)},
                "inventory": [], "query_decode_guard_ns": 0.,
                "ep_groups": {"degree": degree, "group_of": group_of, "requests": [batch // degree] * degree,
                              "mean_context_tokens": [30.] * degree, "rank_ids": [[0], [1]][:degree],
                              "component_group": {}},
                "import_pool_occupancy_by_group": [pool // degree] * degree}
    return runtime, prepared, SimpleNamespace(target_relative_error=.1), ages


def test_one_group_keeps_the_global_import_term(monkeypatch):
    runtime, prepared, options, ages = _evaluate_fixture(monkeypatch, degree=1)
    result = stationary.evaluate_stationary(runtime, prepared, {}, options)
    bounds = result["metric_bounds"]["mean_complete_request_tpot_ns"]
    # step + B*e/output, exactly as before the group factor existed.
    assert bounds["lower"] == 100. + 8 * 2. / 8 and bounds["upper"] == 110. + 8 * 3. / 8
    factor = result["error_budget"]["pd_import_group_factor"]
    assert factor["rho"] == 1.0 and factor["peak_group_landings"] == factor["total_landings"]
    assert result["error_budget"]["ep_groups"]["degree"] == 1
    plain = stationary._refill_schedule(ages, 8, 4, 5, 14)
    refill = result["error_budget"]["refill_schedule"]
    assert (refill["most_admissions_per_step"], refill["fewest_import_steps"]) == plain[1:]
    assert result["error_budget"]["import_pool"] == 4


def test_two_groups_scale_the_import_term_by_rho_and_record_it(monkeypatch):
    runtime, prepared, options, ages = _evaluate_fixture(monkeypatch, degree=2)
    result = stationary.evaluate_stationary(runtime, prepared, {}, options)
    schedule = stationary._group_refill_schedule([ages[0::2], ages[1::2]], 8, 2, 5, 14)
    rho = schedule["rho"]
    assert 0.5 <= rho <= 1.0
    bounds = result["metric_bounds"]["mean_complete_request_tpot_ns"]
    assert bounds["lower"] == 100. + 8 * 2. * rho / 8 and bounds["upper"] == 110. + 8 * 3. * rho / 8
    budget = result["error_budget"]
    assert budget["pd_import_group_factor"]["rho"] == rho
    assert budget["ep_groups"]["pool_per_group"] == 2 and budget["import_pool"] == 4
    assert [row["requests"] for row in budget["refill_schedule"]["groups"]] == [4, 4]
    assert budget["ep_groups"]["rank_ids"] == [[0], [1]]


def test_each_group_needs_its_own_prepared_imports(monkeypatch):
    runtime, prepared, options, _ages = _evaluate_fixture(monkeypatch, degree=2)
    prepared["import_pool_occupancy_by_group"] = [3, 1]
    result = stationary.evaluate_stationary(runtime, prepared, {}, options)
    assert result["reasons"] == ["native_stationary_import_pool_not_full_at_cutoff"]
    # One group's pool is full, the other short: the total alone would pass.
    assert sum(prepared["import_pool_occupancy_by_group"]) >= 4


def test_a_seed_outside_the_first_cohort_groups_is_rejected(monkeypatch):
    runtime, prepared, options, _ages = _evaluate_fixture(monkeypatch, degree=2)
    del prepared["ep_groups"]["group_of"]["seed3"]
    assert stationary.evaluate_stationary(runtime, prepared, {}, options)["reasons"] == [
        "native_stationary_ep_groups_unbalanced"]
    # One group schedules every seed, as before groups existed.
    runtime, prepared, options, _ages = _evaluate_fixture(monkeypatch, degree=1)
    expected = stationary.evaluate_stationary(runtime, prepared, {}, options)
    del prepared["ep_groups"]["group_of"]["seed3"]
    assert stationary.evaluate_stationary(runtime, prepared, {}, options) == expected


# ---------------------------------------------------- balance proof and records

def _plan(degree, batch, ranks):
    mapping = tuple(SimpleNamespace(rank=index, component_id=f"gpu{index}", memory_component_id=f"hbm{index}",
                                    cim_component_id=None, ep_rank=group)
                    for index, group in enumerate(ranks))
    components = tuple(SimpleNamespace(component_id=f"{kind}{index}", package_id=f"package{index}")
                       for index in range(len(ranks)) for kind in ("gpu", "hbm", "hbf")) + (
        SimpleNamespace(component_id="pd_host", package_id=None),)
    scenario = SimpleNamespace(hardware=SimpleNamespace(components=components),
        placement=SimpleNamespace(parallel=SimpleNamespace(ep_degree=degree, rank_mapping=mapping)))
    return SimpleNamespace(plan=SimpleNamespace(scenario=scenario,
                                                scheduler=SimpleNamespace(max_num_seqs=batch)))


def _cohort(groups, contexts=None):
    return SimpleNamespace(items=tuple(SimpleNamespace(request_id=f"r{index}", ep_group=group,
        context_tokens=100 + index if contexts is None else contexts[index]) for index, group in enumerate(groups)))


def test_groups_must_hold_exactly_batch_over_ep_requests():
    runtime = _plan(2, 4, (0, 0, 1, 1))
    groups = stationary._ep_groups(runtime, _cohort([0, 1, 0, 1], [10, 20, 30, 50]))
    assert groups["degree"] == 2 and groups["requests"] == [2, 2]
    assert groups["mean_context_tokens"] == [20., 35.]
    assert groups["rank_ids"] == [[0, 1], [2, 3]]
    assert groups["group_of"] == {"r0": 0, "r1": 1, "r2": 0, "r3": 1}
    for placement in ([0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0]):
        with pytest.raises(NativeLayerSamplingGuardError) as error:
            stationary._ep_groups(runtime, _cohort(placement))
        assert str(error.value) == "native_stationary_ep_groups_unbalanced"
    with pytest.raises(NativeLayerSamplingGuardError, match="ep_groups_unbalanced"):
        stationary._ep_groups(runtime, _cohort([0, 1, 2, 1]))


def test_one_group_is_never_unbalanced_and_a_short_cohort_keeps_its_old_reason():
    runtime = _plan(1, 4, (0, 0))
    groups = stationary._ep_groups(runtime, _cohort([0, 0]))
    assert groups["requests"] == [2] and groups["rank_ids"] == [[0, 1]] and groups["component_group"] == {}


def test_import_destination_group_comes_from_the_rank_mapping_and_packages():
    runtime = _plan(2, 4, (0, 0, 1, 1))
    known = stationary._ep_groups(runtime, _cohort([0, 1, 0, 1]))
    assert known["component_group"]["gpu2"] == 1 and known["component_group"]["hbm0"] == 0
    # hbf stores are not in the rank mapping; their rank's package names the group.
    assert known["component_group"]["hbf3"] == 1 and "pd_host" not in known["component_group"]
    prepared = {"ep_groups": known}
    one = {"destinations": [{"component_id": "hbf2"}, {"component_id": "hbf3"}]}
    assert stationary._destination_group(prepared, one) == 1
    for rows in ([{"component_id": "hbf0"}, {"component_id": "hbf2"}], [{"component_id": "pd_host"}]):
        with pytest.raises(NativeLayerSamplingGuardError, match="destination_group_ambiguous"):
            stationary._destination_group(prepared, {"destinations": rows})
    assert stationary._destination_group({"ep_groups": {"degree": 1}}, {"destinations": []}) == 0


def test_a_package_shared_by_two_groups_names_neither():
    scenario = _plan(2, 4, (0, 1)).plan.scenario
    shared = tuple(replace_package(component, "shared") for component in scenario.hardware.components)
    scenario = SimpleNamespace(hardware=SimpleNamespace(components=shared), placement=scenario.placement)
    groups = stationary._component_groups(scenario)
    assert groups["gpu0"] == 0 and groups["gpu1"] == 1 and "hbf0" not in groups


def replace_package(component, package):
    return SimpleNamespace(component_id=component.component_id, package_id=package)


def test_import_pool_occupancy_is_counted_per_owner_group():
    runtime = SimpleNamespace(_external_pending={"a": 1, "b": 2}, _external_ready={"c", "d"},
        states={name: SimpleNamespace(ep_group=group) for name, group in zip("abcd", (0, 1, 1, 1))})
    assert stationary._import_pool_occupancy(runtime, 2) == [1, 3]
    assert stationary._import_pool_occupancy(runtime, 1) == [4]


# ------------------------------------------------------------ sampled PD imports

def _stub_window(monkeypatch):
    calls = []

    def execute(source, specs, ids, *, origin_ns, control, sample_id, concurrent_imports=None, row_terms=None):
        calls.append((sample_id, tuple(ids)))
        done = {identity.split(".")[0]: float(index + 1) for index, identity in enumerate(ids)}
        return {"sample_id": sample_id, "start_ns": origin_ns, "end_ns": origin_ns + 10., "duration_ns": 10.,
                "boundary_origin_ns": origin_ns, "elapsed_from_boundary_ns": 10., "task_count": len(ids),
                "layer_completion_ns": done, "first_access_ns": None if row_terms is None else {},
                "concurrent_imports": None, "wall_seconds": 0.,
                "physical_observations": {}}
    monkeypatch.setattr(stationary, "_execute_window", execute)
    return calls


def _sampling_fixture(degree, pending):
    inventory = [{"layer_id": f"L{index}"} for index in range(6)]
    prepared = {"inventory": inventory, "by_layer": {row["layer_id"]: [row["layer_id"] + ".task"] for row in inventory},
                "boundary": {"before_first_layer": ["prefix"], "after_last_layer": ["suffix"]},
                "specs": (), "source": object(), "origin": 0., "context": {"r0": 20, "r1": 40},
                "record_layers": frozenset(row["layer_id"] for row in inventory),
                "ranks": {identity: index % 2 for index, identity in enumerate(
                    ["prefix", "suffix", *(row["layer_id"] + ".task" for row in inventory)])},
                "cohort": SimpleNamespace(cohort_id="cohort", kind="prefill"),
                "ep_groups": {"degree": degree, "component_group": {"hbm0": 0, "hbm1": 1}}}
    states = {index: {"request_id": name, "done": False, "row": 0, "offset": 0, "logical_bytes": size,
                      "destinations": [{"component_id": f"hbm{group}", "size_bytes": size}]}
              for index, (name, size, group) in enumerate(pending)}
    runtime = SimpleNamespace(_external_executor=SimpleNamespace(_async_plan_states=states),
                              _external_pending={name: None for name, _size, _group in pending})
    return runtime, prepared


def test_the_second_sampled_import_comes_from_another_group_when_there_is_one(monkeypatch):
    _stub_window(monkeypatch)
    runtime, prepared = _sampling_fixture(2, [("a", 300, 0), ("b", 200, 0), ("c", 100, 1)])
    samples = stationary.sample_stationary(runtime, prepared, ExecutionControl(), (2, 2), [])
    rows = samples["pd_imports"]["samples"]
    assert [(row["request_id"], row["destination_group"]) for row in rows] == [("a", 0), ("c", 1)]
    assert [row["logical_bytes"] for row in rows] == [300, 100]


def test_without_another_group_the_two_largest_imports_are_sampled(monkeypatch):
    _stub_window(monkeypatch)
    runtime, prepared = _sampling_fixture(2, [("a", 300, 0), ("b", 200, 0), ("c", 100, 0)])
    rows = stationary.sample_stationary(runtime, prepared, ExecutionControl(), (2, 2), [])["pd_imports"]["samples"]
    assert [(row["request_id"], row["destination_group"]) for row in rows] == [("a", 0), ("b", 0)]
    runtime, prepared = _sampling_fixture(1, [("a", 100, 0), ("b", 300, 1), ("c", 200, 0)])
    rows = stationary.sample_stationary(runtime, prepared, ExecutionControl(), (2, 2), [])["pd_imports"]["samples"]
    assert [row["request_id"] for row in rows] == ["b", "c"] and {row["destination_group"] for row in rows} == {0}


def test_sample_runs_stay_inside_calibration_and_holdout_layers(monkeypatch):
    calls = _stub_window(monkeypatch)
    runtime, prepared = _sampling_fixture(1, [("a", 300, 0), ("b", 200, 0)])
    stationary.sample_stationary(runtime, prepared, ExecutionControl(), (3, 2), [])
    roles = stationary._sample_roles(prepared["inventory"], (3, 2))
    kept = {layer for ids in roles.values() for layer in ids}
    assert kept == {"L0", "L1", "L2", "L4", "L5"}
    ran = {identity.split(".")[0] for _sample, ids in calls for identity in ids}
    assert ran - {"prefix", "suffix"} <= kept
    assert stationary._sample_roles(prepared["inventory"], (2, 5)) is None
    assert stationary._sample_roles(prepared["inventory"], (1, 2)) is None


def test_a_sample_never_runs_a_layer_whose_records_were_dropped(monkeypatch):
    calls = _stub_window(monkeypatch)
    runtime, prepared = _sampling_fixture(1, [("a", 300, 0), ("b", 200, 0)])
    # Preparation kept the (2, 2) runs' layers; a (3, 2) sample needs L2 too.
    prepared["record_layers"] = frozenset({"L0", "L1", "L4", "L5"})
    with pytest.raises(NativeLayerSamplingGuardError, match="sample_layer_records_dropped"):
        stationary.sample_stationary(runtime, prepared, ExecutionControl(), (3, 2), [])
    assert calls == []
    stationary.sample_stationary(runtime, prepared, ExecutionControl(), (2, 2), [])
    # A too-short stack dropped nothing.
    prepared["record_layers"] = None
    stationary.sample_stationary(runtime, prepared, ExecutionControl(), (3, 2), [])


# --------------------------------------------------------- dropping dead records

def _synthetic_views(layers=20, tasks_per_layer=10, rows=300):
    """Three views (spec, stage, capsule shells) sharing each task's record lists."""
    views = ([], [], [])
    for layer in range(layers):
        for task in range(tasks_per_layer):
            shared = {key: [{"byte_count": index, "buffer_id": f"kv:{layer}:{task}:{index}"} for index in range(rows)]
                      for key in ("memory_accesses", "buffer_accesses")}
            for view in views:
                view.append({"layer_id": f"L{layer}", "operator_class": "memory", "rank": task,
                             "persistent_buffer_ids": shared["buffer_accesses"], **shared})
    # Work outside any layer is never touched.
    views[0].append({"operator_class": "embedding", "memory_accesses": [{"byte_count": 1}]})
    return views


def test_only_unsampled_layers_lose_their_request_records():
    views = _synthetic_views(layers=6, tasks_per_layer=2, rows=3)
    kept = {"L0", "L1", "L5"}
    dropped = sampling.drop_unsampled_request_records(chain(*views), kept)
    # three unsampled layers x two tasks x three views x three record keys
    assert dropped == 3 * 2 * 3 * 3
    for view in views:
        for metadata in view:
            layer = metadata.get("layer_id")
            present = [key for key in sampling.PER_REQUEST_RECORD_KEYS if key in metadata]
            if layer is None:
                assert present == ["memory_accesses"]
            elif layer in kept:
                assert present == ["memory_accesses", "buffer_accesses", "persistent_buffer_ids"]
                assert all(len(metadata[key]) == 3 for key in present)
            else:
                assert present == []
            if layer is not None:
                assert metadata["operator_class"] == "memory" and "rank" in metadata


def test_dropping_dead_records_releases_most_of_the_cohort_memory():
    tracemalloc.start()
    try:
        views = _synthetic_views()
        before = tracemalloc.get_traced_memory()[0]
        sampling.drop_unsampled_request_records(chain(*views), {"L0", "L1"})
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    # Eighteen of twenty layers go; their lists are freed once no view holds them.
    assert after < 0.2 * before
    assert len(views[0][0]["memory_accesses"]) == 300 and len(views[2][19]["buffer_accesses"]) == 300
    assert views[0][-1]["memory_accesses"] == [{"byte_count": 1}]


# ------------------------------------------------- real toy: strip and the guard

KEYS = sampling.PER_REQUEST_RECORD_KEYS


def _record_counts(metadata_rows):
    counts = {}
    for metadata in metadata_rows:
        layer = metadata.get("layer_id")
        if layer is not None:
            counts[layer] = counts.get(layer, 0) + sum(key in metadata for key in KEYS)
    return counts


def _capsule_metadata(cost):
    return [fact["metadata"] for stage in cost.metadata["execution_stages"]
            for fact in stage["execution_tasks"] if "metadata" in fact]


def _fast_toy(calibration=4, holdout=3):
    from heterollm_sim.reporting import run_scenario
    scenario = native_toy.toy_scenario(8, 1., batch_size=8, output=6, warmup=5, measurement=10)
    return run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(calibration_layers=calibration, holdout_layers=holdout))),
        retention_policy="aggregate")


def test_sampler_path_cuts_unsampled_layers_in_every_view_before_the_decode_ir(monkeypatch):
    seen = {}
    original_prepare, original_guard = stationary.prepare_stationary, stationary._query_decode_guard
    order = []

    def prepare(runtime, cohort, stages, ready, control, cost, layers):
        seen["args"] = (stages, cost, layers)
        prepared = original_prepare(runtime, cohort, stages, ready, control, cost, layers)
        seen["specs"] = _record_counts(task.metadata for task in prepared["specs"])
        seen["capsule"] = _record_counts(_capsule_metadata(cost))
        seen["stages"] = _record_counts(task.metadata for stage in stages for task in stage.execution_tasks)
        seen["kept"] = prepared["record_layers"]
        return prepared

    def guard(runtime, cohort, stages, *, before_decode=None):
        def release():
            seen["before"] = _record_counts(task.metadata for stage in stages for task in stage.execution_tasks)
            order.append("release")
            before_decode()
            seen["released"] = _record_counts(task.metadata for stage in stages for task in stage.execution_tasks)
        return original_guard(runtime, cohort, stages, before_decode=release)

    from heterollm_sim import serving
    original_lower = serving._lower_cost

    def lower(lowerer, scenario, cohort):
        order.append("lower:" + cohort.kind)
        return original_lower(lowerer, scenario, cohort)

    monkeypatch.setattr(stationary, "prepare_stationary", prepare)
    monkeypatch.setattr(stationary, "_query_decode_guard", guard)
    monkeypatch.setattr(serving, "_lower_cost", lower)
    result = _fast_toy()
    assert result.approximation["rejection_reasons"] == []
    # The first IR is released before the Decode IR is lowered, once.
    assert order == ["lower:prefill", "release", "lower:decode"]
    assert seen["args"][2] == (4, 3)
    assert seen["before"]["dense4"] > 0 and seen["released"]["dense4"] == 0
    assert seen["kept"] == frozenset(seen["before"]) - {"dense4"}
    for layer, count in seen["before"].items():
        if layer != "dense4":
            assert seen["released"][layer] == count > 0
    # Every view lost the same records: none left outside the sampled layers.
    for view in ("specs", "capsule", "stages"):
        assert seen[view]["dense4"] == 0
        assert all(seen[view][layer] == seen["before"][layer] for layer in seen["before"] if layer != "dense4")


def test_dropping_dead_records_does_not_change_the_forecast_or_the_guard(monkeypatch):
    kept = _fast_toy()
    monkeypatch.setattr(stationary, "drop_unsampled_request_records", lambda rows, layers: 0)
    untouched = _fast_toy()
    assert kept.approximation["rejection_reasons"] == untouched.approximation["rejection_reasons"] == []
    left, right = kept.native_window_forecast, untouched.native_window_forecast
    assert left.metric_bounds == right.metric_bounds and left.metrics == right.metrics
    assert left.error_budget == right.error_budget


# --------------------------------------------- Decode guard: digest vs the oracle

def _oracle_guard(runtime, cohort, stages):
    """The previous guard: both IRs fully described as tuples, compared in order."""
    from heterollm_sim.serving import _lower_cost
    if not cohort.items or cohort.kind != "prefill" or any(item.token_count != 1 for item in cohort.items):
        raise NativeLayerSamplingGuardError("native_stationary_requires_final_prompt_query_cohort")
    items = tuple(replace(item, phase="decode", token_count=1, main_tokens=1, draft_tokens=0,
        verifier_tokens=1, committed_tokens=1, kv_append_tokens=1, kv_materialized_tokens=1,
        logit_tokens=1, completion_cursor=2) for item in cohort.items)
    projected = replace(cohort, cohort_id=cohort.cohort_id + ".native-stationary-decode",
                        kind="decode", items=items)
    cost = _lower_cost(runtime.lowerer, runtime.plan.scenario, projected)
    candidate, fallback = runtime._execution_stage_metadata_cache.resolve(cost.metadata)
    if fallback or not candidate:
        raise NativeLayerSamplingGuardError("native_stationary_decode_ir_unavailable")
    candidate = runtime._with_gpu_controller_stages(candidate, projected)

    def describe(rows):
        tasks = [task for stage in rows for task in stage.execution_tasks]
        positions = {task.task_id: index for index, task in enumerate(tasks)}
        return [(tuple(positions.get(parent, -1) for parent in task.dependencies),
                 str(task.category), str(task.metadata.get("operator_class", "")),
                 task.metadata.get("layer_id"), task.metadata.get("rank"),
                 task.metadata.get("expert_index"), task.metadata.get("expert_tokens"),
                 tuple((access.get("physical_owner", task.metadata.get("physical_owner")),
                        access.get("operation"), access.get("byte_count"),
                        access.get("allocation_generation", 0) == 0,
                        str(access.get("buffer_id", "")).startswith("kv:"))
                       for access in task.metadata.get("memory_accesses", ())),
                 tuple((row.resource_id, row.service_ns, row.bytes_moved, row.work_units)
                       for row in task.demands)) for task in tasks]

    left, right = describe(stages), describe(candidate)
    if len(left) != len(right):
        raise NativeLayerSamplingGuardError("native_stationary_query_decode_task_counts_differ")
    guard = 0.
    for before, after in zip(left, right):
        if before[:-1] != after[:-1]:
            raise NativeLayerSamplingGuardError("native_stationary_query_decode_graph_or_io_differs")
        if len(before[-1]) != len(after[-1]) or any((x[0], x[2], x[3]) != (y[0], y[2], y[3])
                                                    for x, y in zip(before[-1], after[-1])):
            raise NativeLayerSamplingGuardError("native_stationary_query_decode_work_shape_differs")
        guard += sum(abs(x[1] - y[1]) for x, y in zip(before[-1], after[-1]))
    return guard


def _outcome(function, *args):
    try:
        return "value", function(*args)
    except NativeLayerSamplingGuardError as error:
        return "error", str(error)


def _tamper(stages, predicate, change):
    """Apply ``change`` to the first task (in stage order) satisfying ``predicate``."""
    done = False
    rows = []
    for stage in stages:
        tasks = []
        for task in stage.execution_tasks:
            if not done and predicate(task):
                task, done = change(task), True
            tasks.append(task)
        rows.append(replace(stage, execution_tasks=tuple(tasks)))
    assert done
    return tuple(rows)


def _guard_inputs():
    captured = {}
    original = stationary.prepare_stationary

    def prepare(runtime, cohort, stages, ready, control, cost, layers):
        # Compare the guards on the untouched IR: no dead records dropped.
        captured.update(runtime=runtime, cohort=cohort, stages=stages)
        raise _Stop()

    class _Stop(BaseException):
        pass

    return captured, prepare, _Stop, original


@pytest.fixture
def guard_inputs(monkeypatch):
    captured, prepare, stop, _original = _guard_inputs()
    monkeypatch.setattr(stationary, "prepare_stationary", prepare)
    with pytest.raises(stop):
        _fast_toy()
    return captured


def test_digest_guard_returns_the_same_value_as_the_full_description(guard_inputs):
    runtime, cohort, stages = guard_inputs["runtime"], guard_inputs["cohort"], guard_inputs["stages"]
    expected = _oracle_guard(runtime, cohort, stages)
    assert stationary._query_decode_guard(runtime, cohort, stages) == expected
    # A service-time difference is summed per demand, in the old order.
    bumped = _tamper(stages, lambda task: task.demands and task.demands[0].service_ns > 0,
                     lambda task: replace(task, demands=(replace(task.demands[0],
                         service_ns=task.demands[0].service_ns + 12.5), *task.demands[1:])))
    shifted = stationary._query_decode_guard(runtime, cohort, bumped)
    assert shifted == _oracle_guard(runtime, cohort, bumped) and shifted > expected


def _with_access(task, **change):
    rows = list(task.metadata["memory_accesses"])
    rows[0] = {**rows[0], **change}
    return replace(task, metadata={**task.metadata, "memory_accesses": tuple(rows)})


TAMPERS = {
    "dependency": (lambda task: len(task.dependencies) > 1,
                   lambda task: replace(task, dependencies=task.dependencies[:-1])),
    "category": (lambda task: True,
                 lambda task: replace(task, category=next(c for c in type(task.category) if c != task.category))),
    "operator": (lambda task: task.metadata.get("operator_class"),
                 lambda task: replace(task, metadata={**task.metadata, "operator_class": "other"})),
    "layer": (lambda task: task.metadata.get("layer_id") is not None,
              lambda task: replace(task, metadata={**task.metadata, "layer_id": "elsewhere"})),
    "rank": (lambda task: type(task.metadata.get("rank")) is int,
             lambda task: replace(task, metadata={**task.metadata, "rank": task.metadata["rank"] + 1})),
    "access_bytes": (lambda task: task.metadata.get("memory_accesses"),
                     lambda task: _with_access(task, byte_count=task.metadata["memory_accesses"][0]["byte_count"] + 1)),
    "access_operation": (lambda task: task.metadata.get("memory_accesses"),
                         lambda task: _with_access(task, operation="write"
                             if task.metadata["memory_accesses"][0].get("operation") == "read" else "read")),
    "access_owner": (lambda task: task.metadata.get("memory_accesses"),
                     lambda task: _with_access(task, physical_owner="another-owner")),
    "access_kv_flag": (lambda task: task.metadata.get("memory_accesses"),
                       lambda task: _with_access(task, buffer_id="kv:flipped"
                           if not str(task.metadata["memory_accesses"][0].get("buffer_id", "")).startswith("kv:")
                           else "plain")),
    "access_removed": (lambda task: task.metadata.get("memory_accesses"),
                       lambda task: replace(task, metadata={**task.metadata,
                                            "memory_accesses": task.metadata["memory_accesses"][1:]})),
    "demand_bytes": (lambda task: task.demands,
                     lambda task: replace(task, demands=(replace(task.demands[0],
                         bytes_moved=task.demands[0].bytes_moved + 1), *task.demands[1:]))),
    "demand_work": (lambda task: task.demands,
                    lambda task: replace(task, demands=(replace(task.demands[0],
                        work_units=task.demands[0].work_units + 1.), *task.demands[1:]))),
    "demand_resource": (lambda task: task.demands,
                        lambda task: replace(task, demands=(replace(task.demands[0],
                            resource_id=task.demands[0].resource_id + "x"), *task.demands[1:]))),
    "demand_count": (lambda task: task.demands,
                     lambda task: replace(task, demands=task.demands[1:])),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_digest_guard_raises_the_same_error_as_the_full_description(guard_inputs, name):
    runtime, cohort, stages = guard_inputs["runtime"], guard_inputs["cohort"], guard_inputs["stages"]
    predicate, change = TAMPERS[name]
    tampered = _tamper(stages, predicate, change)
    expected = _outcome(_oracle_guard, runtime, cohort, tampered)
    assert _outcome(stationary._query_decode_guard, runtime, cohort, tampered) == expected
    assert expected[0] == "error"


def test_digest_guard_rejects_a_changed_task_count_like_the_full_description(guard_inputs):
    runtime, cohort, stages = guard_inputs["runtime"], guard_inputs["cohort"], guard_inputs["stages"]
    shorter = (replace(stages[0], execution_tasks=stages[0].execution_tasks[:-1]), *stages[1:])
    expected = _outcome(_oracle_guard, runtime, cohort, shorter)
    assert expected == ("error", "native_stationary_query_decode_task_counts_differ")
    assert _outcome(stationary._query_decode_guard, runtime, cohort, shorter) == expected


def test_guard_still_requires_the_final_prompt_query_cohort(guard_inputs):
    runtime, stages = guard_inputs["runtime"], guard_inputs["stages"]
    decode = replace(guard_inputs["cohort"], kind="decode")
    expected = ("error", "native_stationary_requires_final_prompt_query_cohort")
    assert _outcome(stationary._query_decode_guard, runtime, decode, stages) == expected
    assert _outcome(_oracle_guard, runtime, decode, stages) == expected


def test_the_digest_keeps_no_per_request_rows(guard_inputs):
    stages = guard_inputs["stages"]
    digest = stationary._IRDigest(stages)
    tasks = sum(len(stage.execution_tasks) for stage in stages)
    assert len(digest.graph) == len(digest.shape) == len(digest.demand_count) == tasks
    assert len(digest.service_ns) == sum(len(task.demands) for stage in stages for task in stage.execution_tasks)
    assert stationary._IRDigest.__slots__ == ("graph", "shape", "demand_count", "service_ns")


# ------------------------------------------------ real EP toy through the sampler

def test_two_ep_groups_run_through_prepare_and_sample_both_groups(monkeypatch):
    from heterollm_sim.reporting import run_scenario
    seen = {}
    original_prepare, original_sample = stationary.prepare_stationary, stationary.sample_stationary

    def prepare(*args):
        prepared = original_prepare(*args)
        seen["prepared"] = prepared
        return prepared

    def sample(runtime, prepared, control, layers, records, **kwargs):
        samples = original_sample(runtime, prepared, control, layers, records, **kwargs)
        seen["imports"] = samples["pd_imports"]["samples"]
        return samples

    monkeypatch.setattr(stationary, "prepare_stationary", prepare)
    monkeypatch.setattr(stationary, "sample_stationary", sample)
    scenario = native_toy.maverick_pattern_scenario(8, batch_size=4, output=6, warmup=5, measurement=10, ep=2)
    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(calibration_layers=4, holdout_layers=4))),
        retention_policy="aggregate")
    groups = seen["prepared"]["ep_groups"]
    assert groups["degree"] == 2 and groups["requests"] == [2, 2]
    assert groups["rank_ids"] == [[0, 1], [2, 3]]
    assert sorted(set(groups["group_of"].values())) == [0, 1]
    assert seen["prepared"]["import_pool_occupancy_by_group"] == [2, 2]
    # Two imports in flight, one per group: both are sampled.
    assert sorted(row["destination_group"] for row in seen["imports"]) == [0, 1]
    assert sorted(row.role for row in fast.layer_window_samples) == [
        "calibration", "entry", "exit", "holdout", "pd_import", "pd_import"]
    # The Maverick toy's imports outlast a step (as with one group): rejected, not extrapolated.
    assert fast.native_window_forecast is None
    assert "native_stationary_refill_not_always_ready" in fast.approximation["rejection_reasons"]


def test_two_ep_group_forecast_brackets_the_full_native_window():
    scenario = native_toy.moe_toy_scenario(8, batch_size=4, output=6, warmup=5, measurement=10, ep=2)
    fast, window, observed, complete = native_toy.stationary_and_full_window(scenario, 4, 4)
    assert fast.approximation["rejection_reasons"] == []
    assert fast.approximation["error_budget"]["ep_groups"]["requests"] == [2, 2]
    bounds = fast.native_window_forecast.metric_bounds
    for name, value in observed.items():
        assert bounds[name]["lower"] <= value <= bounds[name]["upper"], name
    assert fast.native_window_forecast.metrics["complete_fresh_requests"] == complete
    # The step ends are widened by the boundary rows their DRAM banks start from.
    step = fast.approximation["error_budget"]["step"]
    assert min(step["entry_first_access_ns"]) > 0 and min(step["exit_first_access_ns"]) > 0
    # Same-class address differences of unsampled layers are an explicit, unbounded assumption.
    assert fast.approximation["error_budget"]["assumptions"] == [stationary.SAME_CLASS_ADDRESS_ASSUMPTION]
