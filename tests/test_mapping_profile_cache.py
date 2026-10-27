"""Profile conversion is shared only within one unchanged placement run."""
from collections import Counter
from dataclasses import replace

from heterollm_sim import control_plane_planner, planner
from heterollm_sim.reference import build_reference_scenario


def test_mapping_profile_pairs_are_reused_and_new_runs_read_changed_profiles(monkeypatch):
    scenario = build_reference_scenario()
    original = planner._gpu_profiles
    calls = Counter()

    def counted(scenario, gpu, memory=None):
        calls[gpu, memory] += 1
        return original(scenario, gpu, memory)

    monkeypatch.setattr(planner, "_gpu_profiles", counted)
    first = control_plane_planner.plan_runtime_placement(scenario)
    assert calls and max(calls.values()) == 1
    component = scenario.hardware.get_component("gpu0")
    registry = scenario.component_profiles["gpu"]
    registry[component.cost_profile_id] = replace(registry[component.cost_profile_id], kernel_launch_ns=12345.)
    calls.clear()
    cached = control_plane_planner.plan_runtime_placement(scenario)
    assert calls and max(calls.values()) == 1

    class Uncached(dict):
        def get(self, key, default=None):
            return default

    context = control_plane_planner._MappingRunContext

    def uncached_context(*args, **kwargs):
        return context(*args, **kwargs, gpu_profile_pairs=Uncached())

    monkeypatch.setattr(control_plane_planner, "_MappingRunContext", uncached_context)
    calls.clear()
    expected = control_plane_planner.plan_runtime_placement(scenario)
    assert max(calls.values()) > 1
    # Solver elapsed time is the sole nondeterministic field.
    assert replace(cached, elapsed_s=0.) == replace(expected, elapsed_s=0.)
    assert first.decisions != cached.decisions
