"""Native stage constraints at a real later Prefill chunk boundary."""
from dataclasses import replace

import pytest

from heterollm_sim import native_prefill
from heterollm_sim.native_stationary import cohort_layers, compose_layers, prefill_stage_ready, run_window
from heterollm_sim.reporting import run_scenario
from heterollm_sim.simulation_options import NativePrefillPassConfig, SimulationConfig
from test_native_prefill import _chunked_prefill


@pytest.mark.parametrize("ep", [1, 2])
def test_later_prefill_chunk_preserves_native_ready_and_layer_completion(monkeypatch, ep):
    scenario = _chunked_prefill(ep=ep)
    runtime = native_prefill._OnlineRuntime
    original_schedule = runtime._schedule_execution_stages
    original_replay = runtime._replay_authored_token_program
    seen = {}

    def replay(self, stages, ready_by_stage, namespace, program):
        if len(self.batches) == 2:
            seen["actual_ready"] = dict(ready_by_stage)
        return original_replay(self, stages, ready_by_stage, namespace, program)

    def schedule(self, cohort, cost, stages, host_ns, device_ns, request_ready_ns, *args, **kwargs):
        terminal = len(self.batches) == 2
        if terminal:
            ready = prefill_stage_ready(self, cohort, stages, request_ready_ns, host_ns=host_ns)
            seen["prepared_ready"] = ready
            prepared = cohort_layers(self, cohort, stages, request_ready_ns,
                "native-prefill-real-ready.", ready_by_stage=ready)
            seen["prepared"] = prepared
            ids = [row["layer_id"] for row in prepared["inventory"]]
            boundary = prepared["boundary"]
            seen["full"] = run_window(prepared, self.execution_control, [], "prefill_pass", ids,
                (boundary.get("before_first_layer", []), boundary.get("after_last_layer", [])),
                prepared["origin"], "later-chunk-full")
            sampled = native_prefill.sample_prefill(prepared, self.execution_control,
                NativePrefillPassConfig(calibration_layers=3, holdout_layers=3), [])
            seen["bounds"] = compose_layers(prepared["inventory"], sampled["runs"], exit=False)[:2]
            seen["entry_second"] = sampled["runs"]["entry_at_holdout_origin"]
        outcome = original_schedule(self, cohort, cost, stages, host_ns, device_ns,
                                    request_ready_ns, *args, **kwargs)
        if terminal:
            seen["actual_layers"] = self._live_cohort_metrics.layer_completion_ns()
        return outcome

    monkeypatch.setattr(runtime, "_schedule_execution_stages", schedule)
    monkeypatch.setattr(runtime, "_replay_authored_token_program", replay)
    detailed = run_scenario(replace(scenario, simulation=SimulationConfig(mode="detailed")),
                            retention_policy="aggregate").serving
    assert len(detailed.batches) == 3
    assert {key: value.hex() for key, value in seen["prepared_ready"].items()} == {
        key: value.hex() for key, value in seen["actual_ready"].items()}
    assert {key: value.hex() for key, value in seen["full"]["layer_completion_ns"].items()} == {
        key: value.hex() for key, value in seen["actual_layers"].items()}
    assert seen["full"]["end_ns"].hex() == detailed.batches[-1].end_ns.hex()
    assert seen["full"]["start_ns"] < seen["prepared"]["origin"]
    assert seen["entry_second"]["start_ns"] < seen["entry_second"]["origin_ns"]
    actual_service = seen["actual_layers"]["layer-007"] - seen["prepared"]["origin"]
    lower, upper = seen["bounds"]
    assert lower <= actual_service <= upper
