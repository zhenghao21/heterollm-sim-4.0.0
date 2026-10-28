"""Paper-style deployment search over native fast candidates."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exploring_hbf_native_search as search
import run_exploring_hbf_native as run

SLO = 100e6
TP8 = {"dp": 1, "tp": 8, "pp": 1, "ep": 1}
DP2_TP4 = {"dp": 2, "tp": 4, "pp": 1, "ep": 1}


def test_search_modes_are_fixed_and_default_to_max_batch():
    assert search.DEFAULT_SEARCH_MODE == "max_batch_under_slo"
    assert search.search_policy("throughput").objective == "per_gpu_throughput"
    assert search.search_policy().objective == "maximum_batch_under_slo"


def entry(layout=TP8, candidate=1555, line=(11.4e6, .0448e6), capacity=None):
    return {"layout": layout, "quantum": layout["ep"], "bypass": {"candidate_batch": candidate,
            "capacity_max_batch": capacity or candidate, "tpot_ns": 81e6, "per_gpu_tokens_s": 2399.,
            "tpot_line_ns": line}}


def linear(batch):
    """405B HBM4 SHORT TP8 native TPOT, as measured at B=512 and B=1024."""
    return (7.1 + .1017 * batch) * 1e6


def fake_row(batch, tpot, wall, gpus=8, factor=1, peak=None):
    estimate = tpot(batch)
    upper = estimate * 1.002
    return {"status": "completed" if upper <= SLO else "slo_rejected", "wall_seconds": wall(batch) * factor,
            "wall_factor": factor, "peak_memory_bytes": peak and peak(batch),
            "diagnostics": {"metric_bounds": {"mean_complete_request_tpot_ns": {
                "lower": estimate * .998, "estimate": estimate, "upper": upper}}},
            "metrics": {"tokens_per_second_per_gpu": batch * 1e9 / (estimate * gpus)} if upper <= SLO else None}


def native(tpot, wall, too_big=lambda batch: None, gpus=8, factor=1, peak=None, capacity=None):
    """A fake native layout whose saved rows appear as the search runs them.

    A rejected row reports ``capacity`` as its native capacity batch.
    """
    rows = {}

    def execute(batch):
        rejected = too_big(batch)
        rows[batch] = ({"status": rejected[0], "rejection_reason": rejected[1], "wall_seconds": 1.,
                        "native_capacity_max_batch": capacity} if rejected
                       else fake_row(batch, tpot, wall, gpus, factor, peak))
    return rows, execute


def drive(entries, layouts, cap=1800., memory=None, policy=search.THROUGHPUT_POLICY):
    runs = []
    while True:
        plan = search.plan_case(entries, [rows.get for rows, _ in layouts], SLO, cap,
                                [entry["layout"]["tp"] * entry["layout"]["ep"] for entry in entries],
                                memory_budget_bytes=memory, policy=policy)
        if not plan["next"]:
            return plan, runs
        for layout, batch in plan["next"]:
            index = next(i for i, item in enumerate(entries) if item["layout"] == layout)
            layouts[index][1](batch)
            runs.append((index, batch))


def selected(plan, layouts):
    rows = [(rows[batch], index, batch) for index, (rows, _) in enumerate(layouts) for batch in rows
            if rows[batch]["status"] == "completed"]
    return max(rows, key=lambda item: item[0]["metrics"]["tokens_per_second_per_gpu"])


def test_two_points_bracket_the_slo_and_one_acceptance_settles_the_best_layout():
    entries = [entry(), entry(DP2_TP4, candidate=353, line=(19.2e6, .0592e6), capacity=700)]
    layouts = [native(linear, lambda b: 1.5 * b), native(lambda b: (12 + .2 * b) * 1e6, lambda b: 2. * b, gpus=4)]
    plan, runs = drive(entries, layouts)
    first = plan["layouts"][0]
    # The first point halves the bypass batch, capped at 512; the second
    # scales the bypass TPOT line by the native/bypass ratio.
    assert [point["batch"] for point in first["points"]][0] == 512 and len(first["points"]) == 2
    assert first["outcome"]["kind"] == "interpolated"
    # Linear native TPOT: the interpolated batch is exactly the last one
    # whose upper bound meets the SLO.
    assert plan["acceptance"]["layout"] == TP8 and plan["acceptance"]["batch"] == 911
    assert linear(911) * 1.002 <= SLO < linear(912) * 1.002
    assert plan["acceptance"]["status"] == "completed"
    # One acceptance run, on the better layout only.
    assert runs.count((0, 911)) == 1
    assert [batch for index, batch in runs if index == 1] == [run["batch"] for run in plan["layouts"][1]["runs"]]
    assert plan["layouts"][1]["outcome"]["kind"] == "interpolated"
    row, index, batch = selected(plan, layouts)
    assert (index, batch) == (0, 911) and search.search_result(plan, batch, TP8) == "slo_accepted"


def test_a_layout_beyond_native_reach_stops_at_its_largest_point_without_extrapolation():
    # 100 ms only near B=2000, but the wall cap allows ~830.
    layouts = [native(lambda b: (7 + .0465 * b) * 1e6, lambda b: 1.95 * b)]
    plan, runs = drive([entry()], layouts)
    outcome = plan["layouts"][0]["outcome"]
    assert [batch for _index, batch in runs] == [512, 830]
    assert outcome == {"kind": "limited", "reason": "native_wall_limited", "batch": 830}
    assert plan["acceptance"]["kind"] == "limited" and "status" not in plan["acceptance"]
    row, _index, batch = selected(plan, layouts)
    assert batch == 830 and search.search_result(plan, batch, TP8) == "native_wall_limited"


def test_a_too_big_first_point_is_halved():
    wall_exhausted = ("forecast_rejected", "native_total_candidate_wall_budget_exhausted")
    layouts = [native(linear, lambda b: 4. * b, too_big=lambda b: wall_exhausted if b >= 400 else None)]
    plan, runs = drive([entry()], layouts)
    batches = [batch for _index, batch in runs]
    assert batches[:2] == [512, 256] and len(batches) <= search.MAX_RUNS
    assert all(batch < 400 for batch in batches[1:])
    assert plan["layouts"][0]["outcome"]["kind"] == "limited"
    assert plan["layouts"][0]["outcome"]["reason"] == "native_wall_limited"


@pytest.mark.parametrize("reason", ["native_stationary_refill_not_always_ready",
                                   "native_pipeline_refill_not_always_ready"])
def test_refill_not_ready_is_too_big_and_marks_import_limit(reason):
    row = {"status": "forecast_rejected", "rejection_reason": reason}
    assert search.classify(row) == "too_big"
    runs = [{"batch": 512, "kind": "too_big", "status": row["status"], "reason": reason, "capacity": None}]
    points = [{"batch": 256, "tpot_upper_ns": 40e6, "tpot_ns": 39e6,
               "native_wall_seconds": 100., "peak_memory_bytes": None}]
    limit, label = search._limit(entry(), runs, points, (1800., None), lambda value: int(value))
    assert limit == 511 and label == "native_import_limited"


def test_pipeline_budget_rejection_is_retried_then_search_continues_to_larger_batch():
    """A cached PP budget rejection must not freeze the layout at B=512."""
    layout = TP8 | {"pp": 2, "tp": 4}
    rows = {512: fake_row(512, lambda _batch: 50e6, lambda _batch: 1.)}
    rejected = {"status": "forecast_rejected",
                "rejection_reason": "native_pipeline_independent_budget_exceeded"}
    retried = set()

    def lookup(batch):
        row = rows.get(batch, rejected)
        if row and search.forecast_retryable(row) and batch not in retried:
            retried.add(batch)
            return None
        return row

    candidate = entry(layout, candidate=2048, line=(0., .228e6), capacity=4096)
    first = search.plan_case([candidate], [lookup], SLO, 1800., [8], policy=search.MAX_BATCH_POLICY)
    assert first["next"] and first["next"][0][1] > 512
    # The fresh attempt succeeds under the wider budget.  The normal planner
    # must then schedule a point above the reused B=1024 evidence.
    retry_batch = first["next"][0][1]
    rows[retry_batch] = fake_row(retry_batch, lambda _batch: 60e6, lambda _batch: 1.)
    second = search.plan_case([candidate], [lookup], SLO, 1800., [8], policy=search.MAX_BATCH_POLICY)
    assert second["next"] and second["next"][0][1] > 1024


@pytest.mark.parametrize("reason", [
    "native_pipeline_independent_budget_exceeded",
    "native_pipeline_activation_transfer_sample_missing",
    "native_pipeline_ingress_ttft_unavailable",
])
def test_only_pipeline_forecast_quality_rejections_are_retryable(reason):
    assert search.forecast_retryable({"status": "forecast_rejected", "rejection_reason": reason})
    assert not search.forecast_retryable({"status": "forecast_rejected",
                                          "rejection_reason": "native_pipeline_independent_budget_exceeded:capacity"})
    assert not search.forecast_retryable({"status": "forecast_rejected",
                                          "rejection_reason": "unable to allocate 1 byte in capacity 2"})
    assert not search.forecast_retryable({"status": "forecast_rejected",
                                          "rejection_reason": "executed task DAG could not be compacted"})


def test_reused_pipeline_budget_rejection_is_retried_once(tmp_path):
    case = run.case_matrix()[0]
    layout = run.make_layout(1, 4, 2, 1)
    params = run.parameters(case, layout, 512, "fast")
    identity = run.params_candidate_id(params)
    run.save_candidate(tmp_path, {"parameters": params, "candidate_id": identity,
                                  "backend": run.BACKEND, "status": "forecast_rejected",
                                  "rejection_reason": "native_pipeline_independent_budget_exceeded"})
    retry_ids = set()
    lookup = run._search_lookup(tmp_path, case, layout, retry_ids=retry_ids)
    assert lookup(512) is None and identity in retry_ids
    # The same rejected result is terminal after its one fresh attempt; a
    # failed retry cannot make the manager spin forever.
    assert lookup(512)["status"] == "forecast_rejected"

    physical = {"parameters": params, "candidate_id": identity,
                "backend": run.BACKEND, "status": "forecast_rejected",
                "rejection_reason": "unable to allocate 1 byte in capacity 2"}
    run.save_candidate(tmp_path, physical)
    assert run._search_lookup(tmp_path, case, layout, retry_ids=set())(512)["rejection_reason"].startswith("unable")


def test_quality_rejection_probes_larger_batches_without_becoming_capacity_evidence():
    layout = TP8 | {"pp": 2, "tp": 4}
    candidate = entry(layout, candidate=2048, capacity=4096)
    rows = {
        512: {"status": "forecast_rejected",
              "rejection_reason": "native_pipeline_independent_budget_exceeded"},
        1024: {"status": "forecast_rejected",
               "rejection_reason": "native_pipeline_independent_budget_exceeded"},
    }
    first = search.plan_layout(candidate, rows.get, SLO, 1800., policy=search.MAX_BATCH_POLICY)
    assert first["next"] == 2048
    assert [run["kind"] for run in first["runs"]] == ["quality_rejected", "quality_rejected"]
    rows[2048] = {"status": "forecast_rejected",
                  "rejection_reason": "native_pipeline_independent_budget_exceeded"}
    rows[4096] = {"status": "capacity_rejected", "rejection_reason": "native KV capacity"}
    bounded = search.plan_layout(candidate, rows.get, SLO, 1800., policy=search.MAX_BATCH_POLICY)
    assert bounded["outcome"]["kind"] == "no_native_measurement"


def test_capacity_rejection_bisects_below_the_rejected_batch():
    capacity = ("capacity_rejected", None)
    layouts = [native(lambda b: (7 + .05 * b) * 1e6, lambda b: .5 * b, too_big=lambda b: capacity if b > 900 else None)]
    plan, runs = drive([entry()], layouts)
    batches = [batch for _index, batch in runs]
    assert batches[0] == 512 and batches[1] > 900 and 512 < batches[2] < batches[1]
    assert plan["layouts"][0]["outcome"]["kind"] == "limited"


def test_bypass_capacity_bounds_points_until_a_native_point_reaches_it():
    full = ("forecast_rejected", "native_stationary_first_cohort_not_a_full_batch")
    layouts = [native(lambda b: (12 + .2 * b) * 1e6, lambda b: 2. * b, too_big=lambda b: full if b > 353 else None,
                      gpus=4)]
    plan, runs = drive([entry(DP2_TP4, candidate=353, line=(19.2e6, .0592e6))], layouts)
    # 100 ms lies near B=440: the second point stops at the bypass capacity;
    # past it rejections without a reported native capacity bisect below themselves.
    assert [batch for _index, batch in runs] == [176, 353, 439, 396]
    assert plan["layouts"][0]["outcome"] == {"kind": "limited", "reason": "native_capacity_limited", "batch": 353}
    assert search.search_result(plan, 353, DP2_TP4) == "native_capacity_limited"


def test_a_kv_capacity_rejection_sends_the_next_point_to_the_native_capacity():
    full = ("forecast_rejected", "native_stationary_first_cohort_not_a_full_batch")
    layouts = [native(lambda b: (12 + .2 * b) * 1e6, lambda b: 2. * b, too_big=lambda b: full if b > 330 else None,
                      gpus=4, capacity=330)]
    plan, runs = drive([entry(DP2_TP4, candidate=353, line=(19.2e6, .0592e6))], layouts)
    # No bisection: the rejection at the bypass capacity names the native one.
    assert [batch for _index, batch in runs] == [176, 353, 330]
    assert plan["layouts"][0]["outcome"] == {"kind": "limited", "reason": "native_capacity_limited", "batch": 330}
    # A too-big first point jumps there too instead of halving.
    layouts = [native(linear, lambda b: 1.5 * b, too_big=lambda b: full if b > 300 else None, capacity=300)]
    plan, runs = drive([entry()], layouts)
    assert [batch for _index, batch in runs] == [512, 300]
    assert plan["layouts"][0]["outcome"] == {"kind": "limited", "reason": "native_capacity_limited", "batch": 300}


def test_measured_wall_projects_as_measured_and_memory_bounds_the_batch():
    # A point slowed by concurrent workers projects its wall as measured: the
    # k-fold cap it ran under does not stretch the projection.
    shared = native(linear, lambda b: 1.5 * b, factor=2)
    plan, runs = drive([entry()], [shared])
    assert [batch for _index, batch in runs] == [512]
    assert plan["layouts"][0]["outcome"] == {"kind": "limited", "reason": "native_wall_limited", "batch": 512}
    assert [batch for _index, batch in drive([entry()], [native(linear, lambda b: 1.5 * b)])[1]] == [512, 1038, 911]
    # A peak of 100 MiB per request in a 64 GiB budget stops near B=655.
    layouts = [native(linear, lambda b: .5 * b, peak=lambda b: b * 100 * 2**20)]
    plan, runs = drive([entry()], layouts, memory=64 * 2**30)
    assert all(batch <= 655 for _index, batch in runs)
    assert plan["layouts"][0]["outcome"]["reason"] == "native_memory_limited"


def test_a_failed_acceptance_falls_back_to_the_best_measured_passing_point():
    # The acceptance batch runs 5% slower than the line through the points.
    layouts = [native(lambda b: linear(b) * (1.05 if 850 < b < 950 else 1.), lambda b: 1.5 * b)]
    plan, runs = drive([entry()], layouts)
    assert plan["acceptance"]["batch"] == 911 and plan["acceptance"]["status"] == "slo_rejected"
    row, _index, batch = selected(plan, layouts)
    assert batch == 512 and search.search_result(plan, batch, TP8) == "acceptance_failed_fallback"


def test_screen_hands_native_the_two_best_layouts_it_can_run():
    def row(dp, tp, pp, ep, micro=1, status="screened_candidate", rate=1.):
        return {"parallelism": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "microbatches": micro}, "status": status,
                "batch_quantum": ep * micro, "candidate_batch": 64, "capacity_max_batch": 128,
                "screen_tpot_seconds": .09, "screen_per_gpu_tokens_s": rate,
                "trials": [{"batch_size": 1, "tpot_seconds": .01}, {"batch_size": 64, "tpot_seconds": .09}]}

    rows = [row(1, 1, 1, 8), row(1, 4, 2, 1, micro=2), row(1, 4, 2, 1), row(1, 8, 1, 1), row(2, 4, 1, 1),
            row(1, 2, 1, 4)]
    chosen, skipped = search.screen_layouts(rows, lambda layout, quantum: "experts" if layout["ep"] == 8 else None)
    assert [item["layout"] for item in chosen] == [
        {"dp": 1, "tp": 4, "pp": 2, "ep": 1, "microbatches": 2},
        {"dp": 1, "tp": 4, "pp": 2, "ep": 1}]
    assert [item["quantum"] for item in chosen] == [2, 1]
    assert skipped == [{"layout": {"dp": 1, "tp": 1, "pp": 1, "ep": 8}, "reason": "experts"}]
    assert chosen[0]["bypass"]["tpot_line_ns"] == pytest.approx((10e6 - 80e6 / 63, 80e6 / 63))


def test_pp_only_screen_excludes_historical_pp1_layouts():
    def row(dp, tp, pp, ep, micro=1, rate=1.):
        return {"parallelism": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "microbatches": micro},
                "status": "screened_candidate", "batch_quantum": ep * micro,
                "candidate_batch": 64, "capacity_max_batch": 128,
                "screen_tpot_seconds": .09, "screen_per_gpu_tokens_s": rate,
                "trials": [{"batch_size": 1, "tpot_seconds": .01}, {"batch_size": 64, "tpot_seconds": .09}]}

    rows = [row(1, 8, 1, 1), row(1, 4, 2, 1, micro=2), row(1, 4, 2, 1),
            row(2, 2, 2, 1), row(1, 8, 1, 1, micro=2)]
    chosen, _skipped = search.screen_layouts(rows, lambda _layout, _quantum: None, pp_only=True)
    assert all(search.is_pipeline_layout(item["layout"]) for item in chosen)
    assert [item["layout"] for item in chosen] == [
        {"dp": 1, "tp": 4, "pp": 2, "ep": 1, "microbatches": 2},
        {"dp": 1, "tp": 4, "pp": 2, "ep": 1}]


def test_expanded_scope_keeps_existing_layouts_and_adds_pipeline(tmp_path):
    case = run.case_matrix()[0]
    rows = [{"parallelism": {"dp": 1, "tp": 8, "pp": 1, "ep": 1, "microbatches": 1},
             "status": "screened_candidate", "batch_quantum": 1, "candidate_batch": 64,
             "capacity_max_batch": 128, "screen_tpot_seconds": .09, "screen_per_gpu_tokens_s": 1.,
             "trials": [{"batch_size": 1, "tpot_seconds": .01}, {"batch_size": 64, "tpot_seconds": .09}]},
            {"parallelism": {"dp": 1, "tp": 4, "pp": 2, "ep": 1, "microbatches": 1},
             "status": "screened_candidate", "batch_quantum": 1, "candidate_batch": 64,
             "capacity_max_batch": 128, "screen_tpot_seconds": .09, "screen_per_gpu_tokens_s": 1.,
             "trials": [{"batch_size": 1, "tpot_seconds": .01}, {"batch_size": 64, "tpot_seconds": .09}]}]
    run.case_screen(tmp_path, case, screen=lambda _case: rows, layout_scope="all")
    expanded = run.case_screen(tmp_path, case, screen=lambda _case: rows, layout_scope="expanded")
    layouts = [item["layout"] for item in expanded["layouts"]]
    assert any(layout["pp"] == 1 for layout in layouts)
    assert any(layout["pp"] > 1 for layout in layouts)


def test_pipeline_search_ignores_point_and_attempt_caps_until_physical_limit():
    pipeline = {"dp": 1, "tp": 4, "pp": 2, "ep": 1}
    rows, execute = native(lambda _batch: 10e6, lambda batch: .000001 * batch,
                           too_big=lambda batch: ("capacity_rejected", None) if batch > (1 << 20) else None,
                           gpus=8, capacity=1 << 20)
    plan, runs = drive([entry(pipeline, candidate=1 << 20, capacity=1 << 20)], [(rows, execute)],
                       policy=search.MAX_BATCH_POLICY)
    assert len(runs) > search.MAX_RUNS
    assert len(plan["layouts"][0]["points"]) > search.MAX_POINTS
    assert plan["layouts"][0]["outcome"]["reason"] == "native_capacity_limited"


def test_max_batch_policy_ignores_caps_for_non_pipeline_layout():
    rows, execute = native(lambda _batch: 10e6, lambda batch: .000001 * batch,
                           too_big=lambda batch: ("capacity_rejected", None) if batch > (1 << 20) else None,
                           gpus=8, capacity=1 << 20)
    plan, runs = drive([entry(TP8, candidate=1 << 20, capacity=1 << 20)], [(rows, execute)],
                       policy=search.MAX_BATCH_POLICY)
    assert len(runs) > search.MAX_RUNS
    assert len(plan["layouts"][0]["points"]) > search.MAX_POINTS


def test_pipeline_search_selects_largest_batch_before_throughput():
    first = {"dp": 1, "tp": 4, "pp": 2, "ep": 1}
    second = {"dp": 1, "tp": 2, "pp": 4, "ep": 1}
    first_rows, first_execute = native(lambda _batch: 50e6, lambda batch: .0001 * batch,
                                       gpus=8, capacity=1000)
    second_rows, second_execute = native(lambda _batch: 1e6, lambda batch: .0001 * batch,
                                         gpus=8, capacity=900)
    plan, _runs = drive([entry(first, candidate=1000, capacity=1000),
                         entry(second, candidate=900, capacity=900)],
                        [(first_rows, first_execute), (second_rows, second_execute)],
                        policy=search.MAX_BATCH_POLICY)
    assert plan["acceptance"]["layout"] == first
    assert plan["acceptance"]["batch"] == plan["layouts"][0]["outcome"]["batch"]
    assert plan["layouts"][0]["outcome"]["batch"] > plan["layouts"][1]["outcome"]["batch"]


def test_a_case_selection_runs_only_those_cases_in_matrix_order(tmp_path, monkeypatch):
    matrix = run.case_matrix()
    chosen = [matrix[5]["case_id"], matrix[0]["case_id"]]
    assert [case["case_id"] for case in run.selected_cases(chosen)] == [matrix[0]["case_id"], matrix[5]["case_id"]]
    assert run.selected_cases() == matrix
    for wrong in (["native__nothing"], []):
        with pytest.raises(ValueError, match="unknown or empty case selection"):
            run.selected_cases(wrong)
    calls = []
    monkeypatch.setattr(run, "run_search_manager", lambda directory, **options: calls.append(options) or 0)
    assert run.main(["--search", "--directory", str(tmp_path), "--workers", "1", "--cases", chosen[0]]) == 0
    assert calls == [{"workers": 1, "case_ids": [chosen[0]]}]


def test_search_manager_runs_each_point_once_and_reports_the_accepted_batch(tmp_path, monkeypatch):
    cases = run.case_matrix()[:2]
    monkeypatch.setattr(run, "available_memory_bytes", lambda: 1 << 50)
    bypass = [{"parallelism": {"dp": 1, "tp": 8, "pp": 1, "ep": 1, "microbatches": 1}, "status": "screened_candidate",
               "batch_quantum": 1, "candidate_batch": 1555, "capacity_max_batch": 1555,
               "screen_tpot_seconds": .081, "screen_per_gpu_tokens_s": 2399.,
               "trials": [{"batch_size": 1, "tpot_seconds": .0114}, {"batch_size": 1555, "tpot_seconds": .081}]}]
    spawned = []

    class Process:
        pid = 1

        def __init__(self, case, layout, batch):
            self.case, self.layout, self.batch = case, layout, batch

        def poll(self):
            params = run.parameters(self.case, self.layout, self.batch, "fast")
            row = {**fake_row(self.batch, linear, lambda b: 1.5 * b), "parameters": params, "backend": run.BACKEND,
                   "runner_contract": run.FAST_CONTRACT, "candidate_id": run.params_candidate_id(params)}
            if row["metrics"]:
                row["metrics"].update(cluster_tokens_per_second=row["metrics"]["tokens_per_second_per_gpu"] * 8,
                                      mean_complete_request_tpot_ns=linear(self.batch))
            run.write_json(tmp_path / "candidates" / (row["candidate_id"] + ".json"), row)
            return 0

    def spawn(case, layout, batch, directory):
        spawned.append((case["case_id"], batch))
        return Process(case, layout, batch)

    selection = [case["case_id"] for case in cases]
    assert run.run_search_manager(tmp_path, workers=1, spawn=spawn, poll_seconds=0,
                                  screen=lambda case: bypass, case_ids=selection) == 0
    for case in cases:
        first, second, acceptance = [batch for identity, batch in spawned if identity == case["case_id"]]
        assert (first, acceptance) == (512, 911) and linear(second) > SLO
        point = run.read_json(tmp_path / "points" / (case["case_id"] + ".json"))
        assert point["status"] == "completed" and point["selected"]["parameters"]["batch_size"] == 911
        assert point["search"]["result"] == "slo_accepted"
        assert point["search"]["acceptance"]["batch"] == 911
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "部署按论文式搜索" in report and "插值到100ms并通过验收" in report
    manifest = run.read_json(tmp_path / "run_manifest.json")
    assert manifest["deployment_search"] == search.SETTINGS and manifest["simulation_mode"] == "fast"
    assert manifest["cases"] == cases
    # A resumed search replays the saved rows and runs nothing again.
    assert run.run_search_manager(tmp_path, workers=1, spawn=lambda *a: pytest.fail("rerun"), poll_seconds=0,
                                  screen=lambda case: pytest.fail("screen recomputed"), case_ids=selection) == 0
    # The directory's case selection cannot change silently.
    with pytest.raises(RuntimeError, match="manifest differs"):
        run.run_search_manager(tmp_path, workers=1, spawn=lambda *a: pytest.fail("started"), poll_seconds=0,
                               screen=lambda case: bypass, case_ids=selection[:1])
