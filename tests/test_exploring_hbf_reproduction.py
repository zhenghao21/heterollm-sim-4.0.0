"""Runner/reporter boundaries use mocked events, never the formal matrix."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import reproduce_exploring_hbf as runner
import exploring_hbf_reproduction_report as report


def test_capacity_refinement_adds_only_new_physical_candidates_and_reuses_results(tmp_path, monkeypatch):
    from dataclasses import asdict
    runs = []

    class Engine:
        def __init__(self, *args, batch_size, assumptions, **kwargs):
            self.batch, self.assumptions = batch_size, assumptions
            if batch_size > 9 and assumptions.prepared_lookahead > 2:
                raise runner.PaperMemoryCapacityError("one request exceeds prepared capacity")

        def import_kv(self, request, now):
            return SimpleNamespace(ready_ns=now+1000.)

        def run(self, **kwargs):
            runs.append(self.batch)
            return _result(research_assumptions=asdict(self.assumptions))

    candidate = {**_candidate(), "candidate_batch": 10}
    monkeypatch.setattr(runner, "search_candidates", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(runner, "PaperInferenceEngine", Engine)
    monkeypatch.setattr(runner, "maximum_capacity_batch", lambda *args, **kwargs: {"batch_size": 9})
    directory = _directory(tmp_path)
    result = runner.run_case(_case(), directory)
    assert result["status"] == "completed" and runs == [9]
    assert result["selected"]["batch_size"] == 9
    assert result["validations"][0]["capacity_refinement"]["batch_size"] == 9
    path = directory/"points"/(runner.case_id(_case())+".json")
    result["search_policy"] = "old_coarse_capacity_policy"
    path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(runner, "search_candidates", lambda *args, **kwargs: pytest.fail("cached screening must be reused"))
    refined = runner.run_case(_case(), directory)
    assert refined["status"] == "completed" and runs == [9]
    assert refined["reused_validation_count"] == 1
    assert refined["selected"]["result"] == result["selected"]["result"]


def test_primary_cases_explicitly_select_author_peak_performance():
    assert all(case["assumptions"] == {"matrix_efficiency": 1., "scalar_efficiency": 1.}
               for case in runner.experiments("all"))


def test_report_separates_author_peak_primary_from_efficiency_control(tmp_path):
    directory = _directory(tmp_path)
    text, _ = report._markdown(directory, {"cases": []}, [], [], [],
                               report.eight_gpu_comparison([]))
    assert "主实验矩阵与标量效率均设为1" in text
    assert "旧60%效率探索记录单独保留" in text
    assert "HBF-for-LLM-cal26-talk.pdf" in text
    assert "提前准备窗口由实际PD费用推导后" in text


def test_core_scope_excludes_deferred_cases_and_embeds_interpretation_only_when_complete(tmp_path):
    directory = _directory(tmp_path)
    record = _point(directory, "completed", selected=_selected())
    (directory / "interpretation.md").write_text("## 核心对照解释\n\n这是独立对照结论。", encoding="utf-8")
    rows, records, errors = report.load_terminal_rows(directory)
    deferred = [{**_case("offline"), "gpus": value} for value in (1, 2, 4, 16)]
    manifest = {"cases": [record["case"]], "deferred_cases": deferred}
    text, status = report._markdown(directory, manifest, rows, records, errors,
                                    report.eight_gpu_comparison(rows))
    assert status["complete"] and status["expected"] == 1
    assert "另4个GPU数量或SLO目标的探索配置未验收" in text
    assert "部分初始工作已隔离保留，不纳入对照" in text
    assert "只有后续明确请求才运行" in text
    assert "这是独立对照结论" in text
    manifest["cases"].append({**_case(), "hardware": "HBF"})
    text, status = report._markdown(directory, manifest, rows, records, errors,
                                    report.eight_gpu_comparison(rows))
    assert not status["complete"] and "这是独立对照结论" not in text


def test_eight_gpu_core_plots_actual_metrics_as_storage_bars(tmp_path, monkeypatch):
    import matplotlib.axes
    import matplotlib.figure
    bars, files = [], []
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", lambda self, x, height, **kwargs: bars.append(height))
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", lambda *args, **kwargs: pytest.fail("core scope must not draw a GPU-scaling curve"))
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, path, **kwargs: files.append(path.name))
    row = {"variant": "primary", "screen_slo_seconds": .1, "slo_statistic": "mean_iteration", "status": "completed",
           "model": "llama3_405b", "workload": "SHORT", "hardware": "HBM4", "gpus": 8,
           "per_gpu_tokens_s": 1234., "max_validated_batch_per_gpu": 194.}
    report._plots(tmp_path, [row], scope="eight_gpu_core")
    assert bars == [1234., 194.]
    assert files == ["throughput.png", "batch.png"]


def _directory(tmp_path):
    (tmp_path / "points").mkdir(exist_ok=True, parents=True)
    return tmp_path


def _case(slo=.1, *, statistic=None):
    case = {"model": "llama3_405b", "hardware": "HBM4", "workload": "SHORT", "gpus": 8, "slo": slo}
    if statistic is not None:
        case["slo_statistic"] = statistic
    return case


def _result(**updates):
    result = {"status": "completed", "per_gpu_batch_size": 1,
              "per_gpu_decode_tokens_per_second": 10., "cluster_decode_tokens_per_second": 80.,
              "mean_tpot_ns": 1_000_000, "mean_iteration_ns": 9_000_000, "max_iteration_ns": 10_000_000,
              "measurement_start_ns": 1e9, "measurement_end_ns": 2e9, "measured_steps": 3,
              "completed_request_elapsed_count": 2, "max_completed_request_elapsed_ns": 100_000_000,
              "research_assumptions": {"capacity_unit_bytes": 10**9}}
    result.update(updates)
    return result


def _selected(result=None):
    return {"accepted": True, "batch_size": 8,
            "parallelism": {"dp": 1, "pp": 1, "tp": 8, "ep": 1, "microbatches": 1},
            "result": result or _result()}


def _point(directory, status, *, case=None, selected=None, screening=None):
    case = case or _case()
    record = {"case_id": "point", "case": case, "status": status, "selected": selected}
    if screening is not None:
        record["screening"] = screening
    (directory / "points/point.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def _candidate():
    return {"status": "screened_candidate", "candidate_batch": 1, "screen_tpot_seconds": .01,
            "parallelism": {"dp": 1, "pp": 1, "tp": 1, "ep": 1, "microbatches": 1}}


def _mock_runner(monkeypatch, result):
    class Engine:
        def _kv_stage_bytes(self, stage, context):
            return 1024

        def __init__(self, *args, **kwargs):
            pass

        def import_kv(self, request, now):
            return SimpleNamespace(ready_ns=now + 1000.)

        def run(self, **kwargs):
            return result

    monkeypatch.setattr(runner, "PaperInferenceEngine", Engine)
    monkeypatch.setattr(runner, "search_candidates", lambda *args, **kwargs: [_candidate()])


def test_search_execution_error_cannot_be_recorded_as_hardware_infeasible(tmp_path, monkeypatch):
    directory = _directory(tmp_path)
    candidates = [_candidate(), {"status": "execution_failed", "reason": "RuntimeError: executor fault"}]
    monkeypatch.setattr(runner, "search_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(runner, "PaperInferenceEngine", lambda *args, **kwargs: pytest.fail("must not validate after search failure"))
    record = runner.run_case(_case(statistic="max_iteration"), directory)
    assert record["status"] == "failed"
    assert record["screening_failures"] == candidates[1:]
    assert record.get("selected") is None


def test_report_reclassifies_legacy_infeasible_search_error_as_failed(tmp_path):
    directory = _directory(tmp_path)
    _point(directory, "infeasible", screening=[{"status": "execution_failed"}])
    rows, _, errors = report.load_terminal_rows(directory)
    assert not errors and rows[0]["status"] == "failed"
    assert rows[0]["per_gpu_tokens_s"] is None


def test_screened_but_unaccepted_runner_is_no_validated_deployment(tmp_path, monkeypatch):
    directory = _directory(tmp_path)
    _mock_runner(monkeypatch, _result(max_iteration_ns=1e9))
    record = runner.run_case(_case(statistic="max_iteration"), directory)
    assert record["status"] == "no_validated_deployment"
    assert record["selected"] is None and record["validations"]
    rows, _, _ = report.load_terminal_rows(directory)
    assert rows[0]["status"] == "no_validated_deployment"
    assert rows[0]["batch_per_gpu"] is None and rows[0]["per_gpu_tokens_s"] is None
    assert rows[0]["normalized_throughput"] is None


def test_representative_or_capacity_rejection_is_not_execution_failure(tmp_path, monkeypatch):
    directory = _directory(tmp_path)
    monkeypatch.setattr(runner, "search_candidates", lambda *args, **kwargs: [
        {"status": "capacity_rejected", "reason": "private capacity"},
        {"status": "slo_rejected", "reason": "representative event"}])
    record = runner.run_case(_case(), directory)
    assert record["status"] == "infeasible"
    rows, _, _ = report.load_terminal_rows(directory)
    assert rows[0]["status"] == "infeasible" and rows[0]["per_gpu_tokens_s"] is None


@pytest.mark.parametrize("status", ["searching", "validating"])
def test_in_progress_selected_is_excluded_even_with_finished_looking_metrics(tmp_path, status):
    directory = _directory(tmp_path)
    _point(directory, status, selected=_selected())
    rows, records, _ = report.load_terminal_rows(directory)
    assert not rows and len(records) == 1


@pytest.mark.parametrize("selected", [None, {"accepted": False, "result": _result()},
                                      {"accepted": True, "result": {"status": "running"}}])
def test_invalid_terminal_selected_never_supplies_a_metric(tmp_path, selected):
    directory = _directory(tmp_path)
    _point(directory, "completed", selected=selected)
    rows, _, _ = report.load_terminal_rows(directory)
    assert rows[0]["status"] == "invalid_terminal"
    assert rows[0]["per_gpu_tokens_s"] is None and rows[0]["normalized_batch"] is None


def test_offline_runner_uses_whole_fresh_request_deadline_not_single_step_conversion(tmp_path, monkeypatch):
    directory = _directory(tmp_path)
    # This step exceeds the conservative 24h/L_OUT screening conversion,
    # while the actually completed fresh request remains within 24 hours.
    result = _result(max_iteration_ns=300e9, max_completed_request_elapsed_ns=86399e9)
    _mock_runner(monkeypatch, result)
    record = runner.run_case(_case("offline"), directory)
    assert record["status"] == "completed" and record["selected"]["accepted"]
    assert record["selected"]["slo_statistic"].startswith("maximum completed fresh request")
    rows, _, _ = report.load_terminal_rows(directory)
    assert rows[0]["status"] == "completed"
    assert rows[0]["final_deadline_seconds"] == 86400
    assert rows[0]["max_completed_request_elapsed_seconds"] == 86399
    assert rows[0]["screen_slo_seconds"] == 86400 / 373


def test_offline_rejects_excess_residence_even_when_every_step_is_small(tmp_path, monkeypatch):
    directory = _directory(tmp_path)
    _mock_runner(monkeypatch, _result(max_iteration_ns=1e6, max_completed_request_elapsed_ns=86401e9))
    record = runner.run_case(_case("offline"), directory)
    assert record["status"] == "no_validated_deployment"
    assert record["selected"] is None


@pytest.mark.parametrize("deadline", [None, 86401e9])
def test_report_does_not_accept_old_offline_step_only_or_overdue_result(tmp_path, deadline):
    directory = _directory(tmp_path)
    _point(directory, "completed", case=_case("offline"),
           selected=_selected(_result(max_completed_request_elapsed_ns=deadline)))
    rows, _, _ = report.load_terminal_rows(directory)
    assert rows[0]["status"] == "invalid_terminal"
    assert rows[0]["per_gpu_tokens_s"] is None
    assert rows[0]["final_deadline_seconds"] == 86400


def test_no_validated_terminal_counts_processed_but_not_success_or_fake_zero(tmp_path):
    directory = _directory(tmp_path)
    record = _point(directory, "no_validated_deployment")
    rows, records, errors = report.load_terminal_rows(directory)
    comparisons = report.eight_gpu_comparison(rows)
    text, status = report._markdown(directory, {"cases": [record["case"]]}, rows, records, errors, comparisons)
    assert status["processed"] == status["no_validated_deployment"] == 1
    assert status["successful"] == status["infeasible"] == status["failed"] == 0
    assert status["complete"] is True
    assert comparisons[0]["simulated_tokens_s_per_gpu"] is None
    assert "未找到已验收部署 1" in text


def test_report_warns_if_old_success_lacks_current_host_staging_path(tmp_path):
    directory = _directory(tmp_path)
    record = _point(directory, "completed", selected=_selected())
    rows, records, errors = report.load_terminal_rows(directory)
    text, _ = report._markdown(directory, {"cases": [record["case"]]}, rows, records, errors,
                               report.eight_gpu_comparison(rows))
    assert "有1个成功结果未记录当前PD HOST暂存元数据" in text


def test_primary_mean_can_pass_while_explicit_max_control_rejects(tmp_path, monkeypatch):
    result = _result(mean_iteration_ns=50e6, max_iteration_ns=500e6)
    _mock_runner(monkeypatch, result)
    average = runner.run_case(_case(), _directory(tmp_path / "mean"))
    maximum = runner.run_case(_case(statistic="max_iteration"), _directory(tmp_path / "max"))
    assert average["status"] == "completed"
    assert maximum["status"] == "no_validated_deployment"
    assert average["selected"]["slo_statistic"].startswith("mean_iteration")
    rows, _, _ = report.load_terminal_rows(tmp_path / "mean")
    assert rows[0]["slo_statistic"] == "mean_iteration"
    assert rows[0]["mean_iteration_ms"] == 50
    assert rows[0]["max_iteration_ms"] == 500
    assert report._is_primary(rows[0])


def test_explicit_max_success_never_normalizes_into_mean_primary(tmp_path):
    directory = _directory(tmp_path)
    _point(directory, "completed", case=_case(statistic="max_iteration"), selected=_selected())
    rows, _, _ = report.load_terminal_rows(directory)
    assert rows[0]["slo_statistic"] == "max_iteration"
    assert not report._is_primary(rows[0])
    assert rows[0]["normalized_batch"] is None
    assert all(row["status"] == "missing" for row in report.eight_gpu_comparison(rows))


def test_legacy_missing_mean_statistic_is_unknown_not_an_invented_average(tmp_path):
    directory = _directory(tmp_path)
    result = _result()
    del result["mean_iteration_ns"]
    _point(directory, "completed", selected=_selected(result))
    rows, _, _ = report.load_terminal_rows(directory)
    assert rows[0]["status"] == "completed"
    assert rows[0]["slo_statistic"] == "unknown" and rows[0]["mean_iteration_ms"] is None
    assert not report._is_primary(rows[0]) and rows[0]["normalized_throughput"] is None


def test_report_preserves_program_boundary_and_pec_lower_upper_bounds(tmp_path):
    directory = _directory(tmp_path)
    case = _case()
    case["hardware"] = "HBF"
    result = _result(projected_three_year_average_pec=110.,
        measurement_program_bytes_lower_bound=1000, measurement_program_bytes_upper_bound=1500,
        projected_three_year_pec_lower_bound=100., projected_three_year_pec_upper_bound=150.)
    record = _point(directory, "completed", case=case, selected=_selected(result))
    rows, records, errors = report.load_terminal_rows(directory)
    assert rows[0]["measurement_program_bytes_lower_bound"] == 1000
    assert rows[0]["measurement_program_bytes_upper_bound"] == 1500
    assert rows[0]["projected_three_year_pec_lower_bound"] == 100.
    assert rows[0]["projected_three_year_pec_upper_bound"] == 150.
    text, _ = report._markdown(directory, {"cases": [record["case"]]}, rows, records, errors,
                               report.eight_gpu_comparison(rows))
    assert "不是逐页完成时刻的精确计数" in text
    assert "P/E下界|P/E上界" in text


def test_maximum_validated_batch_is_separate_from_throughput_selected_batch(tmp_path):
    directory = _directory(tmp_path)
    selected = _selected(_result(per_gpu_batch_size=2, per_gpu_decode_tokens_per_second=20))
    selected["batch_size"] = 16
    larger = _selected(_result(per_gpu_batch_size=4, per_gpu_decode_tokens_per_second=15))
    larger["batch_size"] = 32
    rejected = _selected(_result(per_gpu_batch_size=8, per_gpu_decode_tokens_per_second=10))
    rejected.update(batch_size=64, accepted=False)
    record = _point(directory, "completed", selected=selected)
    record["validations"] = [selected, larger, rejected]
    (directory / "points/point.json").write_text(json.dumps(record), encoding="utf-8")
    rows, _, _ = report.load_terminal_rows(directory)
    row = rows[0]
    assert row["batch_per_gpu"] == 2 and row["per_gpu_tokens_s"] == 20
    assert row["max_validated_batch_per_gpu"] == 4
    assert row["max_validated_batch_per_replica"] == 32
    assert json.loads(row["max_validated_batch_parallelism"])["tp"] == 8
    comparison = report.eight_gpu_comparison(rows)[0]
    assert comparison["throughput_selected_batch_per_gpu"] == 2
    assert comparison["max_validated_batch_per_gpu"] == 4
    assert comparison["batch_relative_error_to_rounded"] == pytest.approx(abs(4 / 194 - 1))
    assert comparison["simulated_tokens_s_per_gpu"] == 20


def test_batch_curve_normalization_uses_maxima_on_both_sides_and_throughput_uses_selected(tmp_path):
    directory = _directory(tmp_path)
    for name, hardware, selected_batch, largest_batch, tps, parallel in (
            ("hbm", "HBM4", 16, 32, 20., {"dp": 1, "pp": 1, "tp": 8, "ep": 1, "microbatches": 1}),
            ("hbf", "HBF", 12, 32, 120., {"dp": 2, "pp": 1, "tp": 4, "ep": 1, "microbatches": 1})):
        case = _case()
        case["hardware"] = hardware
        private_gpus = parallel["pp"] * parallel["tp"] * parallel["ep"]
        selected = _selected(_result(per_gpu_batch_size=selected_batch/private_gpus, per_gpu_decode_tokens_per_second=tps))
        selected.update(batch_size=selected_batch, parallelism=parallel)
        largest = _selected(_result(per_gpu_batch_size=largest_batch/private_gpus, per_gpu_decode_tokens_per_second=tps/2))
        largest.update(batch_size=largest_batch, parallelism=parallel)
        point = {"case_id": name, "case": case, "status": "completed", "selected": selected,
                 "validations": [selected, largest]}
        (directory / "points" / (name + ".json")).write_text(json.dumps(point), encoding="utf-8")
    rows, _, _ = report.load_terminal_rows(directory)
    hbf = next(row for row in rows if row["hardware"] == "HBF")
    assert hbf["batch_per_gpu"] == 3
    assert hbf["max_validated_batch_per_gpu"] == 8
    assert hbf["normalized_batch"] == 2
    assert hbf["normalized_throughput"] == 6
