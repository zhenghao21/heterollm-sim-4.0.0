from heterollm_sim import llama_scenario, run_jobs
from heterollm_sim.reference import build_llama_default_scenario


def test_diagnostic_failure_cannot_leave_finished_worker_running(monkeypatch):
    scenario = build_llama_default_scenario()
    monkeypatch.setattr(llama_scenario, "prepare_llama_scenario", lambda value: value)
    monkeypatch.setattr(
        run_jobs,
        "estimate_scenario",
        lambda _scenario: {"recommended_retention_policy": "aggregate"},
    )

    def fail_simulation(*_args, **_kwargs):
        raise ValueError("physical execution failed")

    def fail_diagnostic(*_args, **_kwargs):
        raise OSError("stderr pipe closed")

    monkeypatch.setattr(run_jobs, "run_scenario", fail_simulation)
    monkeypatch.setattr(run_jobs, "record_unexpected_exception", fail_diagnostic)

    with run_jobs.RunJobManager(max_workers=1) as manager:
        job_id = manager.submit(scenario)
        manager._jobs[job_id].future.result(timeout=5)
        snapshot = manager.get(job_id)

    assert snapshot["status"] == run_jobs.FAILED
    assert snapshot["finished_at"] is not None
    assert snapshot["progress"]["stage"] == run_jobs.FAILED
    assert snapshot["error"]["exception_type"] == "ValueError"
    assert snapshot["error"]["message"] == "physical execution failed"
    assert snapshot["error"]["diagnostic_error"] == {
        "exception_type": "OSError",
        "message": "stderr pipe closed",
    }
