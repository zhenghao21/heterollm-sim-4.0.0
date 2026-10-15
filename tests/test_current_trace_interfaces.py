"""Replay clients consume current report events without old-field synthesis."""
import json
from pathlib import Path
import re
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import VisualizationTraceOptions, RetentionPolicy
from heterollm_sim.engine import simulate_schedule
from heterollm_sim.reporting import (
    OnlineScenarioResult, ScenarioResult, _task_event, _visualization_payload, online_summary_dict,
    page_online_batch_trace,
)
from test_async_kv_prefetch import runtime
from test_backend_execution_semantics import core_schedule, manifest


def test_rank_resolution_uses_current_rank_records():
    from heterollm_sim.parallel import LogicalRank
    from heterollm_sim.reporting import _rank_payload

    rank = LogicalRank(2, "gpu2", 1, 0, 1)
    assert _rank_payload({"rank": 2}, {2: rank}) == {
        "rank": 2, "component_id": "gpu2", "tp_rank": 1, "pp_rank": 0, "ep_rank": 1}
    with pytest.raises(AttributeError):
        _rank_payload({"rank": 2}, {2: "gpu2"})


def test_static_report_resolves_compute_component_before_browser_replay():
    from heterollm_sim.reference import build_reference_scenario
    from heterollm_sim.planner import _parallel_plan

    scenario = build_reference_scenario()
    rank = _parallel_plan(scenario).ranks[0]
    schedule = core_schedule()
    compute = replace(schedule.tasks[-1], metadata={"rank": rank.rank,
        "target_component": rank.component_id})
    trace = simulate_schedule(replace(schedule, tasks=(compute,)))
    result = ScenarioResult(scenario, SimpleNamespace(trace=trace, task_count=1,
        retention_policy=RetentionPolicy.EXACT), (), "exact")
    event = _visualization_payload(result, VisualizationTraceOptions())["events"][0]
    assert event["rank"]["component_id"] == rank.component_id
    assert event["rank"]["rank"] == rank.rank


def test_replay_preserves_current_task_and_batch_events_without_legacy_fields():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser contract checks")
    runner = runtime()
    result = OnlineScenarioResult(runner.plan.scenario, runner.run(), manifest(), (), "aggregate")
    visualization = _visualization_payload(result, VisualizationTraceOptions())
    trace = simulate_schedule(core_schedule())
    tasks = [_task_event(task, lookup={}, protocols={}) for task in trace.tasks]
    replay = {"schema_version": "1.0", "events": tasks, "fidelity": "exact",
              "scope": {"batch_id": "batch", "time_range_ns": [0, trace.makespan_ns]},
              "time_range_ns": [0, trace.makespan_ns], "limitations": []}
    page = page_online_batch_trace(replay, limit=1)
    source = (Path(__file__).parents[1] / "src/heterollm_sim/webui/app.js").read_text(encoding="utf-8")
    functions = []
    for name in ("finiteTraceNumber", "normalizeTraceRank", "normalizeTraceHops", "normalizeTraceEvent",
                 "normalizedTraceBatchIndex", "traceVisualizationFromReport", "tracePagePagination", "tracePageData"):
        match = re.search(r"^function " + name + r"\([^\n]*\) \{.*?^\}", source, re.M | re.S)
        assert match, name
        functions.append(match.group())
    script = """
const assert = require('node:assert/strict');
const asObject = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const asArray = value => Array.isArray(value) ? value : [];
""" + "\n".join(functions) + "\nconst data = " + json.dumps({
        "visualization": visualization, "tasks": tasks, "page": page}) + ";\n" + """
for (const raw of [...data.tasks, ...data.visualization.events]) {
  const event = normalizeTraceEvent(raw);
  for (const field of ['event_id', 'event_kind', 'start_ns', 'end_ns']) {
    assert.equal(event[field], raw[field]);
  }
  assert.equal(event.rank?.rank, raw.rank?.rank);
  assert.equal(event.resources.length, raw.resources.length);
  if (raw.transfer) assert.deepEqual(event.transfer.hops.map(h => [h.start_ns,h.end_ns]),
    raw.transfer.hops.map(h => [h.start_ns,h.end_ns]));
}
const aggregate = traceVisualizationFromReport({visualization: data.visualization});
assert.equal(aggregate.total_events, data.visualization.pagination.total);
const page = tracePageData(data.page, 'batch', aggregate);
assert.equal(page.events.length, 1);
assert.equal(page.pagination.total, data.tasks.length);
assert.equal(page.pagination.has_more, data.page.pagination.has_more);
assert.equal(page.pagination.next_offset, data.page.pagination.next_offset);
assert.throws(() => normalizeTraceEvent({task_id: 'old', timestamp_ns: 2, metadata: {end_ns: 3}}));
const noSynthesis = normalizeTraceEvent({event_id: 'current', start_ns: 0, end_ns: 1,
  rank_id: 7, event_type: 'old', operator: 'old', resource_intervals: [{id: 'old'}],
  metadata: {source_component: 'old', tensor: 'old', rank: 7}});
assert.equal(noSynthesis.rank, null);
assert.equal(noSynthesis.transfer, null);
assert.equal(noSynthesis.tensor, null);
assert.equal(noSynthesis.event_kind, '');
assert.equal(noSynthesis.operator_id, '');
assert.deepEqual(noSynthesis.resources, []);
const hop = normalizeTraceHops([{id: 'old', source: 'old', target: 'old'}])[0];
assert.equal(hop.link_id, '');
assert.equal(hop.start_ns, null);
assert.equal(hop.source_component, '');
"""
    completed = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert completed.returncode == 0, completed.stderr
    for row in online_summary_dict(result)["requests"].values():
        assert "rejection_reason" in row
        assert "reason" not in row
