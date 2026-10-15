"""Current report transfers retain their kind and direction through replay."""
from dataclasses import replace
import json
from pathlib import Path

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.engine import simulate_schedule
from heterollm_sim.reporting import _task_event
from test_backend_execution_semantics import core_schedule
from test_current_report_callers import browser_check


def test_report_transfer_kinds_reach_current_flow_and_particle_consumers():
    tasks = tuple(TaskSpec(kind, "request", kind, TaskCategory.COMMUNICATION,
        demands=(ResourceDemand("link.command", 10., bytes_moved=64),), metadata={
            "source_component": source, "target_component": target, "transfer_kind": kind,
            "route_hops": ({"link_id": "command", "resource_id": "link.command",
                "source_component": source, "target_component": target, "protocol": "PCIe"},),
        }) for kind, source, target in (
            ("instruction", "cpu0", "gpu0"), ("data", "gpu0", "cpu0")))
    schedule = replace(core_schedule(), tasks=tasks, resource_owners={},
                       resource_capacities={"link.command": 1})
    events = [_task_event(task, lookup={}, protocols={}) for task in simulate_schedule(schedule).tasks]
    core = Path(__file__).parents[1] / "src/heterollm_sim/webui/trace-view-core.js"
    browser_check(("finiteTraceNumber", "normalizeTraceRank", "normalizeTraceHops", "normalizeTraceEvent",
        "traceTransferEndpoints", "traceTransferKind", "traceTransferIdentityValues", "traceIntervalActive",
        "traceLinkIdentityMatches", "traceLinkDirection", "traceEventHasExactLinkIdentity",
        "traceLinkFlowDirections", "traceParticleRouteForDirection", "renderTraceParticles"), events, """
assert.equal(TraceView.routePointsForTransfer, undefined);
assert.equal(TraceView.buildTransferParticles, undefined);
const link = {link_id: 'command', source_component: 'cpu0', target_component: 'gpu0', bidirectional: true};
const points = [{x: 0, y: 0}, {x: 100, y: 0}, {x: 100, y: 50}];
const routes = [];
for (const raw of payload) {
  const event = normalizeTraceEvent(raw);
  const flows = traceLinkFlowDirections(link, [event], new Set(), (event.start_ns + event.end_ns) / 2);
  assert.ok(flows.length > 0);
  const flow = flows[0];
  assert.equal(flow.transfer_kind, raw.transfer.kind);
  assert.equal(flow.direction, raw.transfer.kind === 'instruction' ? 1 : -1);
  assert.equal(flow.event_id, raw.event_id);
  assert.equal(flow.transferKind, undefined);
  const route = traceParticleRouteForDirection({points, transfer_kind: flow.transfer_kind}, flow.direction);
  assert.equal(route.transfer_kind, raw.transfer.kind);
  assert.equal(route.transferKind, undefined);
  assert.deepEqual(route.points, flow.direction < 0 ? points.toReversed() : points);
  routes.push(route);
}
renderTraceParticles(routes);
assert.match(dom.traceParticleLayer.innerHTML, /trace-flow-particle is-instruction/);
assert.match(dom.traceParticleLayer.innerHTML, /trace-flow-particle is-data/);
assert.match(dom.traceParticleLayer.innerHTML, /data-trace-direction="reverse"/);
assert.match(dom.traceParticleLayer.innerHTML, /data-trace-transfer-kind="instruction"/);
assert.equal(traceTransferKind({kind: 'instruction'}), 'instruction');
assert.equal(traceTransferKind({}, 'instruction'), 'instruction');
for (const retired of [{transferKind: 'instruction'}, {transfer_kind: 'instruction'},
  {transfer: {kind: 'instruction'}}, {kind: 'instructions'}]) {
  assert.equal(traceTransferKind(retired), 'data');
}
assert.deepEqual(traceTransferEndpoints({source: 'old', target: 'old'}), {source: '', target: ''});
assert.deepEqual(traceLinkFlowDirections(link, normalizeTraceEvent(payload[0]), new Set(), 5), []);
""", f"const TraceView = require({json.dumps(str(core))});\n" + """
const state = {tracePlayback: {timeNs: 0}};
const topologyDisplayLinkId = link => link.link_id;
let traceParticleRoutes = [], traceParticleFrame = null, traceParticleMode = 'static';
const TRACE_PARTICLES_PER_ROUTE = 3;
const dom = {traceParticleLayer: {innerHTML: ''}};
const traceLayoutView = () => ({zoom: 1, offsetX: 0, offsetY: 0});
const traceParticleAnimationCapability = () => ({mode: 'static'});
const tracePointAlongRoute = (points, progress) => TraceView.pointAtProgress(points, progress);
""")
