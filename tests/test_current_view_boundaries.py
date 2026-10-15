"""Current topology producers and browser interaction callers stay in sync."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from heterollm_sim.architecture_presets import architecture_preset_detail, architecture_preset_page


WEBUI = Path(__file__).parents[1] / "src/heterollm_sim/webui"


def run_browser(script, payload=None):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser interaction checks")
    imports = "\n".join(
        f"const {name} = require({json.dumps(str(WEBUI / file))});"
        for name, file in (("Topology", "topology-core.js"), ("TraceView", "trace-view-core.js"),
                           ("ModelGraph", "model-graph-core.js")))
    source = "const assert = require('node:assert/strict');\n" + imports
    source += "\nconst payload = " + json.dumps(payload) + ";\n" + script
    result = subprocess.run([node], input=source, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_saved_architecture_topology_uses_current_groups_and_layout_fields():
    presets = [architecture_preset_detail(item["id"]) for item in
               architecture_preset_page()["items"] if item["loadable"]][:2]
    assert presets
    run_browser("""
for (const item of payload) {
  const ids = item.hardware.components.map(c => c.component_id);
  const stored = item.hardware.metadata.topology_view;
  const view = Topology.normalizeTopologyView(stored, ids);
  assert.deepEqual(view.groups, stored.groups);
  assert.deepEqual(view.layout.positions, stored.layout.positions);
  assert.deepEqual(Topology.normalizeTopologyView(view, ids), view);
}
assert.throws(() => Topology.normalizeTopologyView({positions: {gpu: {x: 9, y: 7}}}, ['gpu']));
for (const retired of [{id: 'old'}, {root_id: 'gpu'}]) {
  assert.throws(() => Topology.normalizeTopologyView({groups: [{members: ['gpu'], ...retired}]}, ['gpu']));
}
""", presets)


def test_trace_arrange_fit_and_drag_preserve_current_layout_contract():
    run_browser("""
const components = [{component_id: 'a', kind: 'gpu'}, {component_id: 'b', kind: 'hbm'}];
const view = {components, links: [], layout: {positions: {a: {x: 10, y: 20}, b: {x: 250, y: 20}},
  node_sizes: {a: {width: 100, height: 50}, b: {width: 90, height: 60}}}};
const arranged = TraceView.arrangeTraceTopologyView(view);
assert.deepEqual(Object.keys(arranged.layout.positions).sort(), ['a', 'b']);
const fit = TraceView.fitTraceViewport(view, {width: 800, height: 500});
const dragged = TraceView.dragTraceNode(fit, 'a', {x: 80, y: 90});
assert.deepEqual(dragged.layout.positions.a, {x: 80, y: 90});
assert.deepEqual(dragged.layout.positions.b, view.layout.positions.b);
assert.deepEqual(dragged.layout.node_sizes, view.layout.node_sizes);
assert.deepEqual(TraceView.normalizeTraceTopologyView(dragged), dragged);
assert.throws(() => TraceView.normalizeTraceTopologyView({trace_topology_view: view}));
assert.equal(TraceView.applyTraceViewAction, undefined);
const activity = TraceView.activeTraceSets([{start_ns: 0, end_ns: 10, rank: {component_id: 'a'},
  transfer: {source_component: 'a', target_component: 'b', link_id: 'wire'}}], 5, ['a', 'b']);
assert.deepEqual([...activity.nodeIds].sort(), ['a', 'b']);
assert.deepEqual([...activity.linkIds], ['wire']);
for (const old of ['activeNodeIds', 'activeLinkIds', 'nodes', 'links']) assert.equal(activity[old], undefined);
assert.ok(TraceView.buildTraceNodeModels(components, view, activity).every(node => node.active));
assert.equal(TraceView.nodePresentation({id: 'old'}).component_id, '');
""")


def test_browser_drag_commit_keeps_moved_and_untouched_node_coordinates():
    source = (WEBUI / "app.js").read_text(encoding="utf-8")
    function = re.search(r"^function endTraceCanvasPointer\([^\n]*\) \{.*?^\}", source, re.M | re.S)
    assert function
    run_browser("""
const element = {style: {transform: 'preview'}};
const state = {tracePlayback: {layoutView: {positions: {a: {x: 10, y: 20}, b: {x: 250, y: 20}}},
  layoutDrag: {pointerId: 1, componentId: 'a', moved: true, previewPosition: {x: 80, y: 90}, element,
    layout: {components: [{component_id: 'a'}, {component_id: 'b'}], links: [],
      sizes: {a: {width: 100, height: 50}, b: {width: 100, height: 50}},
      bounds: {x: 0, y: 0, width: 400, height: 150}}}}};
const dom = {traceTopologyCanvas: {classList: {remove: () => {}}}};
let tracePendingPointer = {};
const cancelTraceInteractionFrame = () => {};
const applyTraceInteractionFrame = () => {};
let rendered = 0;
const renderTraceTopology = () => { rendered += 1; };
""" + function.group() + """
endTraceCanvasPointer({type: 'pointerup', pointerId: 1, clientX: 80, clientY: 90});
assert.deepEqual(state.tracePlayback.layoutView.positions, {a: {x: 80, y: 90}, b: {x: 250, y: 20}});
assert.equal(state.tracePlayback.layoutDrag, null);
assert.equal(state.tracePlayback.topologyLayout, null);
assert.equal(rendered, 1);
""")


def test_current_route_segments_survive_repeated_routing_and_avoid_existing_lines():
    run_browser("""
const a = {id: 'a', x: 0, y: 0, width: 50, height: 60};
const b = {id: 'b', x: 300, y: 0, width: 50, height: 60};
const options = {sourcePoint: {x: 50, y: 30, side: 'right'}, targetPoint: {x: 300, y: 30, side: 'left'},
  sourceSide: 'right', targetSide: 'left', shortCurveDistance: 40};
const existing = Topology.routeSegments([{x: 100, y: 30}, {x: 250, y: 30}],
  {sourcePortKey: 'a:out', targetPortKey: 'b:in', routeId: 'existing'});
const normalized = Topology.planOrthogonalRoutes([], [a, b], {occupiedSegments: existing}).occupiedSegments;
assert.deepEqual(normalized, existing);
assert.deepEqual(Topology.planOrthogonalRoutes([], [a, b], {occupiedSegments: normalized}).occupiedSegments, existing);
const direct = ModelGraph.modelGraphRouteEdge(a, b, [], options);
const separated = ModelGraph.modelGraphRouteEdge(a, b, [], {...options, occupiedSegments: existing});
assert.ok(separated.obstacle_free);
assert.equal(separated.overlaps, 0);
assert.notDeepEqual(separated.points, direct.points);
for (const retired of [{start: {x: 100, y: 30}, end: {x: 250, y: 30}},
  {from: {x: 100, y: 30}, to: {x: 250, y: 30}}, {points: [{x: 100, y: 30}, {x: 250, y: 30}]}]) {
  assert.deepEqual(ModelGraph.modelGraphRouteEdge(a, b, [], {...options, occupiedSegments: [retired]}).points, direct.points);
}
""")
