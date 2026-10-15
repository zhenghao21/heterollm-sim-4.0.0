"""The graph editor and replay controls use their current input contracts."""
import json
from pathlib import Path

from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive
from test_current_report_callers import browser_check
from test_current_view_boundaries import run_browser


def test_current_browser_port_handlers_preview_and_commit_a_real_graph_connection():
    graph = to_primitive(build_reference_scenario().model.graph)
    module = Path(__file__).parents[1] / "src/heterollm_sim/webui/model-graph-core.js"
    browser_check(("modelGraphPortEndpoint", "modelGraphEndpointIdentity",
                   "modelGraphPortInteractionClass", "startModelGraphConnectionPreview",
                   "updateModelGraphConnectionPreview", "commitModelGraphConnectionPreview"),
                  graph, """
const graph = ModelGraph.normalizeModelGraph(payload);
const tensor = graph.tensors.find(item => item.producer_operator_id && item.consumer_operator_ids.length);
const sourceOperator = graph.operators.find(item => item.operator_id === tensor.producer_operator_id);
const targetOperator = graph.operators.find(item => item.operator_id === tensor.consumer_operator_ids[0]);
const sourcePort = sourceOperator.ports.find(port => port.direction === 'output' && port.tensor_id === tensor.tensor_id);
const targetPort = targetOperator.ports.find(port => port.direction === 'input' && port.tensor_id === tensor.tensor_id);
const button = (operator, port) => ({dataset: {modelPortOperator: operator.operator_id,
  modelPortId: port.port_id, modelPortDirection: port.direction}});
const source = modelGraphPortEndpoint(button(sourceOperator, sourcePort));
targetButton = button(targetOperator, targetPort);
const target = modelGraphPortEndpoint(targetButton);
assert.equal(source.operator_id, sourceOperator.operator_id);
assert.equal(source.port_id, sourcePort.port_id);
assert.equal(source.operatorId, undefined);
assert.equal(modelGraphEndpointIdentity({operatorId: source.operator_id, portId: source.port_id}), null);
state.scenario.model.graph = ModelGraph.disconnect(graph, target);
assert.equal(startModelGraphConnectionPreview(source, {x: 1, y: 2}), true);
assert.equal(modelGraphPortInteractionClass(state.scenario.model.graph, source), 'is-connect-source');
assert.equal(modelGraphPortInteractionClass(state.scenario.model.graph, target), 'is-connect-compatible');
updateModelGraphConnectionPreview({pointerId: 1, clientX: 30, clientY: 40});
assert.equal(state.modelGraphEditor.connectPreview.status, 'compatible');
assert.deepEqual(state.modelGraphEditor.connectPreview.target, modelGraphEndpointIdentity(target));
assert.deepEqual(state.modelGraphEditor.connectPreview.pointer, {x: 30, y: 40});
assert.equal(commitModelGraphConnectionPreview(target), true);
assert.equal(ModelGraph.validateModelGraph(state.scenario.model.graph).valid, true);
const connectedTarget = state.scenario.model.graph.operators.find(op => op.operator_id === target.operator_id)
  .ports.find(port => port.port_id === target.port_id);
assert.equal(connectedTarget.tensor_id, tensor.tensor_id);
assert.equal(state.modelGraphEditor.connectPreview, null);
assert.equal(semanticChanges, 1);
const oldSource = {operatorId: source.operator_id, portId: source.port_id};
assert.equal(ModelGraph.createConnectionPreview(graph, oldSource).active, false);
assert.equal(ModelGraph.validateConnection(graph, source, {operatorId: target.operator_id, portId: target.port_id}).ok, false);
assert.equal(ModelGraph.connectionPreviewCompatibility(graph, source, target).ok, false);
const preview = ModelGraph.createConnectionPreview(graph, source);
assert.equal(ModelGraph.updateConnectionPreview(graph, preview,
  {pointer: {x: 1, y: 2}, target}).target, null);
""", f"const ModelGraph = require({json.dumps(str(module))});\n" + """
const state = {scenario: {model: {}}, modelGraphEditor: {}};
let targetButton, semanticChanges = 0;
const ensureScenarioModelGraph = () => state.scenario.model.graph;
const modelGraphSnapshot = () => structuredClone(state.scenario.model.graph);
const modelGraphPortAtPoint = () => targetButton;
const modelGraphPointerWorldPoint = event => ({x: event.clientX, y: event.clientY});
const refreshModelGraphConnectionPreviewPath = () => {};
const renderModelGraphDiagnostics = () => {};
const clearModelGraphConnectionPreviewWatchdog = () => {};
const applyModelGraphSemanticChange = () => { semanticChanges += 1; };
""")


def test_replay_animation_and_semantic_filters_use_current_fields():
    run_browser("""
const points = [{x: 0, y: 0}, {x: 100, y: 0}, {x: 100, y: 100}];
assert.deepEqual(TraceView.pointAtProgress(points, 0.75), {x: 100, y: 50});
assert.deepEqual(TraceView.pointAtProgress(points, {ratio: 0.75}), {x: 0, y: 0});
const current = {periodMs: 1000, phase: 0.25};
assert.equal(TraceView.particleLoopProgress(500, current), 0.75);
assert.equal(TraceView.particleLoopProgress(1500, current), 0.75);
assert.equal(TraceView.particleLoopProgress(1700000000500, current), 0.75);
assert.equal(TraceView.particleLoopProgress(600, {durationMs: 1000, offset: 0.25}), 0.5);
const events = [
  {event_id: 'compute', category: 'compute', start_ns: 0, end_ns: 10},
  {event_id: 'memory', category: 'memory', start_ns: 20, end_ns: 30},
];
assert.deepEqual(TraceView.filterSemanticTraceEvents(events, {query: 'compute', temporal: 'current'}, 5), [events[0]]);
assert.deepEqual(TraceView.filterSemanticTraceEvents(events, {query: 'memory', temporal: 'future'}, 5), [events[1]]);
assert.equal(TraceView.filterSemanticTraceEvents(events, {keyword: 'missing', status: 'past'}, 5).length, 2);
""")


def test_graph_edits_and_inspector_preserve_current_nested_view_state():
    graph = to_primitive(build_reference_scenario().model.graph)
    module = Path(__file__).parents[1] / "src/heterollm_sim/webui/model-graph-core.js"
    browser_check(("modelGraphUi", "updateModelGraphInspector"), graph, """
let graph = ModelGraph.addNode(payload, {operator_id: 'note', op_kind: 'annotation', ports: [], parameters: {}},
  {position: {x: 10, y: 20}});
assert.deepEqual(graph.attributes.ui.detail.positions.note, {x: 10, y: 20});
graph = ModelGraph.addNode(graph, {operator_id: 'keep', op_kind: 'annotation', ports: [], parameters: {}});
const groupId = graph.operators.find(op => op.op_kind === 'layer_group').operator_id;
const projection = ModelGraph.buildOverviewProjection(graph, {collapsed_groups: [groupId]});
const layout = ModelGraph.overviewResponsiveLayout(projection, 1200);
graph.attributes.ui.overview = {positions: layout.positions, collapsed_groups: [groupId]};
const display = (graph, id) => ModelGraph.buildOverviewProjection(graph,
  {collapsed_groups: graph.attributes.ui.overview.collapsed_groups}).nodes.find(node =>
    (node.group_ids || []).includes(id) || node.operator_ids.includes(id)).display_id;
const noteDisplay = display(graph, 'note'), keepDisplay = display(graph, 'keep');
const notePosition = structuredClone(layout.positions[noteDisplay]);
const keepPosition = structuredClone(layout.positions[keepDisplay]);
graph.attributes.ui.detail.focus_operator_id = 'note';
graph.attributes.ui.positions = {note: {x: 999, y: 999}};
graph = ModelGraph.normalizeModelGraph(graph);
assert.equal(graph.attributes.ui.positions, undefined);
assert.deepEqual(graph.attributes.ui.detail.positions.note, {x: 10, y: 20});
const copied = ModelGraph.duplicateNode(graph, 'note');
const duplicate = copied.operators.find(op => !graph.operators.some(original => original.operator_id === op.operator_id));
assert.deepEqual(copied.attributes.ui.detail.positions[duplicate.operator_id], {x: 58, y: 68});
assert.deepEqual(copied.attributes.ui.overview.positions[display(copied, duplicate.operator_id)],
  {x: notePosition.x + 48, y: notePosition.y + 48});
assert.deepEqual(copied.attributes.ui.overview.positions[noteDisplay], notePosition);
state.scenario.model.graph = copied;
const note = copied.operators.find(op => op.operator_id === 'note');
updateModelGraphInspector(note, {dataset: {modelOperatorId: ''}, value: 'renamed'});
assert.equal(semanticChanges, 1);
assert.equal(state.modelGraphEditor.selectedOperatorId, 'renamed');
const renamed = state.scenario.model.graph;
const ui = modelGraphUi(renamed);
assert.equal(ui.detail.positions.note, undefined);
assert.deepEqual(ui.detail.positions.renamed, {x: 10, y: 20});
assert.equal(ui.detail.focus_operator_id, 'renamed');
const renamedDisplay = display(renamed, 'renamed');
assert.equal(ui.overview.positions[noteDisplay], undefined);
assert.deepEqual(ui.overview.positions[renamedDisplay], notePosition);
assert.deepEqual(ui.overview.positions[keepDisplay], keepPosition);
assert.deepEqual(ui.overview.collapsed_groups, [groupId]);
const saved = ModelGraph.withUiProjection(renamed, ModelGraph.uiProjection(renamed));
assert.deepEqual(ModelGraph.uiProjection(JSON.parse(JSON.stringify(saved))), ui);
const removed = ModelGraph.deleteNode(saved, 'renamed');
assert.equal(removed.attributes.ui.detail.positions.renamed, undefined);
assert.equal(removed.attributes.ui.detail.focus_operator_id, '');
assert.equal(removed.attributes.ui.overview.positions[renamedDisplay], undefined);
assert.deepEqual(removed.attributes.ui.overview.positions[keepDisplay], keepPosition);
assert.equal(ModelGraph.validateModelGraph(removed).valid, true);
const oldGroupDisplay = display(removed, groupId);
const groupPosition = structuredClone(removed.attributes.ui.overview.positions[oldGroupDisplay]);
const renamedGroup = ModelGraph.updateNode(removed, groupId, {operator_id: 'renamed-group'});
const newGroupDisplay = display(renamedGroup, 'renamed-group');
assert.notEqual(newGroupDisplay, oldGroupDisplay);
assert.equal(renamedGroup.attributes.ui.overview.positions[oldGroupDisplay], undefined);
assert.deepEqual(renamedGroup.attributes.ui.overview.positions[newGroupDisplay], groupPosition);
""", f"const ModelGraph = require({json.dumps(str(module))});\n" + """
const state = {scenario: {model: {}}, modelGraphEditor: {}};
let semanticChanges = 0;
const ensureScenarioModelGraph = () => state.scenario.model.graph;
const modelGraphSnapshot = () => ({graph: structuredClone(state.scenario.model.graph)});
const applyModelGraphSemanticChange = () => {
  assert.equal(ModelGraph.validateModelGraph(state.scenario.model.graph).valid, true);
  assert.equal(ModelGraph.graphToLayerSpecs(state.scenario.model.graph).ok, true);
  semanticChanges += 1;
};
const chineseMessage = error => { throw error; };
""")
