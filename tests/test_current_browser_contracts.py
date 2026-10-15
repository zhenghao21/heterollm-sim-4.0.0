"""Current report and catalog data reach their actual browser consumers."""
from dataclasses import replace
import json

from heterollm_sim.architecture_presets import architecture_preset_detail, architecture_preset_page
from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.engine import simulate_schedule
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.protocol_presets import protocol_preset_detail
from heterollm_sim.reporting import _runtime_placement_payload, _task_event
from heterollm_sim.serde import to_primitive
from heterollm_sim.serving import _execution_layers
from test_backend_execution_semantics import core_schedule
from test_current_mapping_views import current_placement
from test_current_report_callers import browser_check
from test_current_view_boundaries import WEBUI


def test_memory_tiers_and_ranks_only_use_the_current_report(current_placement):
    scenario = build_reference_scenario()
    layer_map = {layer.layer_id: "hbm" + str(index % 2)
                 for index, layer in enumerate(_execution_layers(scenario))
                 if not layer.is_linear_attention}
    targets = {layer + ".kv_cache": owner for layer, owner in layer_map.items()}
    scenario = replace(scenario, workload=replace(scenario.workload, mtp=None),
        placement=replace(scenario.placement,
            kv_policy=replace(scenario.placement.kv_policy, layout_mode="legacy_single",
                              offload_component=None, offload_ratio=0, preemption_mode="recompute"),
            tensor_to_component={**scenario.placement.tensor_to_component, **targets},
            metadata={**scenario.placement.metadata,
                "memory_tiers": {"kv_layer_components": layer_map},
                "control_plane": {"decision": {"generated_tensor_ids": list(targets)}}}))
    browser_check(("runtimePlacementForView", "controlPlaneMemoryTierMaps", "effectiveParallelRanks"),
        {**current_placement, "tier_runtime": _runtime_placement_payload(scenario)}, """
state.scenario = payload.scenario;
state.scenario.placement.metadata.memory_tiers = {kv_layer_components: {old: 'old-memory'}};
assert.deepEqual(controlPlaneMemoryTierMaps(), {});
assert.deepEqual(effectiveParallelRanks(), []);
state.report = {runtime_placement: payload.runtime};
assert.deepEqual(effectiveParallelRanks(), payload.runtime.parallel.rank_mapping);
state.report.runtime_placement = payload.tier_runtime;
assert.ok(Object.keys(controlPlaneMemoryTierMaps().kv_layer_components).length);
assert.deepEqual(controlPlaneMemoryTierMaps(), payload.tier_runtime.memory_tiers);
delete state.report.runtime_placement.memory_tiers;
state.report.runtime_placement.metadata = {memory_tiers: {old: 'metadata'}};
state.report.runtime_placement.control_plane.memory_tiers = {old: 'control-plane'};
assert.deepEqual(controlPlaneMemoryTierMaps(), {});
for (const flag of ['reportStale', 'mappingStale']) {
  state[flag] = true;
  assert.deepEqual(controlPlaneMemoryTierMaps(), {});
  assert.deepEqual(effectiveParallelRanks(), []);
  state[flag] = false;
}
""", "const state = {};")


def test_mapping_filters_and_tables_use_the_current_execution_target(current_placement):
    browser_check(("controlPlaneRank", "objectArray", "operatorTargetRows", "rankCoordinateLabel",
        "filterEffectiveMappingGroups", "effectiveMappingGroupMarkup", "syncEffectiveMappingFilters"),
        current_placement, """
const rows = operatorTargetRows({operator_execution_targets: {op: [payload.colocated]}}, []);
const groups = [{label: 'op', rows}];
const view = {query: '', rank: '', component: 'cim0'};
assert.equal(filterEffectiveMappingGroups(groups, 'operator', view).length, 1);
assert.deepEqual(filterEffectiveMappingGroups(groups, 'operator', {...view, component: 'gpu0'}), []);
assert.match(effectiveMappingGroupMarkup(groups[0], 'operator'), /cim0/);
syncEffectiveMappingFilters(rows, []);
assert.match(dom.effectiveMappingComponentFilter.innerHTML, /value="cim0"/);
assert.doesNotMatch(dom.effectiveMappingComponentFilter.innerHTML, /gpu0/);
const retired = [{label: 'op', rows: [{...rows[0], componentId: undefined}]}];
assert.deepEqual(filterEffectiveMappingGroups(retired, 'operator', {...view, component: 'gpu0'}), []);
assert.doesNotMatch(effectiveMappingGroupMarkup(retired[0], 'operator'), /gpu0/);
""", """
const uiText = zh => zh, escapeHtml = value => String(value), formatNumber = value => String(value);
const state = {effectiveMappingView: {query: '', rank: '', component: ''}};
const dom = {effectiveMappingSearchInput: {}, effectiveMappingRankFilter: {}, effectiveMappingComponentFilter: {}};
""")


def test_semantic_filters_keep_event_category_distinct_from_operation_kind():
    task = TaskSpec("command", "request", "command", TaskCategory.COMMUNICATION,
        demands=(ResourceDemand("link.command", 10., bytes_moved=64),),
        metadata={"event_kind": "weight_transfer", "phase": "transfer"})
    schedule = replace(core_schedule(), tasks=(task,), resource_owners={},
                       resource_capacities={"link.command": 1})
    event = _task_event(simulate_schedule(schedule).tasks[0], lookup={}, protocols={})
    browser_check(("syncTraceSemanticFilters",), event, """
state.tracePlayback.filteredEvents = [payload];
syncTraceSemanticFilters();
assert.deepEqual(dom.traceEventCategoryFilter.values.map(item => item.value), ['communication']);
assert.deepEqual(TraceView.filterSemanticTraceEvents([payload], {category: 'communication'}, 5), [payload]);
assert.deepEqual(TraceView.filterSemanticTraceEvents([payload], {category: 'weight_transfer'}, 5), []);
const retired = {...payload}; delete retired.category;
state.tracePlayback.filteredEvents = [retired];
syncTraceSemanticFilters();
assert.deepEqual(dom.traceEventCategoryFilter.values, []);
assert.deepEqual(TraceView.filterSemanticTraceEvents([retired], {category: 'weight_transfer'}, 5), []);
""", f"const TraceView = require({json.dumps(str(WEBUI / 'trace-view-core.js'))});\n" + """
const state = {tracePlayback: {semanticCategory: '', semanticPhase: ''}};
const dom = {traceEventCategoryFilter: {}, traceEventPhaseFilter: {}};
const traceFriendlyTerm = value => value;
const syncTraceSelect = (target, values) => { target.values = values; };
""")


def test_architecture_load_uses_the_hardware_owned_topology():
    row = next(row for row in architecture_preset_page()["items"] if row["loadable"])
    browser_check(("architecturePresetDetailItem", "architecturePresetIsLoadable", "componentPresetId",
        "objectArray", "requireScenarioSchemaV4", "normalizeArchitecturePresetHardware",
        "collisionSafeArchitectureTopologyView", "clearParallelLayerToStage",
        "resetPlacementForArchitecturePreset", "applyArchitecturePresetDetail"),
        architecture_preset_detail(row["id"]), """
const retired = {groups: [], layout: {positions: {retired: {x: 1, y: 2}}}};
const detail = architecturePresetDetailItem({...payload, topology_view: retired});
assert.equal(detail.topology_view, undefined);
detail.topology_view = retired;
const result = applyArchitecturePresetDetail(detail);
assert.deepEqual(result.hardware, payload.hardware);
assert.deepEqual(state.topologyView.groups, payload.hardware.metadata.topology_view.groups);
assert.equal(state.topologyView.layout.positions.retired, undefined);
assert.equal(Object.keys(state.topologyView.layout.positions).length, payload.hardware.components.length);
assert.equal(state.scenario.placement.hardware_name, payload.hardware.name);
assert.deepEqual(state.scenario.placement.parallel.rank_mapping, []);
""", f"const Topology = require({json.dumps(str(WEBUI / 'topology-core.js'))});\n" + """
const AUTHORING_SCHEMA_VERSION = '4.0.0';
const state = {settings: {fontScale: 1}, scenario: {profiles: {}, placement: {parallel: {}, metadata: {}}}};
const topologyHistorySnapshot = () => ({}), resetArchitectureDependentProfiles = () => ({});
const topologyLayoutMetrics = () => ({nodeW: 130, nodeH: 66, nodeGap: 30});
const layoutScaleForFont = () => 1;
const saveTopologyView = () => {}, commitTopologyHistory = () => {}, markScenarioChanged = () => {};
const requestAnimationFrame = callback => callback(), renderTopology = () => {}, fitTopologyViewport = () => {};
const restoreTopologyHistorySnapshot = () => {};
""")


def test_gddr_link_creation_uses_current_port_protocol_and_version():
    from test_current_topology_inputs import PRESETS

    payload = [{"memory": to_primitive(get_component_preset(component).component),
                "defaults": protocol_preset_detail(protocol)["simulation_defaults"],
                "generation": generation} for component, protocol, generation in PRESETS]
    browser_check(("createProtocolLink", "componentKindClass", "normalizedComponentKind",
                   "isDedicatedHbm", "gddrGeneration"), payload, """
for (const item of payload) {
  defaults = {...item.defaults.link};
  const gpu = {component_id: 'gpu', kind: 'gpu', metadata: {}, ports: [
    {port_id: 'controller', ...item.defaults.source_port}]};
  const memory = {...structuredClone(item.memory), component_id: 'memory', ports: []};
  state.scenario = {hardware: {components: [gpu, memory], links: []}};
  createProtocolLink('gpu', 'memory', 'GDDR');
  const link = state.scenario.hardware.links[0];
  assert.equal(link.protocol, 'GDDR');
  assert.equal(link.version, item.generation);
  assert.equal(gpu.ports.at(-1).version, item.generation);
  assert.equal(memory.ports.at(-1).protocol, 'GDDR');
  for (const retired of [{protocol: item.generation}, {protocol: 'GDDR', version: 'old-' + item.generation}]) {
    gpu.ports = [{port_id: 'old', ...retired}];
    state.scenario.hardware.links = [];
    assert.throws(() => createProtocolLink('gpu', 'memory', 'GDDR'), /未声明支持/);
    assert.deepEqual(state.scenario.hardware.links, []);
  }
}
""", """
const AUTHORING_SCHEMA_VERSION = '4.0.0';
const state = {};
let defaults;
const currentProtocolConnectionDefaults = () => defaults;
const isActiveMemoryComponent = value => (typeof value === 'object' ? value.kind : value) === 'gddr';
const topologyHistorySnapshot = () => ({}), componentSharedBandwidthGbps = () => 5120;
const costProfileKeyForComponentKind = () => 'gddr', boundCostProfile = () => ({});
const uniquePortId = component => 'new-' + component.component_id, uniqueLinkId = () => 'new-link';
const refreshColocatedRankMapping = () => {}, commitTopologyHistory = () => {};
const markScenarioChanged = () => {}, toast = () => {};
""")
