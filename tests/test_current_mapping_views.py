"""Current runtime placement and model storage reach browser consumers."""
import pytest

from heterollm_sim.control_plane_planner import RuntimeExecutionTarget, plan_runtime_placement
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import _runtime_placement_payload
from heterollm_sim.serde import to_primitive
from heterollm_sim.web import scenario_to_payload
from test_current_report_callers import browser_check
from test_current_view_boundaries import run_browser


@pytest.fixture(scope="module")
def current_placement():
    scenario = build_reference_scenario()
    placed = plan_runtime_placement(scenario).apply(scenario)
    return {"runtime": _runtime_placement_payload(placed),
            "scenario": scenario_to_payload(scenario),
            "colocated": to_primitive(RuntimeExecutionTarget(1, 1, 0, 0, "gpu0", "cim0"))}


def test_runtime_mapping_rows_use_explicit_rank_targets_and_physical_shard_bytes(current_placement):
    browser_check(("controlPlaneRank", "objectArray", "operatorTargetRows", "tensorShardRows"),
                  current_placement, """
const decision = payload.runtime.control_plane.decision;
const ranks = payload.runtime.parallel.rank_mapping;
const operators = operatorTargetRows(decision, ranks);
const shards = tensorShardRows(decision);
assert.equal(operators.length, Object.values(decision.operator_execution_targets).flat().length);
assert.equal(shards.length, Object.values(decision.rank_weight_shards).flat().length);
for (const row of operators) {
  const raw = decision.operator_execution_targets[row.operator].find(item => item.rank_id === row.rank.rank);
  assert.equal(row.componentId, raw.component_id);
  assert.equal(row.rank.component_id, raw.compute_component_id);
  assert.equal(row.rank.tp_rank, raw.tp_rank);
}
for (const row of shards) {
  const raw = decision.rank_weight_shards[row.tensorId].find(item => item.rank_id === row.rank.rank);
  assert.equal(row.storage, raw.storage_component_id);
  assert.equal(row.compute, raw.compute_component_id);
  assert.equal(row.logicalBytes, raw.logical_bytes);
  assert.equal(row.physicalBytes, raw.physical_bytes);
  assert.equal(row.residency, raw.residency);
}
const colocated = operatorTargetRows({operator_execution_targets: {op: [payload.colocated]}}, ranks)[0];
assert.equal(colocated.rank.rank, 1);
assert.equal(colocated.rank.tp_rank, 1);
assert.equal(colocated.rank.component_id, 'gpu0');
assert.equal(colocated.componentId, 'cim0');
for (const old of ['gpu0', {rank: 0, execution_component_id: 'gpu0'},
  {...payload.colocated, rank_id: undefined}]) {
  assert.throws(() => operatorTargetRows({operator_execution_targets: {op: [old]}}, ranks));
}
const [tensorId, rawShards] = Object.entries(decision.rank_weight_shards)[0];
for (const [current, old] of [['rank_id', 'rank'], ['storage_component_id', 'component_id'],
  ['logical_bytes', 'length_bytes'], ['physical_bytes', 'logical_bytes']]) {
  const shard = {...rawShards[0], [old]: rawShards[0][current]};
  delete shard[current];
  assert.throws(() => tensorShardRows({rank_weight_shards: {[tensorId]: [shard]}}));
}
const cold = {...rawShards[0], physical_bytes: 0};
assert.equal(tensorShardRows({rank_weight_shards: {[tensorId]: [cold]}})[0].physicalBytes, 0);
""", "const uiText = zh => zh;")


def test_only_current_report_owns_materialized_placement(current_placement):
    browser_check(("runtimePlacementForView", "controlPlaneDecision", "controlPlaneEvidence",
                   "effectiveParallelRanks"), current_placement, """
state.scenario = payload.scenario;
state.scenario.placement.metadata.control_plane = {
  decision: {operator_execution_targets: {retired: []}}, evidence: {input_fingerprint: 'retired'},
};
assert.deepEqual(controlPlaneDecision(state.scenario.placement), {});
assert.deepEqual(controlPlaneEvidence(state.scenario.placement), {});
state.report = {runtime_placement: payload.runtime};
assert.deepEqual(controlPlaneDecision(state.scenario.placement), payload.runtime.control_plane.decision);
assert.deepEqual(controlPlaneEvidence(state.scenario.placement), payload.runtime.control_plane.evidence);
const ranks = effectiveParallelRanks();
assert.equal(ranks.length, payload.runtime.parallel.rank_mapping.length);
for (const rank of ranks) assert.deepEqual(rank, payload.runtime.parallel.rank_mapping.find(item => item.rank === rank.rank));
state.mappingStale = true;
assert.deepEqual(controlPlaneDecision(state.scenario.placement), {});
assert.deepEqual(controlPlaneEvidence(state.scenario.placement), {});
""", "const state = {}; const normalizedComponentKind = value => value;")


def test_model_head_storage_has_one_current_nested_contract():
    run_browser("""
const layers = ModelGraph.graphToLayerSpecs(payload).layer_specs;
const summary = ModelGraph.modelGraphAuthoringSummary(payload);
const storage = summary.output_weight_storage;
assert.ok(storage.shape.length);
assert.ok(storage.logical_bytes > 0);
const retiredFields = ['output_weight_contract', 'output_weight_storage_id',
  'output_weight_storage_bytes', 'output_weight_dtype', 'output_weight_shape', 'output_weight_layout'];
for (const name of retiredFields) assert.equal(summary[name], undefined);
const rebuilt = ModelGraph.buildModelGraphFromLayerSpecs(layers, summary);
assert.equal(ModelGraph.validateModelGraph(rebuilt).valid, true);
assert.deepEqual(ModelGraph.modelGraphAuthoringSummary(rebuilt).output_weight_storage, storage);
const rowMajor = ModelGraph.buildModelGraphFromLayerSpecs(layers, {...summary,
  output_weight_storage: {...storage, layout: 'row_major'}});
assert.equal(ModelGraph.validateModelGraph(rowMajor).valid, true);
assert.equal(ModelGraph.modelGraphAuthoringSummary(rowMajor).output_weight_storage.layout, 'row_major');
const head = rowMajor.operators.find(op => op.op_kind === 'lm_head');
assert.equal(head.ports.find(port => port.direction === 'weight').layout, 'row_major');
const tied = ModelGraph.buildModelGraphFromLayerSpecs(layers, {...summary, tie_word_embeddings: true,
  output_weight_bytes: 0, output_weight_storage: {...storage, storage_id: 'embedding_weights'}});
const tiedSummary = ModelGraph.modelGraphAuthoringSummary(tied);
assert.equal(tiedSummary.tie_word_embeddings, true);
assert.equal(tiedSummary.output_weight_bytes, 0);
assert.equal(tiedSummary.output_weight_storage.logical_bytes, storage.logical_bytes);
const old = {...summary, output_weight_contract: {...storage, logical_bytes: 17},
  output_weight_storage_bytes: 17, output_weight_dtype: 'fp64', output_weight_shape: [1, 1]};
delete old.output_weight_storage;
const ignored = ModelGraph.modelGraphAuthoringSummary(ModelGraph.buildModelGraphFromLayerSpecs(layers, old));
assert.notEqual(ignored.output_weight_storage.dtype, 'fp64');
assert.notEqual(ignored.output_weight_storage.logical_bytes, 17);
""", to_primitive(build_reference_scenario().model.graph))
