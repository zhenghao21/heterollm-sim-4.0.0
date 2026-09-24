import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import memory_tier_sweep as sweep
from tools import qwen38_memory_scenario as runner
from heterollm_sim.config import scenario_from_dict


def observation():
    return {
        'model_sha256': 'model', 'workload_sha256': 'workload', 'placement_sha256': 'placement',
        'metrics': {'ttft_ns': 10., 'tpot_ns': 5., 'e2e_ns': 25.},
        'summary': {'completed_requests': 1, 'rejected_requests': 0},
        'expected_requests': 1, 'expected_output_tokens': 4, 'actual_output_tokens': 4,
        'resource_bytes': {'hbm0.controller': 100}, 'resource_service_ns': {'hbm0.controller': 5},
        'complete_resource_accounting': True, 'accounting_errors': [],
        'counted_resource_bytes': 100, 'batch_resource_bytes': 100, 'reported_resource_bytes': 100,
        'component_ids': ['hbm0', 'hbf0'], 'profile_owners': {'hbm0': 'hbm0.controller'},
        'kv_cache': {'logical_bytes_per_token': 64}, 'linear_state': {'bytes_per_request': 128},
        'placement': {'tensor_to_component': {'weight': 'hbm0', 'kv_cache': 'hbm0'},
            'options': {'weight_tensor_targets': {'weight': 'hbm0'}, 'kv_cache_target': 'hbm0'},
            'weight_tensor_details': {'weight': {'residency': 'rank_sharded_storage', 'total_physical_bytes': 100}},
            'rank_weight_shards': {'weight': [{'storage_component_id': 'hbm0', 'physical_bytes': 100}]}}
    }


def test_valid_accounting_and_corrupt_placement_fail_closed():
    value = observation()
    assert all(c['status'] == 'PASS' for c in sweep.validate_observation(value))
    value['counted_resource_bytes'] += 1
    value['placement']['tensor_to_component']['weight'] = 'unknown'
    value['placement']['rank_weight_shards']['weight'][0]['physical_bytes'] = 99
    checks = {c['check']: c['status'] for c in sweep.validate_observation(value)}
    assert checks['resource_byte_conservation'] == 'FAIL'
    assert checks['placement_constraints'] == 'FAIL'
    assert checks['weight_shard_attribution'] == 'FAIL'


def test_zero_traffic_requires_absence_of_service_and_same_placement():
    a, b = observation(), observation()
    assert sweep.zero_traffic_invariance(a, b, 'hbf0')['status'] == 'PASS'
    b['metrics']['ttft_ns'] = 11
    assert sweep.zero_traffic_invariance(a, b, 'hbf0')['status'] == 'FAIL'
    b['placement_sha256'] = 'changed'
    assert sweep.zero_traffic_invariance(a, b, 'hbf0')['status'] == 'NOT_COVERED'
    b = observation(); b['resource_service_ns']['component.hbf0.read'] = 10
    assert sweep.zero_traffic_invariance(a, b, 'hbf0')['status'] == 'NOT_COVERED'
    b = observation(); b['complete_resource_accounting'] = False
    assert sweep.zero_traffic_invariance(a, b, 'hbf0')['status'] == 'NOT_COVERED'


def test_monotonicity_rejects_nan_regression_and_remapping():
    a, b = observation(), observation()
    points = [{'value': 50, 'observation': a}, {'value': 100, 'observation': b}]
    assert sweep.monotonicity(points)['status'] == 'PASS'
    assert not sweep.monotonicity(points)['observed_sensitivity']
    b['metrics']['e2e_ns'] = 24
    assert sweep.monotonicity(points)['status'] == 'FAIL'
    b['metrics']['e2e_ns'] = float('nan')
    assert sweep.monotonicity(points)['status'] == 'FAIL'
    b['workload_sha256'] = 'changed'
    assert sweep.monotonicity(points)['status'] == 'NOT_COVERED'
    assert sweep.monotonicity([])['status'] == 'NOT_COVERED'


def test_remote_flash_requires_actual_bidirectional_swap():
    value = observation()
    value['kv_cache'].update(swap_events=1, swap_in_bytes=4096, swap_out_bytes=4096)
    assert sweep.scenario_coverage('hbf_remote_flash', value)['remote_flash_kv'] == 'NOT_COVERED'
    value['resource_bytes'].update({'component.hbf0.read': 4096, 'component.hbf0.write': 4096})
    assert sweep.scenario_coverage('hbf_remote_flash', value)['remote_flash_kv'] == 'COVERED'


def test_matrix_is_small_and_long_shape_not_cartesian():
    cells = sweep.plan_cells(('baseline', 'hbf_idle'), include_long=True)
    assert len(cells) == 8
    assert sum(c['shape'] == 'long' for c in cells) == 2
    assert len({c['id'] for c in cells}) == len(cells)


def test_output_is_exclusive_and_cannot_touch_r0(tmp_path, monkeypatch):
    allowed = tmp_path / 'new-results'
    monkeypatch.setattr(runner, 'OUTPUT_ROOT', allowed)
    path = allowed / 'result.json'
    runner.write_output(path, {'status': 'UNVALIDATED'})
    with pytest.raises(FileExistsError):
        runner.write_output(path, {'status': 'changed'})
    with pytest.raises(ValueError, match='NEW file'):
        runner.write_output(tmp_path / 'optimization_loop/state.json', {})
    assert json.loads(path.read_text())['status'] == 'UNVALIDATED'


@pytest.fixture(scope='module')
def real_model():
    if not runner.DEFAULT_MODEL.exists():
        pytest.skip('project Qwen3.8 sidecar not installed')
    return runner.load_model()


def test_real_model_sidecar_and_configuration_roundtrip(real_model):
    model, identity = real_model
    assert len(identity['layers']['linear_attention']) == 48
    assert len(identity['layers']['full_attention']) == 16
    assert identity['gguf_payload_rehashed'] is False
    scenario = runner.build_scenario(model)
    payload = runner.scenario_payload(scenario)
    assert runner.scenario_payload(scenario_from_dict(payload)) == payload
    assert scenario.workload.mtp is None
    assert len(scenario.workload.requests) == 1
    assert scenario.workload.requests[0].prompt_tokens == 8


def test_hbf_idle_does_not_change_placement(real_model):
    model, _ = real_model
    baseline = runner.build_scenario(model)
    idle = runner.build_scenario(model, 'hbf_idle')
    assert baseline.placement == idle.placement
    assert baseline.workload == idle.workload
    assert idle.hardware.get_component('hbf0').metadata['access_mode'] == 'remote_flash'


def test_real_short_end_to_end_resource_checks(real_model):
    model, _ = real_model
    value = runner.execute_scenario(runner.build_scenario(model))
    assert all(c['status'] == 'PASS' for c in sweep.validate_observation(value))
    assert value['linear_state']['bytes_per_request'] > 0
    assert value['kv_cache']['logical_bytes_per_token'] > 0
    assert value['validation_status'] == 'UNVALIDATED'


def test_layer_constraints_must_match_lowered_maps():
    value = observation()
    value['placement']['options']['kv_layer_targets'] = {'layer-003': 'hbm0'}
    checks = {c['check']: c['status'] for c in sweep.validate_observation(value)}
    assert checks['placement_constraints'] == 'FAIL'


def test_unexercised_hardware_is_not_a_success():
    for scenario in ('hbf_remote_flash', 'hbf_remote_weights', 'hbf_active_kv', 'dual_dram_cim'):
        coverage = sweep.scenario_coverage(scenario, observation())
        assert sweep.coverage_check(scenario, coverage)['status'] == 'NOT_COVERED'


def test_partial_request_completion_fails():
    value = observation()
    value['expected_requests'] = 2
    checks = {c['check']: c['status'] for c in sweep.validate_observation(value)}
    assert checks['request_completion'] == 'FAIL'
    value['expected_requests'] = 1
    value['actual_output_tokens'] = 3
    checks = {c['check']: c['status'] for c in sweep.validate_observation(value)}
    assert checks['request_completion'] == 'FAIL'


def test_dual_dram_requires_both_vertical_paths_and_cim_array():
    value = observation()
    value['component_ids'] += ['dram0', 'dram1', 'cim0']
    value['profile_owners'].update(dram0='dram0.controller', dram1='dram1.controller')
    value['resource_bytes'].update({'dram0.controller': 10, 'dram1.controller': 10,
        'link.vertical_dram0.soc0->dram0': 10, 'cim0.load': 10})
    value['placement']['op_to_component'] = {'layer.mlp': 'cim0'}
    value['resource_service_ns']['cim0.load'] = 10
    coverage = sweep.scenario_coverage('dual_dram_cim', value)
    assert coverage['vertical_traffic'] == 'NOT_COVERED'
    assert coverage['cim_execution'] == 'NOT_COVERED'
    value['resource_bytes']['link.vertical_dram1.dram1->soc0'] = 10
    value['resource_service_ns']['cim0.array'] = 10
    coverage = sweep.scenario_coverage('dual_dram_cim', value)
    assert sweep.coverage_check('dual_dram_cim', coverage)['status'] == 'PASS'
    coverage = sweep.scenario_coverage('dual_dram_cim_shared', value)
    assert sweep.coverage_check('dual_dram_cim_shared', coverage)['status'] == 'NOT_COVERED'


def test_cli_rejects_existing_output_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUTPUT_ROOT', tmp_path)
    path = tmp_path / 'result.json'
    path.write_text('{}')
    monkeypatch.setattr(runner, 'load_model', lambda *_: pytest.fail('must reject output first'))
    assert runner.main(['--output', str(path)]) == 2


def test_no_fallback_reads_gguf_when_sidecar_is_missing(tmp_path):
    path = tmp_path / 'model.gguf'
    path.write_bytes(b'not a gguf')
    with pytest.raises(ValueError, match='metadata cache'):
        runner.load_model(path)


def test_transfer_ledger_is_explicit_and_not_double_counted():
    demand = {'resource_id': 'component.hbf0.read', 'bytes_moved': 4096, 'service_ns': 100}
    rows, source = runner.batch_demands({'execution_stages': [], 'resource_demands': [demand]})
    assert rows == [demand] and source == 'resource_demands'
    rows, source = runner.batch_demands({'execution_stages': [{'execution_tasks': [
        {'task_id': 't0', 'resource_demands': [demand]}]}], 'resource_demands': [demand]})
    assert rows == [demand] and source == 'execution_stages'
    with pytest.raises(ValueError, match='no complete'):
        runner.batch_demands({'resource_accounted_bytes': 4096, 'resource_busy_ns': {'hbf0': 100}})


def test_dedicated_fp16_cim_is_not_qwen_weight_conversion():
    from heterollm_sim.cost_models import DigitalSramCimProfile, GemmWorkload, estimate_cim_gemm
    profile = DigitalSramCimProfile(array_count=1, p_m=1, p_k=4, p_n=4, frequency_ghz=1,
        arithmetic_mode='fp16_fp32_analytical', float_cycles_per_eval=7,
        float_accumulator_outputs_per_cycle=2,
        float_contract_basis='UNVALIDATED synthetic FP16 tile with FP32 reduction; NOT Qwen GGUF conversion')
    workload = GemmWorkload(m=2, k=8, n=4, activation_bits=16, weight_bits=16,
                            accumulator_bits=32, cim_arithmetic='fp16')
    estimate = estimate_cim_gemm(profile, workload)
    contract = estimate.metadata['arithmetic_contract']
    assert not contract['calibrated'] and not contract['numerical_equivalence_verified']
    assert contract['accumulation_model'] == 'fp32_rounded_partial_sums'
    assert not contract['bit_slice_model_applied']
    assert estimate.metadata['array_cycles'] == 28


def test_real_qwen_has_no_dense_f16_matrix(real_model):
    _, identity = real_model
    assert identity['dense_f16_matrix_count'] == 0


def test_real_flash_pressure_swaps_close_resource_ledger(real_model):
    model, _ = real_model
    scenario = runner.build_scenario(model, 'hbf_remote_flash')
    assert scenario.placement.metadata['linear_state_offload_mode'] == 'pressure'
    value = runner.execute_scenario(scenario)
    assert all(c['status'] == 'PASS' for c in sweep.validate_observation(value))
    coverage = sweep.scenario_coverage('hbf_remote_flash', value)
    assert coverage['remote_flash_kv'] == 'COVERED'
    assert coverage['kv_swap_events'] > 0
    assert coverage['kv_swap_in_bytes'] == coverage['kv_swap_out_bytes'] == 1048576
    assert value['resource_ledger_sources']['resource_demands'] == 4
    assert value['linear_state_traffic_bytes'].get('offload', 0) == 0
    assert value['linear_state']['swap_in_bytes'] > 0
    assert value['resource_bytes']['component.hbf0.read'] > coverage['kv_swap_in_bytes']
    assert value['resource_bytes']['component.hbf0.write'] > coverage['kv_swap_out_bytes']


def test_shared_vertical_logical_resource_ids_resolve_declared_owner():
    value = observation()
    value['component_ids'] += ['dram0', 'dram1']
    value['profile_owners'].update(dram0='dram0.controller', dram1='dram1.controller')
    value['resource_bytes'].update({'dram0.controller': 10, 'dram1.controller': 10,
        'link.vertical_dram0': 10, 'link.vertical_dram1': 20})
    value['physical_resource_owners'] = {'link.vertical_dram0': 'stack0.shared_phy_noc',
        'link.vertical_dram1': 'stack0.shared_phy_noc'}
    coverage = sweep.scenario_coverage('dual_dram_only_shared', value)
    assert coverage['vertical_traffic'] == coverage['shared_fabric'] == 'COVERED'
    assert sum(coverage['shared_fabric_logical_bytes'].values()) == 30
    assert sweep.coverage_check('dual_dram_only_shared', coverage)['status'] == 'PASS'


def test_fixed_mapping_rejects_remap_even_with_same_observation_hash():
    from copy import deepcopy
    expected = sweep.placement_mapping(observation()['placement'])
    for key in ('op_to_component', 'tensor_to_component', 'memory_tiers', 'rank_weight_shards'):
        changed = deepcopy(expected)
        changed[key]['changed'] = 'other_component'
        with pytest.raises(runner.UnsupportedScenario, match='mapping changed'):
            sweep.require_fixed_mapping(expected, changed)
    a, b = observation(), observation()
    b['placement_sha256'] = 'remapped'
    assert sweep.monotonicity([{'value': 50, 'observation': a},
                               {'value': 100, 'observation': b}])['status'] != 'PASS'


def test_fixed_missing_operator_interface_rejects(real_model, monkeypatch):
    from dataclasses import dataclass
    from heterollm_sim import control_plane_planner
    @dataclass
    class WithoutOperatorTargets:
        pass
    monkeypatch.setattr(control_plane_planner, 'PlacementPolicy', WithoutOperatorTargets)
    model, _ = real_model
    with pytest.raises(runner.UnsupportedScenario, match='requires PlacementPolicy.operator_targets'):
        sweep.fixed_reference(runner.build_scenario(model), 'hbm0')


@pytest.mark.parametrize('name,component', [
    ('baseline', 'hbm0'), ('dual_dram_only', 'dram0'), ('dual_dram_only_shared', 'dram0'),
    ('hbf_active_split', 'hbf0')])
def test_real_fixed_placement_solve_matches_at_each_latency(real_model, name, component):
    from dataclasses import fields
    from heterollm_sim.control_plane_planner import PlacementPolicy
    if 'operator_targets' not in {f.name for f in fields(PlacementPolicy)}:
        pytest.skip('pending core operator_targets interface; fixed rejects rather than passing')
    model, _ = real_model
    original = runner.build_scenario(model, name)
    configured, expected = sweep.fixed_reference(original, component)
    assert not configured.placement.op_to_component  # no copied generated mapping/fingerprint
    options = configured.placement.metadata['control_plane']['policy']['options']
    assert options['operator_targets'] == expected['op_to_component']
    assert options['weight_tensor_targets']
    for latency in (50., 100., 200.):
        point = sweep.set_read_latency(configured, component, latency)
        actual = sweep.placement_mapping(sweep.solve_placement(point))
        sweep.require_fixed_mapping(expected, actual)
        assert actual == expected
    if name == 'baseline':
        values = [runner.execute_scenario(sweep.set_read_latency(configured, component, latency))
                  for latency in (50., 200.)]
        for value in values:
            sweep.require_fixed_mapping(expected, sweep.placement_mapping(value['placement']))
            assert 'storage_distribution' in value
        assert sweep.monotonicity([{'value': latency, 'observation': value}
                                  for latency, value in zip((50., 200.), values)])['status'] == 'PASS'


@pytest.mark.parametrize('name', ['dual_dram_cim_converted', 'dual_dram_cim_converted_shared'])
def test_converted_cim_is_opt_in_and_requires_array_execution(name):
    assert name not in sweep.PRIMARY_SCENARIOS
    value = observation()
    coverage = sweep.scenario_coverage(name, value)
    result = sweep.coverage_check(name, coverage)
    assert result['status'] == 'NOT_COVERED'
    assert 'cim_execution' in result['detail']['required']
    required = result['detail']['required']
    coverage = {key: 'COVERED' for key in required}
    coverage['cim_execution'] = 'NOT_COVERED'
    assert sweep.coverage_check(name, coverage)['status'] == 'NOT_COVERED'
    coverage['cim_execution'] = 'COVERED'
    assert sweep.coverage_check(name, coverage)['status'] == 'PASS'
    value['placement']['op_to_component'] = {'test': 'cim0'}
    value['resource_service_ns']['cim0.load'] = 10
    assert sweep.scenario_coverage(name, value)['cim_execution'] == 'NOT_COVERED'
    value['resource_service_ns']['cim0.array'] = 10
    assert sweep.scenario_coverage(name, value)['cim_execution'] == 'COVERED'


def test_unknown_scenario_cannot_pass_empty_requirements():
    assert sweep.coverage_check('dual_dram_new_unregistered', {})['status'] == 'NOT_COVERED'


def test_operator_constraint_mismatch_cannot_pass_validation():
    value = observation()
    value['placement']['options']['operator_targets'] = {'op': 'gpu0'}
    value['placement']['op_to_component'] = {'op': 'cpu0'}
    checks = {c['check']: c['status'] for c in sweep.validate_observation(value)}
    assert checks['placement_constraints'] == 'FAIL'


@pytest.mark.parametrize('remap', [False, True])
def test_fixed_cli_checks_each_solve_without_resigning(tmp_path, monkeypatch, remap):
    from copy import deepcopy
    from types import SimpleNamespace
    model, scenario = object(), SimpleNamespace(workload=SimpleNamespace(requests=(1,)))
    expected = sweep.placement_mapping(observation()['placement'])
    cells = sweep.plan_cells(('baseline',))
    monkeypatch.setattr(sweep, 'check_output_path', lambda p: p)
    monkeypatch.setattr(sweep, 'source_identity', lambda: {'unchanged': True})
    monkeypatch.setattr(sweep, 'load_model', lambda p: (model, {}))
    monkeypatch.setattr(sweep, 'build_scenario', lambda *a, **k: scenario)
    monkeypatch.setattr(sweep, 'plan_cells', lambda *a, **k: cells)
    references, solved = [], []
    def reference(value, component):
        references.append(component)
        return value, expected
    def solve(value):
        solved.append(value)
        actual = deepcopy(expected)
        if remap:
            actual['op_to_component'] = {'op': 'different'}
        return actual
    monkeypatch.setattr(sweep, 'fixed_reference', reference)
    monkeypatch.setattr(sweep, 'set_read_latency', lambda value, component, latency: value)
    monkeypatch.setattr(sweep, 'solve_placement', solve)
    monkeypatch.setattr(sweep, 'scenario_payload', lambda value: {'model': {}, 'schema_version': '4'})
    monkeypatch.setattr('heterollm_sim.config.scenario_from_dict', lambda value: scenario)
    written = []
    monkeypatch.setattr(sweep, 'write_output', lambda path, payload: written.append(payload))
    code = sweep.main(['--scenario', 'baseline', '--placement-mode', 'fixed', '--dry-run',
                       '--output', str(tmp_path / 'test.json')])
    assert references == ['hbm0']
    assert len(solved) == 3
    assert code == (2 if remap else 0)
    assert written[0]['placement_mode'] == 'fixed'
    assert all(row['status'] == ('BLOCKED' if remap else 'PLANNED') for row in written[0]['cells'])


@pytest.mark.parametrize('targets,error', [
    ({'not-a-real-op': 'gpu0'}, 'unknown operator keys'),
    ({'embedding': 'not-a-component'}, 'unknown components')])
def test_operator_targets_reject_unknown_keys_and_components(real_model, targets, error):
    from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
    model, _ = real_model
    scenario = runner.build_scenario(model)
    if error == 'unknown components':
        op = next(iter(plan_runtime_placement(scenario).placement.op_to_component))
        targets = {op: 'not-a-component'}
    with pytest.raises(ValueError, match=error):
        plan_runtime_placement(scenario, PlacementPolicy(operator_targets=targets))


def test_operator_targets_never_fallback_to_different_component(real_model):
    from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
    model, _ = real_model
    scenario = runner.build_scenario(model)
    op = next(iter(plan_runtime_placement(scenario).placement.op_to_component))
    result = plan_runtime_placement(scenario, PlacementPolicy(operator_targets={op: 'hbm0'}))
    assert not result.fully_placed
    assert op not in result.placement.op_to_component


@pytest.mark.parametrize('bad', [[], {'': 'gpu0'}, {'op': ''}, {'op': 1}])
def test_operator_target_input_validation(bad):
    from heterollm_sim.control_plane_planner import PlacementPolicy
    with pytest.raises(ValueError, match='operator_targets'):
        PlacementPolicy(operator_targets=bad)
