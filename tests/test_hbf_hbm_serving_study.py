"""Full backend checks: labels must correspond to real placement and traffic."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
import hbf_hbm_serving_study as study
from heterollm_sim.memory_allocator import AllocationError, PhysicalAddressAllocator


class ServingStudyTests(unittest.TestCase):
    @staticmethod
    def _raise_capacity_failure(*args, **kwargs):
        allocator = PhysicalAddressAllocator(1024, alignment_bytes=64)
        allocator.allocate('resident', 800)
        # 224 bytes remain, but the next aligned base is 832: only 192 fit.
        allocator.allocate('new_buffer', 200)

    def test_physical_capacity_failure_preserves_real_alignment_and_fragmentation(self):
        aligned = PhysicalAddressAllocator(1024, alignment_bytes=64)
        aligned.allocate('resident', 800)
        fragmented = PhysicalAddressAllocator(1024, alignment_bytes=64)
        fragmented.allocate('first', 256, address=0)
        fragmented.allocate('second', 256, address=512)
        fragmented.allocate('shared_prefix', 128, alias_of='first')
        partitioned = PhysicalAddressAllocator(1024, alignment_bytes=64,
                                               workspace_capacity_bytes=256)
        partitioned.allocate('resident', 640)
        partitioned.allocate('workspace', 128, generation=1)
        for allocator, requested, occupied, free, largest, bound, arena_end in (
            (aligned, 200, 800, 224, 192, 1032, 1024),
            (fragmented, 300, 512, 512, 256, 1068, 1024),
            (partitioned, 160, 640, 128, 128, 800, 768),
        ):
            with self.subTest(occupied=occupied, requested=requested):
                before = allocator.allocations()
                try:
                    allocator.allocate('does_not_fit', requested)
                except AllocationError as error:
                    failure = study.physical_capacity_failure(error)
                else:
                    self.fail('the physical allocator must actually reject the extent')
                self.assertEqual(failure['classification'], 'physical_allocator_no_contiguous_extent')
                self.assertEqual(failure['requested_allocation_bytes'], requested)
                self.assertEqual(failure['capacity_bytes'], 1024)
                self.assertEqual(failure['occupied_bytes_at_failure'], occupied)
                self.assertEqual(failure['free_bytes_at_failure'], free)
                self.assertEqual(failure['largest_free_extent_bytes'], largest)
                self.assertEqual(failure['same_layout_minimum_capacity_bytes'], bound)
                self.assertEqual(failure['compacted_occupancy_plus_request_bytes'], occupied + requested)
                self.assertEqual(failure['arena_end_bytes'], arena_end)
                self.assertEqual(allocator.allocations(), before)

    def test_capacity_failure_preserves_failed_extents_after_release(self):
        allocator = PhysicalAddressAllocator(1024, alignment_bytes=64)
        allocator.allocate('resident', 800)
        try:
            allocator.allocate('does_not_fit', 200)
        except AllocationError as error:
            allocator.release('resident')
            self.assertEqual(allocator.allocations(), ())
            failure = study.physical_capacity_failure(error)
        else:
            self.fail('the physical allocator must actually reject the extent')
        self.assertEqual(failure['occupied_bytes_at_failure'], 800)
        self.assertEqual(failure['free_bytes_at_failure'], 224)
        self.assertEqual(failure['largest_free_extent_bytes'], 192)
        self.assertEqual(failure['same_layout_minimum_capacity_bytes'], 1032)

    def test_capacity_classification_does_not_swallow_other_allocation_errors(self):
        import hbf_hbm_experiments as experiments
        message = 'unable to allocate 200 bytes in capacity 1024'
        errors = [ValueError(message), AllocationError('prefix ' + message),
                  AllocationError(message + ' unrelated suffix')]
        allocation = PhysicalAddressAllocator(1024).allocate('bounded', 64)
        try:
            allocation.address(32, 64)
        except AllocationError as error:
            errors.append(error)
        else:
            self.fail('out-of-bounds access must raise a real AllocationError')
        for error in errors:
            with self.subTest(error=str(error)):
                self.assertIsNone(study.physical_capacity_failure(error))
                with mock.patch.object(experiments, 'serving_row', side_effect=error):
                    with self.assertRaises(type(error)) as captured:
                        study.run_case('A_hbm_only', row_id='capacity/unrelated', prompt=8, output=2)
                self.assertIs(captured.exception, error)

    def test_capacity_rejection_is_not_reported_as_executed_throughput(self):
        import hbf_hbm_experiments as experiments
        # Two reference stacks provide 1638.4 GB/s, enough for the fixed
        # 1600 GB/s direction budget. The mocked allocator below, not an
        # invalid one-stack interface, must trigger the capacity rejection.
        parameters = dict(prompt=8, output=2, hbm_capacity_bytes=128 * study.GIB)
        with mock.patch.object(experiments, 'serving_row', side_effect=self._raise_capacity_failure):
            row = study.run_case('A_hbm_only', row_id='capacity/rejected', **parameters)
        self.assertEqual(row['id'], 'capacity/rejected')
        self.assertEqual(row['mode'], 'physical_capacity_rejected')
        self.assertEqual(row['parameters'], parameters)
        self.assertFalse(row['executed_serving'])
        self.assertTrue(row['attempted_serving'])
        self.assertIsNone(row['node_tps'])
        self.assertIsNone(row['goodput_tps'])
        self.assertEqual(row['fixed_inventory']['hbm0']['capacity_bytes'], 128 * study.GIB)
        self.assertGreater(row['model_structure']['weight_bytes'], 0)
        self.assertEqual(row['capacity_failure']['largest_free_extent_bytes'], 192)
        self.assertNotIn('completed_requests', row)
        self.assertNotIn('core_execution_audit', row)

    def test_capacity_rejection_cache_requires_matching_parameters_version_and_evidence(self):
        import hbf_hbm_experiments as experiments
        cases = (('A_hbm_only', 'capacity/cache', dict(prompt=8, output=2)),)
        with tempfile.TemporaryDirectory(prefix='hbf-capacity-cache-') as directory, \
             mock.patch.object(experiments, 'serving_row', side_effect=self._raise_capacity_failure):
            initial = study.study_cases(cases, point_cache_dir=directory, progress=False)
            self.assertEqual(initial[0]['point_result_version'], study.POINT_RESULT_VERSION)
            self.assertEqual(initial[0]['mode'], 'physical_capacity_rejected')
            with mock.patch.object(study, 'run_case', side_effect=AssertionError('matching rejection must resume')):
                resumed = study.study_cases(cases, point_cache_dir=directory, resume=True, progress=False)
            self.assertEqual(resumed, initial)
            path = Path(directory) / 'capacity__cache.json'
            for changed in ('parameters', 'version', 'evidence', 'rejection_kind'):
                with self.subTest(changed=changed):
                    stale = json.loads(json.dumps(initial[0]))
                    if changed == 'parameters':
                        stale['parameters']['prompt'] = 16
                    elif changed == 'version':
                        stale['point_result_version'] = study.POINT_RESULT_VERSION - 1
                    elif changed == 'evidence':
                        stale['capacity_failure'] = None
                    else:
                        stale['mode'] = 'configuration_rejected'
                    path.write_text(json.dumps(stale), encoding='utf-8')
                    with mock.patch.object(study, 'run_case', wraps=study.run_case) as execute:
                        refreshed = study.study_cases(cases, point_cache_dir=directory, resume=True, progress=False)
                    self.assertEqual(execute.call_count, 1)
                    self.assertEqual(refreshed, initial)

    def test_named_architecture_and_context_extension(self):
        eight=study.build_study_scenario('A_hbm_only',model_id='llama3_1-8b',prompt=2048)
        seventy=study.build_study_scenario('B_all_kv_hbf',model_id='llama3_1-70b',prompt=1048576)
        self.assertEqual(eight.model.num_layers,32)
        self.assertEqual(seventy.model.num_layers,80)
        self.assertGreater(seventy.model.total_declared_weight_bytes,100*study.GIB)
        self.assertEqual(seventy.placement.kv_policy.dtype,'bf16')
        self.assertEqual(seventy.model.metadata['study_context_extrapolation']['native_max_tokens'],131072)
        self.assertFalse(seventy.model.metadata['study_context_extrapolation']['accuracy_claim'])

    def test_c_uses_real_physical_swap_and_resume_in_full_serving(self):
        row=study.run_case('C_hot_cold',row_id='test/C',prompt=32,output=4,batch=1,
            request_count=2,interval_ns=10000.,priority_preemption=True)
        self.assertEqual(row['completed_requests'],2)
        self.assertGreater(row['kv']['offload_bytes'],0)
        self.assertGreater(row['kv']['peak_bytes_by_component']['hbf0'],0)
        self.assertEqual(row['kv']['offload_bytes'],row['kv']['swap_in_bytes'])
        self.assertEqual(row['kv']['recompute_events'],0)
        directions={(e['source_component'],e['target_component']) for e in row['migration_ledger']['events']}
        self.assertEqual(directions,{('hbm0','hbf0'),('hbf0','hbm0')})
        self.assertTrue(row['placement_verification']['resolved_kv_on_requested_tier'])
        self.assertTrue(row['placement_verification']['all_resolved_model_ops_on_gpu'])

    def test_d_executes_hbf_weight_reads_with_hbm_active_kv(self):
        row=study.run_case('D_hbf_weights',row_id='test/D',prompt=32,output=4)
        self.assertEqual(row['completed_requests'],1)
        storage=row['resources']['storage_traffic']
        self.assertGreater(storage['logical_read_bytes'],0)
        self.assertEqual(storage['logical_write_bytes'],0)
        self.assertGreater(row['kv']['logical_decode_append_bytes'],0)
        self.assertEqual(set(row['kv']['peak_bytes_by_component']),{'hbm0'})
        self.assertTrue(row['placement_verification']['all_resolved_weights_on_requested_tier'])
        self.assertTrue(row['placement_verification']['all_resolved_model_ops_on_gpu'])

    def test_nvme_baseline_is_storage_not_relabelled_hbf(self):
        scenario=study.build_study_scenario('F_nvme',prompt=32,output=4)
        self.assertEqual(scenario.hardware.get_component('hbf0').kind,'ssd')
        row=study.run_case('F_nvme',row_id='test/F',prompt=32,output=4,batch=1,
            request_count=2,interval_ns=10000.,priority_preemption=True)
        self.assertEqual(row['completed_requests'],2)
        self.assertGreater(row['kv']['swap_in_bytes'],0)
        self.assertGreater(row['node_tps'],0)

    def test_extended_presets_keep_real_dense_and_moe_geometry(self):
        dense=study.build_study_scenario('A_hbm_only',model_id='llama3_1-405b',
            prompt=2048,hbm_capacity_bytes=1024*study.GIB)
        moe=study.build_study_scenario('D_hbf_weights',model_id='qwen3-30b-a3b',prompt=2048)
        self.assertEqual(dense.model.num_layers,126)
        self.assertGreater(dense.model.total_declared_weight_bytes,512*study.GIB)
        self.assertEqual(dense.hardware.get_component('hbm0').capacity_bytes,1024*study.GIB)
        self.assertEqual(moe.model.num_layers,48)
        layers=study.model_graph_execution_layers(moe.model.graph)
        self.assertTrue(all(layer.is_moe for layer in layers))
        self.assertTrue(all(layer.num_experts>layer.experts_per_token>1 for layer in layers))

    def test_speculative_full_serving_observes_acceptance_and_rejection(self):
        report=study.run_serving_study(sections=('speculative',),progress=False)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(len(report['rows']),8)
        for row in report['rows']:
            self.assertEqual(row['completed_requests'],4)
            self.assertEqual(row['model_structure']['mtp_descriptor_count'],2)
            events=row['speculative']
            probability=row['parameters']['speculative_acceptance']
            if probability is None:
                self.assertEqual(events['drafted_tokens'],0)
            else:
                self.assertGreater(events['drafted_tokens'],0)
                self.assertTrue(events['rounds'])
                self.assertEqual(events['drafted_tokens'],events['accepted_draft_tokens']+events['rejected_draft_tokens'])
                self.assertGreater(events['rejected_draft_tokens'],0)
                self.assertEqual(events['accepted_draft_tokens']==0,probability==0.)

    def test_parallel_points_equal_serial_and_resume_only_matching_cache(self):
        cases=(("A_hbm_only","parallel/A",dict(prompt=32,output=4)),
               ("B_all_kv_hbf","parallel/B",dict(prompt=32,output=4)))
        def measured(rows):
            # JSON normalizes tuple/list representation after cache loading;
            # every reported value except real execution wall time is compared.
            return json.loads(json.dumps([{k:v for k,v in row.items() if k!='wall_seconds'}
                                          for row in rows]))
        serial=study.study_cases(cases,workers=1,progress=False)
        with tempfile.TemporaryDirectory(prefix='hbf-study-parallel-') as directory:
            parallel=study.study_cases(cases,workers=2,point_cache_dir=directory,progress=False)
            self.assertEqual([r['id'] for r in parallel],['parallel/A','parallel/B'])
            self.assertEqual(measured(serial),measured(parallel))
            with mock.patch.object(study,'run_case',side_effect=AssertionError('matching cache must resume')), \
                 mock.patch.object(study,'ProcessPoolExecutor',side_effect=AssertionError('no workers needed')):
                resumed=study.study_cases(cases,workers=2,point_cache_dir=directory,resume=True,progress=False)
            self.assertEqual(measured(parallel),measured(resumed))
            first=Path(directory)/'parallel__A.json'
            second=Path(directory)/'parallel__B.json'
            cached=json.loads(first.read_text(encoding='utf-8'))
            cached['parameters']['prompt']=99
            first.write_text(json.dumps(cached),encoding='utf-8')
            cached=json.loads(second.read_text(encoding='utf-8'))
            cached['point_result_version']=study.POINT_RESULT_VERSION-1
            second.write_text(json.dumps(cached),encoding='utf-8')
            with mock.patch.object(study,'run_case',wraps=study.run_case) as execute:
                refreshed=study.study_cases(cases,workers=1,point_cache_dir=directory,resume=True,progress=False)
            self.assertEqual(execute.call_count,2)
            self.assertEqual(measured(serial),measured(refreshed))

    def test_parallel_failure_retains_successful_points_and_raises(self):
        cases=(("A_hbm_only","failure/A",dict(prompt=32,output=4)),
               ("invalid_architecture","failure/bad",dict(prompt=32,output=4)),
               ("B_all_kv_hbf","failure/B",dict(prompt=32,output=4)))
        with tempfile.TemporaryDirectory(prefix='hbf-study-failure-') as directory:
            with self.assertRaises(study.StudyCasesError) as captured:
                study.study_cases(cases,workers=2,point_cache_dir=directory,progress=False)
            self.assertIn('failure/bad',str(captured.exception))
            self.assertEqual([r['id'] for r in captured.exception.completed_rows],['failure/A','failure/B'])
            self.assertFalse((Path(directory)/'failure__bad.json').exists())
            for name in ('A','B'):
                cached=json.loads((Path(directory)/f'failure__{name}.json').read_text(encoding='utf-8'))
                self.assertTrue(cached['executed_serving'])
                self.assertEqual(cached['completed_requests'],1)


if __name__=='__main__':
    unittest.main()
