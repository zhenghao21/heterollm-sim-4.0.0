"""The online task serialization boundary must retain new work and dependencies."""
import json
import unittest

from heterollm_sim import planner, serving
from heterollm_sim.serde import to_primitive
from tests.test_mmq_planner import scenario


class MMQRuntimeTests(unittest.TestCase):
    def test_serialized_online_stages_keep_mmq_cache_work_and_resource_demands(self):
        for count in (128, 64, 5):
            with self.subTest(rows=count):
                case = scenario(tokens=count)
                cohort = serving.BatchCohort('online-source', 'prefill', 0, (
                    serving.BatchItem('sample', 'prefill', count, count, logit_tokens=0),))
                metadata = planner.estimate_serving_cohort_cost(case, cohort)['metadata']
                self.assertIsNone(metadata['execution_stage_fallback_reason'])
                wire = json.loads(json.dumps(to_primitive(metadata)))
                stages, reason = serving._execution_stages_from_metadata(wire)
                self.assertIsNone(reason)
                specs, origins = serving._OnlineRuntime._stage_task_specs(
                    stages, {s.stage_id: 0 for s in stages}, 'replayed')
                ids = {t.task_id for t in specs}
                for task in specs:
                    self.assertTrue(set(task.dependencies).issubset(ids))
                    original = origins[task.task_id][1]
                    self.assertEqual(task.demands, original.demands)
                    for key in ('mmq_source_work', 'native_kv_work'):
                        self.assertEqual(task.metadata.get(key), original.metadata.get(key))
                mmq = [t for t in specs if t.metadata.get('mmq_source_work', {}).get('status') == 'applied']
                self.assertEqual(bool(mmq), count > 5)
                self.assertTrue(all(t.metadata['mmq_source_work']['m'] == count for t in mmq))
                cache = [t for t in specs if t.metadata.get('native_kv_work', {}).get('stage') == 'v_set_rows']
                self.assertTrue(cache)
                self.assertTrue(any(d.bytes_moved > 0 for t in cache for d in t.demands))
                self.assertTrue(all(t.metadata['native_kv_work']['rows'] == count for t in cache))


if __name__ == '__main__':
    unittest.main()
