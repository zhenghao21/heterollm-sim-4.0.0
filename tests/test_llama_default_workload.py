import unittest

from heterollm_sim.reference import build_llama_default_scenario


class LlamaDefaultWorkloadTests(unittest.TestCase):
    def test_web_default_matches_llama_aligned_workload_contract(self):
        scenario = build_llama_default_scenario()
        workload = scenario.workload
        request = workload.requests[0]

        self.assertEqual(workload.metadata["workload_preset_id"], "llama_cpp_default")
        self.assertEqual((request.prompt_tokens, request.output_tokens), (512, 128))
        self.assertEqual((workload.request_count, workload.prompt_tokens, workload.output_tokens), (1, 512, 128))
        self.assertEqual(workload.scheduler.mode, "continuous")
        self.assertEqual(workload.scheduler.max_num_seqs, 1)
        self.assertEqual(workload.scheduler.max_num_batched_tokens, 512)
        self.assertEqual(workload.scheduler.max_num_ubatch_tokens, 512)
        self.assertEqual(workload.scheduler.prefill_chunk_tokens, 512)
        self.assertFalse(workload.scheduler.preemption_enabled)
        self.assertIsNone(workload.mtp)


if __name__ == "__main__":
    unittest.main()
