import unittest

from heterollm_sim.config import placement_from_dict, workload_from_dict
from heterollm_sim.ir import (
    SCHEMA_VERSION,
    KVCachePolicy,
    MTPPolicy,
    ParallelSpec,
    PlacementSpec,
    RankMappingSpec,
    RequestSpec,
    SchedulerSpec,
    WorkloadSpec,
)
from heterollm_sim.serde import to_primitive


class SchemaV400ConfigTests(unittest.TestCase):
    def test_schema_and_policy_defaults_are_v4(self):
        self.assertEqual(SCHEMA_VERSION, "4.0.0")
        self.assertEqual(ParallelSpec(tp_degree=2, pp_degree=3, ep_degree=4).world_size, 24)
        self.assertEqual(KVCachePolicy().tokens_per_page, 16)
        self.assertFalse(hasattr(KVCachePolicy(), "page_tokens"))
        self.assertEqual(MTPPolicy().candidate_tokens, 4)
        self.assertEqual(MTPPolicy().min_draft_tokens, 0)
        self.assertEqual(MTPPolicy().proposal_length_model, "max")
        self.assertTrue(MTPPolicy(candidate_tokens=1).enabled)
        self.assertEqual(SchedulerSpec().max_num_seqs, 1)
        self.assertFalse(SchedulerSpec().mixed_phase_batching)
        self.assertEqual(
            SchedulerSpec().phase_candidate_order,
            "least_recently_served",
        )

    def test_nested_placement_round_trips_without_flat_mirrors(self):
        raw = {
            "schema_version": "4.0.0",
            "model_name": "m",
            "hardware_name": "h",
            "tensor_bytes": {},
            "parallel": {
                "tp_degree": "2", "pp_degree": 1, "ep_degree": 1,
                "rank_mapping": [
                    {"rank": 0, "component_id": "gpu0", "tp_rank": 0, "pp_rank": 0, "ep_rank": 0},
                    {"rank": 1, "component_id": "gpu1", "tp_rank": 1, "pp_rank": 0, "ep_rank": 0},
                ],
                "layer_to_stage": {"layer0": "0"},
            },
            "kv_policy": {"cache_component": "hbm0", "tokens_per_page": "32", "offload_ratio": "0.5"},
        }
        placement = placement_from_dict(raw)
        primitive = to_primitive(placement)
        self.assertNotIn("tp_degree", primitive)
        self.assertNotIn("kv_cache_component", primitive)
        self.assertEqual(placement_from_dict(primitive), placement)

    def test_manual_v4_placement_fields_are_rejected(self):
        base = {
            "schema_version": "4.0.0",
            "model_name": "m",
            "hardware_name": "h",
            "parallel": {},
            "kv_policy": {},
        }
        for field, value in (
            ("op_to_component", {"op": "gpu0"}),
            ("tensor_to_component", {"weight": "hbm0"}),
            ("tensor_bytes", {"weight": 32}),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "manual placement fields"
            ):
                placement_from_dict({**base, field: value})

    def test_nested_workload_round_trips_without_flat_mirrors(self):
        raw = {
            "name": "online", "schema_version": "4.0.0",
            "requests": [{
                "schema_version": "4.0.0", "request_id": "r0", "arrival_ns": "10.5",
                "prompt_tokens": "8", "output_tokens": 4, "priority": "2", "deadline_ns": "1000",
            }],
            "scheduler": {
                "mode": "continuous",
                "max_num_seqs": "8",
                "max_num_batched_tokens": "4096",
                "max_num_ubatch_tokens": "128",
                "prefill_chunk_tokens": "256",
                "mixed_phase_batching": True,
                "policy": "decode_first",
                "phase_candidate_order": "stable_admission",
            },
            "mtp": {
                "method": "head_based",
                "candidate_tokens": "6",
                "min_draft_tokens": "0",
                "continuation_threshold": "0.75",
                "proposal_length_model": "trace",
                "draft_length_trace": ["0", 2, 6],
                "acceptance_model": "trace",
                "acceptance_trace": ["1", 0.5, 0.0],
                "proposal_cost_scale": "0.2",
            },
        }
        workload = workload_from_dict(raw)
        self.assertTrue(workload.scheduler.mixed_phase_batching)
        self.assertEqual(workload.scheduler.max_num_ubatch_tokens, 128)
        self.assertEqual(
            workload.scheduler.phase_candidate_order,
            "stable_admission",
        )
        primitive = to_primitive(workload)
        self.assertNotIn("max_batch_size", primitive)
        self.assertNotIn("mtp_acceptance_rate", primitive)
        self.assertEqual(workload_from_dict(primitive), workload)

    def test_old_schema_and_alias_fields_are_hard_rejected(self):
        for version in ("0.1", "0.5", "2.0.0", "3.0.0"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "exactly 4.0.0"):
                placement_from_dict({"schema_version": version, "model_name": "m", "hardware_name": "h", "parallel": {}, "kv_policy": {}})
        for field, value in (("tp_degree", 2), ("kv_cache_component", "hbm0")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unknown fields"):
                placement_from_dict({"schema_version": "4.0.0", "model_name": "m", "hardware_name": "h", "parallel": {}, "kv_policy": {}, field: value})
        for field, value in (("max_batch_size", 4), ("mtp_acceptance_rate", 0.5)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unknown fields"):
                workload_from_dict({"schema_version": "4.0.0", "name": "w", "scheduler": {}, field: value})
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            placement_from_dict({"schema_version": "4.0.0", "model_name": "m", "hardware_name": "h", "parallel": {}, "kv_policy": {"page_tokens": 16}})
        with self.assertRaisesRegex(ValueError, "mixed_phase_batching"):
            workload_from_dict(
                {
                    "schema_version": "4.0.0",
                    "name": "w",
                    "scheduler": {"mixed_phase_batching": "true"},
                }
            )

    def test_direct_schema_bearing_constructors_reject_old_versions(self):
        with self.assertRaisesRegex(ValueError, "exactly 4.0.0"):
            PlacementSpec("m", "h", schema_version="0.5")
        with self.assertRaisesRegex(ValueError, "exactly 4.0.0"):
            WorkloadSpec("w", schema_version="0.5")
        with self.assertRaisesRegex(ValueError, "exactly 4.0.0"):
            RequestSpec("r", 0, 1, 1, schema_version="0.5")

    def test_aggregate_mtp_placement_aliases_are_hard_rejected(self):
        with self.assertRaisesRegex(ValueError, "op_to_component.mtp"):
            PlacementSpec("m", "h", op_to_component={"mtp": "gpu0"})
        with self.assertRaisesRegex(ValueError, "tensor_to_component.mtp_weights"):
            PlacementSpec(
                "m",
                "h",
                tensor_to_component={"mtp_weights": "hbm0"},
            )
        with self.assertRaisesRegex(ValueError, "tensor_bytes.mtp_weights"):
            PlacementSpec("m", "h", tensor_bytes={"mtp_weights": 1})

    def test_strict_policy_and_rank_validation_remains(self):
        with self.assertRaises(ValueError):
            KVCachePolicy(tokens_per_page=0)
        with self.assertRaises(ValueError):
            MTPPolicy(acceptance_rate=1.1)
        with self.assertRaises(ValueError):
            MTPPolicy(candidate_tokens=2, min_draft_tokens=3)
        with self.assertRaises(ValueError):
            MTPPolicy(continuation_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "expected_draft_tokens_per_round"):
            MTPPolicy(proposal_length_model="expected_mean")
        with self.assertRaisesRegex(ValueError, "draft_length_trace"):
            MTPPolicy(proposal_length_model="trace")
        with self.assertRaises(ValueError):
            SchedulerSpec(mode="training")
        with self.assertRaisesRegex(ValueError, "mixed_phase_batching"):
            SchedulerSpec(mixed_phase_batching=1)
        with self.assertRaisesRegex(ValueError, "phase_candidate_order"):
            SchedulerSpec(phase_candidate_order="physical_slot")
        with self.assertRaisesRegex(ValueError, "world_size"):
            ParallelSpec(tp_degree=2, rank_mapping=(RankMappingSpec(0, "gpu0", 0, 0, 0),))


if __name__ == "__main__":
    unittest.main()
