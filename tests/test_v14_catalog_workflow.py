"""Acceptance coverage for a realistic catalog -> JSON -> mapping -> run flow."""

import json
import unittest
from dataclasses import replace

from heterollm_sim.architecture_presets import architecture_preset_detail
from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.model_presets import materialize_model_payload
from heterollm_sim.planner import validate_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.web import scenario_to_payload


class CatalogWorkflowAcceptanceTests(unittest.TestCase):
    """Keep one representative user workflow genuinely executable end to end."""

    def test_h100_qwen_tp2_empty_rank_mapping_import_maps_and_runs(self):
        payload = scenario_to_payload(build_reference_scenario())
        reference_hardware = payload["hardware"]
        hardware = json.loads(
            json.dumps(
                architecture_preset_detail("nvidia-h100-sxm-8-nvswitch")[
                    "hardware"
                ]
            )
        )
        model = materialize_model_payload("qwen3-8b")

        # Architecture presets describe the accelerator package.  A complete
        # V4 scenario also carries the required host CPU and orchestration path.
        cpu = next(
            component
            for component in reference_hardware["components"]
            if component["component_id"] == "cpu0"
        )
        host_memory = next(
            component
            for component in reference_hardware["components"]
            if component["component_id"] == "hostmem0"
        )
        reference_gpu = next(
            component
            for component in reference_hardware["components"]
            if component["component_id"] == "gpu0"
        )
        pcie_port = next(
            port for port in reference_gpu["ports"] if port["port_id"] == "pcie0"
        )
        host_link = next(
            link
            for link in reference_hardware["links"]
            if link["link_id"] == "cpu-gpu-pcie"
        )
        host_memory_link = next(
            link
            for link in reference_hardware["links"]
            if link["link_id"] == "cpu-hostmem-ddr"
        )
        hardware["components"].append(json.loads(json.dumps(cpu)))
        hardware["components"].append(json.loads(json.dumps(host_memory)))
        accelerator0 = next(
            component
            for component in hardware["components"]
            if component["component_id"] == "accelerator0"
        )
        accelerator0["ports"].append(json.loads(json.dumps(pcie_port)))
        host_link = json.loads(json.dumps(host_link))
        host_link["target_component"] = "accelerator0"
        hardware["links"].append(host_link)
        hardware["links"].append(json.loads(json.dumps(host_memory_link)))

        payload["hardware"] = hardware
        payload["model"] = model
        payload["profiles"]["host_orchestration"][
            "gpu_component_id"
        ] = "accelerator0"
        payload["profiles"]["host_orchestration"][
            "submission_resource_id"
        ] = "accelerator0.command_queue"
        payload["placement"]["hardware_name"] = hardware["name"]
        payload["placement"]["model_name"] = model["name"]
        for field, value in (
            ("tp_degree", 2),
            ("pp_degree", 1),
            ("ep_degree", 1),
        ):
            payload["placement"]["parallel"][field] = value
        payload["placement"]["parallel"]["rank_mapping"] = []
        payload["placement"]["parallel"]["layer_to_stage"] = {}
        payload["placement"]["op_to_component"] = {}
        payload["placement"]["tensor_to_component"] = {}
        payload["placement"]["tensor_bytes"] = {}

        memory_id = next(
            component["component_id"]
            for component in hardware["components"]
            if component["kind"] == "hbm"
        )
        payload["placement"]["kv_policy"]["cache_component"] = memory_id
        payload["placement"]["kv_policy"]["offload_component"] = None

        # Exercise the same plain-JSON boundary used by file import.  An empty
        # V4 rank_mapping means "derive the TP ranks".
        scenario = scenario_from_dict(
            json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        )
        self.assertEqual(scenario.schema_version, "4.0.0")
        self.assertEqual(scenario.hardware.schema_version, "4.0.0")
        self.assertEqual(scenario.model.schema_version, "4.0.0")
        self.assertEqual(scenario.placement.schema_version, "4.0.0")
        self.assertEqual(scenario.workload.schema_version, "4.0.0")
        self.assertEqual(scenario.placement.parallel.rank_mapping, ())
        self.assertEqual(scenario.placement.kv_policy.cache_component, memory_id)
        hardware_before = scenario.hardware
        model_before = scenario.model
        workload_before = scenario.workload

        result = plan_runtime_placement(
            scenario,
            PlacementPolicy(mode="heuristic", objective="balanced"),
        )

        self.assertTrue(result.fully_placed, result.unplaced)
        self.assertEqual(result.status, "feasible")
        self.assertEqual(scenario.hardware, hardware_before)
        self.assertEqual(scenario.model, model_before)
        self.assertEqual(scenario.workload, workload_before)

        compute_targets = {
            target.compute_component_id
            for decision in result.decisions
            for target in decision.rank_execution_targets
        }
        storage_targets = {
            shard.storage_component_id
            for decision in result.decisions
            for shard in decision.rank_tensor_shards
        }
        self.assertEqual(len(compute_targets), 2)
        hbm_components = {
            component["component_id"]: component
            for component in hardware["components"]
            if component["kind"] == "hbm"
        }
        storage_by_memory_subsystem = {}
        for component_id in storage_targets:
            component = hbm_components[component_id]
            memory_subsystem_id = component["metadata"]["physical_composition"][
                "memory_subsystem_id"
            ]
            storage_by_memory_subsystem.setdefault(memory_subsystem_id, set()).add(
                component_id
            )
        self.assertEqual(len(storage_by_memory_subsystem), 2)
        self.assertEqual(
            {
                len(component_ids)
                for component_ids in storage_by_memory_subsystem.values()
            },
            {5},
        )
        self.assertTrue(
            all(
                len(decision.rank_execution_targets) == 2
                for decision in result.decisions
                if decision.mapping_key
            )
        )

        mapped = replace(
            scenario,
            placement=result.placement,
            weights_resident=result.weights_resident,
        )
        validation = validate_scenario(mapped)
        self.assertTrue(validation.is_valid, validation.errors)

        report = report_dict(run_scenario(mapped))
        summary = report["summary"]
        self.assertEqual(summary["parallel"]["tp_degree"], 2)
        self.assertEqual(summary["parallel"]["world_size"], 2)
        self.assertEqual(summary["completed_requests"], 1)
        self.assertGreater(summary["makespan_ns"], 0.0)
        self.assertGreater(summary["ttft_ns"]["p50"], 0.0)
        self.assertGreater(summary["tpot_ns"]["p50"], 0.0)
        self.assertGreater(report["category_time_ns"]["communication"], 0.0)
        self.assertTrue(
            all(
                any(
                    key.startswith(component_id + ".") and value > 0.0
                    for key, value in report["resource_utilization"].items()
                )
                for component_id in compute_targets
            )
        )


if __name__ == "__main__":
    unittest.main()
