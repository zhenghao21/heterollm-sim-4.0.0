"""Behavior boundaries for declared native K/V cache writeback; no engine launch."""
import copy
from dataclasses import replace
import unittest

from heterollm_sim import planner
from tests.test_mmq_planner import KV_CONTRACT, scenario


def _native_tasks(schedule, stage, *, phase="gpu_elementwise"):
    return [
        task for task in schedule.tasks
        if task.metadata.get("phase") == phase
        and task.metadata.get("native_kv_work", {}).get("stage") == stage
    ]


def _ancestors(schedule, task):
    by_id = {item.task_id: item for item in schedule.tasks}
    found, pending = set(), list(task.dependencies)
    while pending:
        task_id = pending.pop()
        if task_id in found:
            continue
        found.add(task_id)
        pending.extend(by_id[task_id].dependencies)
    return found


def _remote_cache(case):
    placement = replace(
        case.placement,
        tensor_to_component={**case.placement.tensor_to_component, "kv_cache": "hostmem0"},
        kv_policy=replace(case.placement.kv_policy, cache_component="hostmem0"),
    )
    return replace(case, placement=placement)


def _mismatched_k_projection(case):
    """Retain source-valid quantized descriptors while changing only physical K width."""
    group = next(operator for operator in case.model.graph.operators if operator.op_kind == "layer_group")
    parameters = copy.deepcopy(group.parameters)
    projections = parameters["overrides"]["full0"]["metadata"]["weight_projection_descriptors"]["projections"]
    for segment in (projections["attention.qkv"]["segments"][1], projections["attention.k"]["segments"][0]):
        bytes_per_row = segment["physical_bytes"] // segment["n"]
        segment["n"] = 129
        segment["physical_bytes"] = 129 * bytes_per_row
    operators = tuple(replace(operator, parameters=parameters) if operator is group else operator
                      for operator in case.model.graph.operators)
    return replace(case, model=replace(case.model, graph=replace(case.model.graph, operators=operators)))


class NativeKVWritebackTests(unittest.TestCase):
    def test_normal_and_neox_f16_fuse_k_but_keep_q_rope_v_set_and_cache_write(self):
        for rope_type in ("normal", "neox"):
            with self.subTest(rope_type=rope_type):
                case = scenario(kv_contract={**KV_CONTRACT, "rope_type": rope_type})
                schedule = planner.compile_scenario(case)
                layer, rows = planner._execution_layers(case)[0], case.workload.prompt_tokens
                cache_bytes = planner._kv_tensor_bytes(case, layer, 1, rows)
                kv_width = planner._physical_kv_width_for_rank(layer, 1)
                q_rope = _native_tasks(schedule, "q_rope")
                k_rope = _native_tasks(schedule, "k_rope")
                v_set = _native_tasks(schedule, "v_set_rows")
                self.assertEqual(len(q_rope), 1)
                self.assertEqual(q_rope[0].metadata["event_kind"], "rope")
                self.assertEqual(len(k_rope), 1)
                self.assertEqual(k_rope[0].metadata["event_kind"], "kv_native_k_rope")
                self.assertEqual(k_rope[0].metadata["persistent_output_bytes"], cache_bytes)
                self.assertFalse(_native_tasks(schedule, "k_set_rows"))
                self.assertEqual(len(v_set), 1)
                self.assertEqual(v_set[0].metadata["persistent_output_bytes"], cache_bytes)
                self.assertEqual(v_set[0].metadata["cost_model"]["read_bytes"], 4 * rows * kv_width + 8 * rows)
                self.assertEqual(v_set[0].metadata["cost_model"]["write_bytes"], cache_bytes)
                for task in (*q_rope, *k_rope, *v_set):
                    self.assertEqual(task.metadata["modeled_memory_write_bytes"], task.metadata["cost_model"]["write_bytes"])
                self.assertTrue(any(demand.bytes_moved > 0 for demand in k_rope[0].demands))
                self.assertTrue(any(demand.bytes_moved > 0 for demand in v_set[0].demands))
                for projection in ("attention.q", "attention.k"):
                    gemm = next(task for task in schedule.tasks if task.metadata.get("phase") == "gpu_gemm"
                                and task.metadata.get("projection_id") == projection)
                    work = gemm.metadata["mmq_source_work"]
                    self.assertEqual(gemm.metadata["cost_model"]["output_bytes"], 4 * work["m"] * work["n"])
                    self.assertTrue(gemm.metadata["native_qkv_f32_output"])
                    self.assertEqual(gemm.metadata["persistent_output_bytes"], 0)

                append = [task for task in schedule.tasks if task.metadata.get("event_kind") == "kv_append"]
                self.assertEqual(len(append), 1)
                self.assertEqual(append[0].metadata["native_kv_work"]["stage"], "append_complete")
                self.assertEqual(append[0].metadata["bytes"], 2 * cache_bytes)
                self.assertFalse(any(task.name.endswith("kv.append.local") for task in schedule.tasks))
                attention = next(task for task in schedule.tasks
                                 if task.metadata.get("event_kind") == "fused_attention"
                                 and task.metadata.get("phase") == "prefill")
                self.assertIn(append[0].task_id, _ancestors(schedule, attention))

    def test_imrope_q4_and_q8_keep_independent_k_v_set_rows(self):
        cases = (
            scenario(kv_contract={**KV_CONTRACT, "rope_type": "imrope"}),
            scenario(cache_format="Q4_0"),
            scenario(cache_format="Q8_0"),
        )
        for case in cases:
            with self.subTest(cache=case.placement.kv_policy.dtype,
                              rope=case.model.metadata["llama_cpp_native_kv_writeback"]["rope_type"]):
                schedule = planner.compile_scenario(case)
                layer, rows = planner._execution_layers(case)[0], case.workload.prompt_tokens
                cache_bytes = planner._kv_tensor_bytes(case, layer, 1, rows)
                kv_width = planner._physical_kv_width_for_rank(layer, 1)
                self.assertEqual(len(_native_tasks(schedule, "q_rope")), 1)
                self.assertEqual(len(_native_tasks(schedule, "k_rope")), 1)
                self.assertEqual(_native_tasks(schedule, "k_rope")[0].metadata["event_kind"], "rope")
                self.assertEqual(len(_native_tasks(schedule, "k_set_rows")), 1)
                self.assertEqual(len(_native_tasks(schedule, "v_set_rows")), 1)
                for stage in ("k_set_rows", "v_set_rows"):
                    task = _native_tasks(schedule, stage)[0]
                    self.assertEqual(task.metadata["cost_model"]["read_bytes"], 4 * rows * kv_width + 8 * rows)
                    self.assertEqual(task.metadata["cost_model"]["write_bytes"], cache_bytes)
                    self.assertEqual(task.metadata["persistent_output_bytes"], cache_bytes)
                    self.assertEqual(task.metadata["modeled_memory_write_bytes"], cache_bytes)
                    self.assertTrue(any(demand.bytes_moved > 0 for demand in task.demands))
                for stage in ("q_rope", "k_rope"):
                    task = _native_tasks(schedule, stage)[0]
                    self.assertEqual(task.metadata["modeled_memory_write_bytes"], task.metadata["cost_model"]["write_bytes"])
                if case.placement.kv_policy.dtype in {"int4", "int8"}:
                    append = next(task for task in schedule.tasks
                                  if task.metadata.get("event_kind") == "kv_append")
                    logical = planner._logical_kv_bytes_per_token_for_rank(case, layer, 1, 0) * rows
                    self.assertEqual(append.metadata["logical_bytes"], logical)
                    self.assertEqual(append.metadata["physical_bytes"], 2 * cache_bytes)
                    self.assertGreater(append.metadata["physical_bytes"], logical)

    def test_cache_side_branches_do_not_publish_activation_liveness(self):
        case = scenario()
        schedule = planner.compile_scenario(case)
        stages = {"k_rope", "k_set_rows", "v_set_rows", "append_complete"}
        for task in schedule.tasks:
            if task.metadata.get("native_kv_work", {}).get("stage") in stages:
                self.assertIsNone(planner._transient_task_geometry(case, task, row_count=case.workload.prompt_tokens))

    def test_missing_or_false_contract_falls_back_without_native_tasks(self):
        for contract in ({}, {**KV_CONTRACT, "explicit_flash_attention": False}):
            with self.subTest(contract=contract):
                schedule = planner.compile_scenario(scenario(kv_contract=contract))
                self.assertFalse(any(task.metadata.get("event_kind") in {"kv_native_k_rope", "kv_native_set_rows"}
                                     for task in schedule.tasks))
                q = next(task for task in schedule.tasks if task.metadata.get("phase") == "gpu_gemm"
                         and task.metadata.get("projection_id") == "attention.q")
                self.assertEqual(q.metadata["native_kv_work"]["status"], "uncovered")
                self.assertEqual(q.metadata["native_kv_work"]["reason"], "ordinary_cache_source_contract_not_declared")

    def test_physical_qkv_cache_width_disagreement_is_explicitly_uncovered(self):
        schedule = planner.compile_scenario(_mismatched_k_projection(scenario()))
        qkv = [task for task in schedule.tasks if task.metadata.get("phase") == "gpu_gemm"
               and task.metadata.get("projection_id") in {"attention.q", "attention.k", "attention.v"}]
        self.assertEqual(len(qkv), 3)
        self.assertTrue(all(task.metadata["native_kv_work"]["reason"]
                            == "physical_qkv_and_cache_row_widths_disagree" for task in qkv))
        self.assertFalse(any(task.metadata.get("event_kind") in {"kv_native_k_rope", "kv_native_set_rows"}
                             for task in schedule.tasks))

    def test_remote_cache_and_cpu_target_keep_old_paths_without_native_writes(self):
        remote = planner.compile_scenario(_remote_cache(scenario()))
        self.assertFalse(any(task.metadata.get("event_kind") in {"kv_native_k_rope", "kv_native_set_rows"}
                             for task in remote.tasks))
        transfers = [task for task in remote.tasks if task.metadata.get("event_kind") == "kv_append"
                     and task.metadata.get("resource_accounting") == "explicit_remote_transfer"]
        self.assertTrue(transfers)

        case = scenario()
        cpu_case = replace(case, placement=replace(
            case.placement, op_to_component={**case.placement.op_to_component, "attention": "cpu0"}))
        cpu = planner.compile_scenario(cpu_case)
        self.assertFalse(any(task.metadata.get("event_kind") in {"kv_native_k_rope", "kv_native_set_rows"}
                             for task in cpu.tasks))
        self.assertTrue(any(task.metadata.get("event_kind") == "operator_input_transfer"
                            and ".attention_q." in task.name for task in cpu.tasks))


if __name__ == "__main__":
    unittest.main()
