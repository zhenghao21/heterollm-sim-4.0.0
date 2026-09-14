import unittest
from dataclasses import replace
from types import SimpleNamespace

from heterollm_sim.planner import (
    _mtp_expected_acceptance,
    _serving_cohort_cache_key,
    compile_scenario,
    compile_serving_cohort_schedule,
)
from heterollm_sim.reference import build_reference_scenario
from tests.model_helpers import execution_layers


class PlannerV3MicroarchitectureTests(unittest.TestCase):
    def setUp(self):
        self.scenario = build_reference_scenario()
        self.layers = execution_layers(self.scenario.model)
        self.schedule = compile_scenario(self.scenario)

    def _task(self, name):
        return next(task for task in self.schedule.tasks if task.name == name)

    @staticmethod
    def _demand(task, resource_id):
        return next(
            demand for demand in task.demands
            if demand.resource_id == resource_id
        )

    def test_qkv_rope_fusion_counts_only_q_and_k_at_three_ops_per_element(self):
        layer = self.layers[0]
        request = self.scenario.workload.requests[0]
        head_dim = layer.effective_attention_head_dim
        rope_elements = request.prompt_tokens * (
            layer.attention_heads + layer.effective_kv_heads
        ) * head_dim
        task = self._task("prefill.dense0.rank000.qkv.gpu_gemm")
        model = task.metadata["cost_model"]

        self.assertEqual(model["epilogue_name"], "rope")
        self.assertEqual(model["epilogue_operations"], 3 * rope_elements)
        self.assertEqual(model["epilogue_transcendental_operations"], 0)
        self.assertEqual(
            self._demand(task, "gpu0.scalar").work_units,
            3 * rope_elements,
        )
        self.assertGreater(
            self._demand(task, "gpu0.tensor_core").work_units,
            0,
        )

    def test_rmsnorm_and_fused_residual_norm_have_exact_scalar_counts(self):
        layer = self.layers[0]
        request = self.scenario.workload.requests[0]
        elements = request.prompt_tokens * layer.hidden_size
        rows = request.prompt_tokens

        reduce_task = self._task(
            "prefill.dense0.rank000.input_norm_reduce.gpu_reduction"
        )
        apply_task = self._task(
            "prefill.dense0.rank000.input_norm_apply.gpu_elementwise"
        )
        fused_task = self._task(
            "prefill.dense0.rank000.fused_residual_norm.gpu_elementwise"
        )

        self.assertEqual(
            self._demand(reduce_task, "gpu0.scalar").work_units,
            2 * elements - rows,
        )
        self.assertEqual(
            self._demand(apply_task, "gpu0.scalar").work_units,
            2 * elements + rows,
        )
        self.assertEqual(
            self._demand(apply_task, "gpu0.sfu").work_units,
            rows,
        )
        self.assertEqual(
            self._demand(fused_task, "gpu0.scalar").work_units,
            5 * elements,
        )
        self.assertEqual(
            self._demand(fused_task, "gpu0.sfu").work_units,
            rows,
        )

    def test_router_softmax_counts_two_scalar_ops_and_one_exp_per_score(self):
        layer = self.layers[1]
        request = self.scenario.workload.requests[0]
        score_elements = request.prompt_tokens * layer.num_experts
        task = self._task(
            "prefill.moe1.rank000.router_softmax_normalize.gpu_elementwise"
        )

        self.assertEqual(
            self._demand(task, "gpu0.scalar").work_units,
            2 * score_elements,
        )
        self.assertEqual(
            self._demand(task, "gpu0.sfu").work_units,
            score_elements,
        )

    def test_each_host_cohort_has_one_logical_pack_dma_and_submit(self):
        tasks = tuple(
            task for task in self.schedule.tasks
            if task.name.startswith("prefill.host_orchestration")
            or task.name.startswith("prefill.invocation_frontend")
        )
        by_kind = {}
        for task in tasks:
            by_kind.setdefault(task.metadata.get("event_kind"), []).append(task)

        self.assertEqual(len(by_kind["host_cohort_prepare"]), 1)
        prepare = by_kind["host_cohort_prepare"][0]
        self.assertEqual(prepare.metadata["cohort_setup_scope"], "request_local")
        self.assertEqual(prepare.metadata["batch_fixed_setup_ns"],
                         self.scenario.host_orchestration_profile.batch_fixed_ns)
        self.assertEqual(prepare.metadata["request_parse_total_ns"],
                         self.scenario.host_orchestration_profile.request_parse_ns)
        self.assertEqual(
            [task.metadata["phase"] for task in by_kind["host_cohort_pack"]],
            ["cpu_dispatch", "cpu_memory"],
        )
        self.assertEqual(len(by_kind["host_cohort_h2d"]), 1)
        self.assertTrue(by_kind["host_cohort_h2d"][0].metadata["single_dma_transaction"])
        self.assertEqual(len(by_kind["host_cohort_submit"]), 1)
        submit = by_kind["host_cohort_submit"][0]
        self.assertEqual(submit.metadata["submission_count"], 1)
        self.assertEqual(submit.metadata["transfer_kind"], "instruction")
        self.assertEqual(submit.metadata["source_component"], "cpu0")
        self.assertEqual(submit.metadata["target_component"], "gpu0")

        pack_model = by_kind["host_cohort_pack"][1].metadata["cost_model"]
        self.assertEqual(
            pack_model["instruction_schedule"]["model"],
            "cpu_ooo_instruction_schedule_v3",
        )
        self.assertEqual(pack_model["cache"]["cache_model"], "v3_working_set_reuse")

    def test_enabled_fusion_reduces_hbm_traffic_and_explicit_kernels(self):
        disabled = replace(
            self.scenario,
            fusion_policy=replace(
                self.scenario.fusion_policy,
                qkv_rope=False,
                flash_attention=False,
                gemm_epilogue_activation=False,
                residual_norm=False,
            ),
        )
        disabled_schedule = compile_scenario(disabled)

        def dense0_hbm_bytes(schedule):
            return sum(
                demand.bytes_moved
                for task in schedule.tasks
                if task.name.startswith("prefill.dense0.rank000")
                for demand in task.demands
                if demand.resource_id == "hbm0.hbm_fabric"
            )

        enabled_names = {task.name for task in self.schedule.tasks}
        disabled_names = {task.name for task in disabled_schedule.tasks}
        self.assertLess(
            dense0_hbm_bytes(self.schedule),
            dense0_hbm_bytes(disabled_schedule),
        )
        self.assertNotIn("prefill.dense0.rank000.rope.gpu_elementwise", enabled_names)
        self.assertIn("prefill.dense0.rank000.rope.gpu_elementwise", disabled_names)
        self.assertIn(
            "prefill.dense0.rank000.attention_flash.gpu_fused_attention",
            enabled_names,
        )
        self.assertNotIn(
            "prefill.dense0.rank000.attention_flash.gpu_fused_attention",
            disabled_names,
        )

    def test_flash_attention_uses_gqa_kv_width_and_policy_dtype_for_storage(self):
        cohort = SimpleNamespace(
            cohort_id="kv-flash",
            kind="decode",
            items=(
                SimpleNamespace(
                    request_id="request",
                    phase="decode",
                    token_count=2,
                    context_tokens=5,
                    kv_append_tokens=2,
                    kv_materialized_tokens=2,
                ),
            ),
        )
        tasks = compile_serving_cohort_schedule(self.scenario, cohort).tasks
        fused = next(
            task
            for task in tasks
            if task.name.endswith(
                "dense0.rank000.attention_flash.gpu_fused_attention"
            )
        )
        qkv = next(
            task
            for task in tasks
            if task.name.endswith("dense0.rank000.qkv.gpu_gemm")
        )
        kv_read = next(
            task
            for task in tasks
            if task.metadata.get("event_kind") == "kv_read"
            and task.metadata.get("layer_id") == "dense0"
        )
        kv_append = next(
            task
            for task in tasks
            if task.metadata.get("event_kind") == "kv_append"
            and task.metadata.get("layer_id") == "dense0"
        )

        # dense0 has query width 512 but only 4 KV heads x 64 dimensions.
        # Two queries each read five persisted tokens: 10 exact accesses.
        model = fused.metadata["cost_model"]
        self.assertEqual(model["query_hidden_size"], 512)
        self.assertEqual(model["kv_hidden_size"], 4 * 64)
        self.assertEqual(model["kv_input_bits"], 8)
        self.assertEqual(model["kv_read_tokens"], 10)
        self.assertEqual(model["kv_read_bytes"], 2 * 10 * 4 * 64)
        self.assertEqual(model["read_bytes"], 2 * 512 + 2 * 10 * 4 * 64)
        self.assertEqual(
            model["tensor_operations"],
            4 * 2 * 7 * 512,
        )

        # QKV materializes query bytes plus the exact K/V append bytes.  The
        # explicit local events remain zero-demand bookkeeping.
        self.assertEqual(qkv.metadata["cost_model"]["output_bytes"], 2048)
        self.assertEqual(qkv.metadata["modeled_memory_write_bytes"], 2048)
        self.assertEqual(qkv.metadata["modeled_kv_write_bytes"], 1024)
        self.assertEqual(qkv.metadata["kv_materialized_bytes"], 1024)
        self.assertEqual(kv_read.metadata["bytes"], 5120)
        self.assertEqual(kv_append.metadata["bytes"], 1024)
        self.assertFalse(kv_read.demands)
        self.assertFalse(kv_append.demands)

    def test_flash_hbm_traffic_scales_with_kv_heads_and_dtype_not_compute(self):
        cohort = SimpleNamespace(
            cohort_id="kv-scaling",
            kind="decode",
            items=(
                SimpleNamespace(
                    request_id="request",
                    phase="decode",
                    token_count=1,
                    context_tokens=128,
                    kv_append_tokens=1,
                    kv_materialized_tokens=1,
                ),
            ),
        )

        def row(kv_heads, dtype):
            operators = tuple(
                replace(
                    operator,
                    parameters={**operator.parameters, "kv_heads": kv_heads},
                )
                if operator.op_kind == "attention"
                else operator
                for operator in self.scenario.model.graph.operators
            )
            scenario = replace(
                self.scenario,
                model=replace(
                    self.scenario.model,
                    graph=replace(
                        self.scenario.model.graph,
                        operators=operators,
                    ),
                ),
                placement=replace(
                    self.scenario.placement,
                    kv_policy=replace(
                        self.scenario.placement.kv_policy,
                        dtype=dtype,
                    ),
                ),
            )
            tasks = compile_serving_cohort_schedule(scenario, cohort).tasks
            fused = next(
                task
                for task in tasks
                if task.name.endswith(
                    "dense0.rank000.attention_flash.gpu_fused_attention"
                )
            )
            model = fused.metadata["cost_model"]
            hbm_bytes = sum(
                demand.bytes_moved
                for demand in fused.demands
                if demand.resource_id == "hbm0.hbm_fabric"
            )
            return model, hbm_bytes

        by_heads = [row(kv_heads, "fp16") for kv_heads in (8, 4, 1)]
        by_dtype = [row(4, dtype) for dtype in ("int8", "fp16", "fp32")]

        expected_ops = by_heads[0][0]["tensor_operations"]
        self.assertTrue(
            all(model["tensor_operations"] == expected_ops for model, _ in by_heads)
        )
        self.assertEqual(
            [model["kv_read_bytes"] for model, _ in by_heads],
            [262144, 131072, 32768],
        )
        self.assertGreater(by_heads[0][1], by_heads[1][1])
        self.assertGreater(by_heads[1][1], by_heads[2][1])

        self.assertTrue(
            all(model["tensor_operations"] == expected_ops for model, _ in by_dtype)
        )
        self.assertEqual(
            [model["kv_read_bytes"] for model, _ in by_dtype],
            [65536, 131072, 262144],
        )
        self.assertLess(by_dtype[0][1], by_dtype[1][1])
        self.assertLess(by_dtype[1][1], by_dtype[2][1])

    def test_remote_kv_stages_through_rank_memory_with_endpoint_directions(self):
        placement = replace(
            self.scenario.placement,
            tensor_to_component={
                **self.scenario.placement.tensor_to_component,
                "kv_cache": "hbm1",
            },
            kv_policy=replace(
                self.scenario.placement.kv_policy,
                cache_component="hbm1",
                offload_component=None,
            ),
        )
        components = tuple(
            replace(
                component,
                read_bandwidth_gbps=1_000.0,
                write_bandwidth_gbps=1_000.0,
            )
            if component.component_id in {"hbm0", "hbm1"}
            else component
            for component in self.scenario.hardware.components
        )
        scenario = replace(
            self.scenario,
            placement=placement,
            hardware=replace(self.scenario.hardware, components=components),
        )
        cohort = SimpleNamespace(
            cohort_id="remote-kv",
            kind="decode",
            items=(
                SimpleNamespace(
                    request_id="request",
                    phase="decode",
                    token_count=1,
                    context_tokens=8,
                    kv_append_tokens=1,
                    kv_materialized_tokens=1,
                ),
            ),
        )

        tasks = compile_serving_cohort_schedule(scenario, cohort).tasks
        dense_read = [
            task
            for task in tasks
            if task.metadata.get("event_kind") == "kv_read"
            and task.metadata.get("layer_id") == "dense0"
        ]
        dense_append = [
            task
            for task in tasks
            if task.metadata.get("event_kind") == "kv_append"
            and task.metadata.get("layer_id") == "dense0"
        ]

        self.assertTrue(dense_read)
        self.assertTrue(dense_append)
        self.assertEqual(
            {(task.metadata["source_component"], task.metadata["target_component"])
             for task in dense_read},
            {("hbm1", "hbm0")},
        )
        self.assertEqual(
            {(task.metadata["source_component"], task.metadata["target_component"])
             for task in dense_append},
            {("hbm0", "hbm1")},
        )

        def direction_for(tasks, phase_kind):
            return {
                task.metadata.get("memory_direction")
                for task in tasks
                if task.metadata.get("transfer_phase_event_kind") == phase_kind
            }

        self.assertEqual(direction_for(dense_read, "memory_read"), {"read"})
        self.assertEqual(direction_for(dense_read, "memory_write"), {"write"})
        self.assertEqual(direction_for(dense_append, "memory_read"), {"read"})
        self.assertEqual(direction_for(dense_append, "memory_write"), {"write"})

    def test_nonfused_qk_pv_keep_query_compute_but_read_unique_kv_storage(self):
        cohort = SimpleNamespace(
            cohort_id="kv-split",
            kind="decode",
            items=(
                SimpleNamespace(
                    request_id="request",
                    phase="decode",
                    token_count=2,
                    context_tokens=5,
                    kv_append_tokens=2,
                    kv_materialized_tokens=2,
                ),
            ),
        )

        rows = {}
        for dtype in ("int8", "fp16"):
            scenario = replace(
                self.scenario,
                fusion_policy=replace(
                    self.scenario.fusion_policy,
                    flash_attention=False,
                ),
                placement=replace(
                    self.scenario.placement,
                    kv_policy=replace(
                        self.scenario.placement.kv_policy,
                        dtype=dtype,
                    ),
                ),
            )
            tasks = compile_serving_cohort_schedule(scenario, cohort).tasks
            qk = next(
                task
                for task in tasks
                if task.name.endswith("dense0.rank000.attention_qk.gpu_gemm")
            )
            pv = next(
                task
                for task in tasks
                if task.name.endswith("dense0.rank000.attention_pv.gpu_gemm")
            )
            qkv = next(
                task
                for task in tasks
                if task.name.endswith("dense0.rank000.qkv.gpu_gemm")
            )
            rows[dtype] = (qk, pv, qkv)

        for dtype, bytes_per_element in (("int8", 1), ("fp16", 2)):
            qk, pv, qkv = rows[dtype]
            expected_one_tensor = 10 * 4 * 64 * bytes_per_element
            self.assertEqual(
                qk.metadata["cost_model"]["weight_bytes"],
                expected_one_tensor,
            )
            self.assertEqual(
                pv.metadata["cost_model"]["weight_bytes"],
                expected_one_tensor,
            )
            self.assertEqual(
                self._demand(qk, "gpu0.tensor_core").work_units,
                2 * 2 * 512 * 7,
            )
            self.assertEqual(
                self._demand(pv, "gpu0.tensor_core").work_units,
                2 * 2 * 7 * 512,
            )
            self.assertEqual(
                qkv.metadata["cost_model"]["output_bytes"],
                2 * 512 + 2 * (2 * 4 * 64 * bytes_per_element),
            )

        by_heads = {}
        for kv_heads in (8, 1):
            operators = tuple(
                replace(
                    operator,
                    parameters={**operator.parameters, "kv_heads": kv_heads},
                )
                if operator.op_kind == "attention"
                else operator
                for operator in self.scenario.model.graph.operators
            )
            scenario = replace(
                self.scenario,
                model=replace(
                    self.scenario.model,
                    graph=replace(
                        self.scenario.model.graph,
                        operators=operators,
                    ),
                ),
                fusion_policy=replace(
                    self.scenario.fusion_policy,
                    flash_attention=False,
                ),
                placement=replace(
                    self.scenario.placement,
                    kv_policy=replace(
                        self.scenario.placement.kv_policy,
                        dtype="fp16",
                    ),
                ),
            )
            tasks = compile_serving_cohort_schedule(scenario, cohort).tasks
            by_heads[kv_heads] = tuple(
                next(
                    task
                    for task in tasks
                    if task.name.endswith(suffix)
                )
                for suffix in (
                    "dense0.rank000.attention_qk.gpu_gemm",
                    "dense0.rank000.attention_pv.gpu_gemm",
                )
            )

        self.assertEqual(
            [by_heads[heads][0].metadata["cost_model"]["weight_bytes"]
             for heads in (8, 1)],
            [10240, 1280],
        )
        for index in (0, 1):
            self.assertEqual(
                self._demand(by_heads[8][index], "gpu0.tensor_core").work_units,
                self._demand(by_heads[1][index], "gpu0.tensor_core").work_units,
            )

    def test_materialized_kv_writes_drive_cost_but_append_stays_persistent(self):
        def cohort(materialized_tokens):
            return SimpleNamespace(
                cohort_id="mtp-kv-{}".format(materialized_tokens),
                kind="mtp",
                items=(
                    SimpleNamespace(
                        request_id="request",
                        phase="mtp",
                        token_count=4,
                        context_tokens=8,
                        proposed_tokens=4,
                        expected_accepted_tokens=0.25,
                        kv_append_tokens=1,
                        kv_materialized_tokens=materialized_tokens,
                    ),
                ),
            )

        accepted_only = cohort(1)
        proposed = cohort(4)
        self.assertNotEqual(
            _serving_cohort_cache_key(accepted_only),
            _serving_cohort_cache_key(proposed),
        )

        tasks = compile_serving_cohort_schedule(self.scenario, proposed).tasks
        qkv = next(
            task
            for task in tasks
            if task.name.endswith("dense0.rank000.qkv.gpu_gemm")
        )
        kv_append = next(
            task
            for task in tasks
            if task.metadata.get("event_kind") == "kv_append"
            and task.metadata.get("layer_id") == "dense0"
        )
        self.assertEqual(qkv.metadata["kv_materialized_tokens"], 4)
        self.assertEqual(qkv.metadata["kv_materialized_bytes"], 2048)
        self.assertEqual(qkv.metadata["kv_persistent_append_tokens"], 1)
        self.assertEqual(qkv.metadata["cost_model"]["output_bytes"], 4096)
        self.assertEqual(kv_append.metadata["kv_token_appends"], 1)
        self.assertEqual(kv_append.metadata["bytes"], 512)
        self.assertFalse(kv_append.demands)

    def test_fusion_fails_closed_when_working_set_exceeds_sram_limit(self):
        constrained = replace(
            self.scenario,
            fusion_policy=replace(
                self.scenario.fusion_policy,
                max_fused_working_set_bytes=1,
            ),
        )
        tasks = compile_scenario(constrained).tasks
        rope = next(
            task for task in tasks
            if task.name == "prefill.dense0.rank000.rope.gpu_elementwise"
        )
        self.assertFalse(rope.metadata["fusion_enabled"])
        self.assertEqual(
            rope.metadata["fusion_decision"],
            "working_set_exceeds_sram_limit",
        )
        self.assertGreater(rope.metadata["fusion_working_set_bytes"], 1)
        self.assertEqual(rope.metadata["fusion_sram_limit_bytes"], 1)

    def test_kv_swap_has_one_cpu_page_table_descriptor_and_dma_submit(self):
        cohort = SimpleNamespace(
            cohort_id="swap-out",
            kind="kv_swap_out",
            metadata={
                "source_component": "hbm0",
                "target_component": "hbm1",
                "byte_count": 4096,
            },
        )
        tasks = compile_serving_cohort_schedule(self.scenario, cohort).tasks
        by_kind = {}
        for task in tasks:
            by_kind.setdefault(task.metadata.get("event_kind"), []).append(task)

        self.assertEqual(len(by_kind["kv_swap_cpu_control"]), 1)
        self.assertEqual(
            [task.metadata["phase"] for task in by_kind["kv_swap_descriptor_pack"]],
            ["cpu_dispatch", "cpu_memory"],
        )
        self.assertEqual(len(by_kind["kv_swap_submit"]), 1)
        submit = by_kind["kv_swap_submit"][0]
        self.assertEqual(submit.metadata["submission_count"], 1)
        self.assertEqual(submit.metadata["transfer_kind"], "instruction")
        self.assertEqual(submit.metadata["source_component"], "cpu0")
        self.assertEqual(submit.metadata["target_component"], "gpu0")
        self.assertEqual(submit.metadata["descriptor_bytes"], 32)
        self.assertEqual(submit.metadata["transfer_bytes"], 4096)
        self.assertEqual(
            sum(
                demand.bytes_moved
                for demand in submit.demands
                if demand.resource_id == "cpu0.h2d_dma"
            ),
            32,
        )

    def test_mtp_acceptance_model_selects_rate_or_trace_explicitly(self):
        expected_policy = replace(
            self.scenario.workload.mtp,
            acceptance_model="expected",
            acceptance_rate=1.0,
            acceptance_trace=(0.0,),
        )
        trace_policy = replace(
            expected_policy,
            acceptance_model="trace",
        )

        self.assertEqual(
            _mtp_expected_acceptance(expected_policy, 4, 0), 4.0
        )
        self.assertEqual(
            _mtp_expected_acceptance(trace_policy, 4, 0), 1.0
        )
        with self.assertRaisesRegex(ValueError, "requires acceptance_trace"):
            _mtp_expected_acceptance(
                SimpleNamespace(
                    acceptance_model="trace",
                    acceptance_rate=1.0,
                    acceptance_trace=(),
                ),
                4,
                0,
            )
        with self.assertRaisesRegex(ValueError, "unsupported MTP acceptance_model"):
            _mtp_expected_acceptance(
                SimpleNamespace(
                    acceptance_model="implicit",
                    acceptance_rate=1.0,
                    acceptance_trace=(0.0,),
                ),
                4,
                0,
            )

        configured = replace(
            self.scenario,
            workload=replace(self.scenario.workload, mtp=expected_policy),
        )
        commits = tuple(
            task
            for task in compile_scenario(configured).tasks
            if task.metadata.get("event_kind") == "mtp_commit"
        )
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].metadata["proposed_tokens"], 3)
        self.assertEqual(commits[0].metadata["accepted_tokens"], 3)

    def test_typed_mtp_replays_full_descriptor_chain_per_serial_draft_step(self):
        tasks = tuple(self.schedule.tasks)
        by_id = {task.task_id: task for task in tasks}
        gemms = [
            task
            for task in tasks
            if task.metadata.get("model_operator_id")
            in {"mtp.prediction_layer.000", "mtp.aux_head"}
            and task.metadata.get("phase") == "gpu_gemm"
        ]
        self.assertEqual(
            {
                (task.metadata["model_operator_id"], task.metadata["draft_step"])
                for task in gemms
            },
            {
                ("mtp.prediction_layer.000", 0),
                ("mtp.aux_head", 0),
                ("mtp.prediction_layer.000", 1),
                ("mtp.aux_head", 1),
            },
        )
        for task in gemms:
            self.assertEqual(
                task.metadata["cost_model"]["weight_bytes"],
                task.metadata["declared_weight_bytes"],
            )
            self.assertTrue(task.metadata["proposal_cost_scale_ignored"])

        draft0_aux = next(
            task
            for task in gemms
            if task.metadata["model_operator_id"] == "mtp.aux_head"
            and task.metadata["draft_step"] == 0
        )
        draft1_prediction = next(
            task
            for task in gemms
            if task.metadata["model_operator_id"] == "mtp.prediction_layer.000"
            and task.metadata["draft_step"] == 1
        )

        def ancestors(task):
            pending = list(task.dependencies)
            found = set()
            while pending:
                task_id = pending.pop()
                if task_id in found:
                    continue
                found.add(task_id)
                pending.extend(by_id[task_id].dependencies)
            return found

        self.assertIn(draft0_aux.task_id, ancestors(draft1_prediction))


if __name__ == "__main__":
    unittest.main()
