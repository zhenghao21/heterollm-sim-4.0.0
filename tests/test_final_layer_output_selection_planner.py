"""Bounded task-graph checks; no hardware or end-to-end performance run."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim import planner
from heterollm_sim.final_layer_output_selection import SOURCE_KEY, source_declaration
from heterollm_sim.ir import LayerSpec, LinearAttentionSpec, ParallelSpec, RankMappingSpec, WorkloadSpec
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.scalable_serving import TaskExecutionRecord
from heterollm_sim.serving import BatchCohort, BatchItem
from tests.model_helpers import model_from_layer_specs


ARCHITECTURES = ("qwen2_decoder", "llama_decoder", "qwen3_5_hybrid_transformer")
AUDIT = "final_layer_output_selection"
GEOMETRY = "output_selection_tensor_geometry"


def _scenario(architecture, *, declared=True):
    """Use the existing small F32-storage fixture geometry and GPU placement."""
    base = build_reference_scenario()
    first = LayerSpec("first", "dense", hidden_size=32, intermediate_size=64,
                      attention_heads=4, kv_heads=2, dtype="fp16")
    last = replace(first, layer_id="last")
    if architecture == "qwen3_5_hybrid_transformer":
        first = replace(first, sequence_mixer="linear_attention",
                        linear_attention=LinearAttentionSpec(
                            key_heads=2, value_heads=2, key_head_dim=16,
                            value_head_dim=16, conv_kernel_size=4))
    model = model_from_layer_specs(
        "output-selection-test", (first, last), architecture=architecture,
        vocabulary_size=64, max_sequence_length=4096, embedding_weight_bytes=640,
        metadata={SOURCE_KEY: source_declaration()} if declared else {},
    )
    placement = replace(
        base.placement, model_name=model.name,
        op_to_component={"attention": "gpu0", "linear_attention": "gpu0",
                         "mlp": "gpu0", "lm_head": "gpu0"},
        tensor_to_component={name: "hbm0" for name in
                             ("embedding_weights", "lm_head_weights", "kv_cache", "linear_state")},
        tensor_bytes={"embedding_weights": 640, "lm_head_weights": 640,
                      "kv_cache": 4096, "linear_state": 4096},
        parallel=ParallelSpec(tp_degree=1, pp_degree=1, ep_degree=1,
                              rank_mapping=(RankMappingSpec(0, "gpu0", 0, 0, 0, "hbm0"),)),
    )
    return replace(base, model=model, placement=placement,
                   workload=WorkloadSpec("output-selection", request_count=1,
                       prompt_tokens=128, output_tokens=1,
                       metadata={"llama_cpp_f32_hidden_storage": True}))


def _cohort(tokens, logits, *, name="selection", context=0, phase="prefill"):
    return BatchCohort(name, phase, 0.0, (
        BatchItem("request", phase, tokens, context,
                  kv_append_tokens=tokens, kv_materialized_tokens=tokens,
                  logit_tokens=logits),
    ))


def _tasks(schedule, event, *, phase=None, layer=None):
    return [task for task in schedule.tasks
            if task.metadata.get("event_kind") == event
            and (phase is None or task.metadata.get("phase") == phase)
            and (layer is None or ".{}.".format(layer) in task.name)]


def _facts(schedule):
    records = tuple(TaskExecutionRecord(
        task.task_id, 0.0, 0.0, task.category, task.dependencies,
        task.metadata, task.demands) for task in schedule.tasks)
    return {row["task_id"]: row for row in json.loads(json.dumps(
        tuple(planner._execution_task_facts(records, ("request",)))))}


class FinalLayerOutputSelectionPlannerTests(unittest.TestCase):
    def assert_projection_rows(self, schedule, layer, rows):
        projections = [task for task in schedule.tasks
                       if task.metadata.get("layer_id") == layer
                       and task.metadata.get("phase") == "gpu_gemm"
                       and task.metadata.get("projection_id") in {"mlp.up_gate", "mlp.down"}]
        self.assertEqual(len(projections), 2 if rows else 0)
        for task in projections:
            width = 32 if task.metadata["projection_id"] == "mlp.up_gate" else 64
            self.assertEqual(task.metadata["cost_model"]["activation_bytes"], rows * width * 4)
        residuals = _tasks(schedule, "mlp_residual", phase="gpu_elementwise", layer=layer)
        self.assertEqual(len(residuals), 1 if rows else 0)
        if residuals:
            self.assertEqual(residuals[0].metadata["cost_model"]["write_bytes"], rows * 32 * 4)

    def test_three_architectures_select_only_the_source_defined_tail(self):
        for architecture in ARCHITECTURES:
            for tokens, logits in ((128, 0), (128, 1), (128, 7), (128, 128), (1, 1)):
                with self.subTest(architecture=architecture, tokens=tokens, logits=logits):
                    schedule = planner.compile_serving_cohort_schedule(
                        _scenario(architecture), _cohort(tokens, logits))
                    late = architecture == "qwen3_5_hybrid_transformer"
                    tail_rows = tokens if late else logits
                    self.assert_projection_rows(schedule, "first", tokens)
                    self.assert_projection_rows(schedule, "last", tail_rows)
                    norms = _tasks(schedule, "final_norm_apply", phase="gpu_elementwise")
                    self.assertEqual(len(norms), 1 if tail_rows else 0)
                    if norms:
                        self.assertEqual(norms[0].metadata["cost_model"]["write_bytes"], tail_rows * 32 * 4)
                    heads = _tasks(schedule, "lm_head_projection", phase="gpu_gemm")
                    self.assertEqual(len(heads), 1 if logits else 0)
                    if heads:
                        self.assertEqual(heads[0].metadata["cost_model"]["activation_bytes"], logits * 32 * 4)
                        self.assertEqual(heads[0].metadata["cost_model"]["output_bytes"], logits * 64 * 4)
                    audited = [task.metadata[AUDIT] for task in schedule.tasks if AUDIT in task.metadata]
                    self.assertTrue(audited, "even a zero-output invocation needs source-selection audit facts")
                    for audit in audited:
                        self.assertEqual(audit["token_rows"], tokens)
                        self.assertEqual(audit["logit_rows"], logits)
                        self.assertEqual(audit["selected_indices"], list(range(tokens - logits, tokens)))
                        self.assertEqual(audit["ffn_rows"], tail_rows)
                        self.assertEqual(audit["final_norm_rows"], tail_rows)
                        self.assertIn("stage", audit)

    def test_attention_kv_and_prior_layer_keep_all_input_rows(self):
        for architecture in ARCHITECTURES:
            for logits in (0, 1, 7):
                with self.subTest(architecture=architecture, logits=logits):
                    cohort = _cohort(128, logits, context=17)
                    baseline = planner.compile_serving_cohort_schedule(_scenario(architecture, declared=False), cohort)
                    selected = planner.compile_serving_cohort_schedule(_scenario(architecture), cohort)
                    before = {task.name: task for task in baseline.tasks}
                    preserved = [task for task in selected.tasks
                                 if task.metadata.get("layer_id") == "first"
                                 or task.metadata.get("projection_id", "").startswith("attention.")
                                 or task.metadata.get("event_kind") == "fused_attention"]
                    self.assertTrue(preserved)
                    for task in preserved:
                        self.assertEqual(task.demands, before[task.name].demands, task.name)
                        self.assertEqual(task.metadata.get("cost_model"), before[task.name].metadata.get("cost_model"), task.name)
                    qkv = [task for task in selected.tasks
                           if task.metadata.get("projection_id") == "attention.qkv"
                           and task.metadata.get("phase") == "gpu_gemm"]
                    self.assertTrue(qkv)
                    for task in qkv:
                        self.assertEqual(task.metadata["cost_model"]["activation_bytes"], 128 * 32 * 4)
                        self.assertEqual(task.metadata["kv_materialized_tokens"], 128)
                        self.assertEqual(task.metadata["kv_persistent_append_tokens"], 128)

    def test_gather_branches_and_single_shared_host_index(self):
        for architecture in ARCHITECTURES:
            for tokens, logits in ((128, 0), (128, 1), (128, 7), (128, 128), (1, 1)):
                with self.subTest(architecture=architecture, tokens=tokens, logits=logits):
                    schedule = planner.compile_serving_cohort_schedule(_scenario(architecture), _cohort(tokens, logits))
                    branches = 1 if architecture == "qwen3_5_hybrid_transformer" else 2
                    copies = [task for task in _tasks(schedule, "output_row_selection")
                              if task.metadata.get("phase") != "kernel_launch"]
                    self.assertEqual(len(copies), branches if logits else 0)
                    indices = _tasks(schedule, "output_row_indices")
                    # Source scans all T byte-sized flags even when none select logits.
                    self.assertEqual(len(indices), 1)
                    transfers = _tasks(schedule, "output_row_index_transfer")
                    self.assertEqual(len(transfers), 1 if logits else 0)
                    self.assertEqual(indices[0].metadata[AUDIT]["stage"], "output_indices")
                    expected_stages = ({"final_norm_rows"} if branches == 1 else
                                       {"attention_output_rows", "residual_input_rows"})
                    self.assertEqual({task.metadata[AUDIT]["stage"] for task in copies},
                                     expected_stages if logits else set())
                    for task in copies:
                        self.assertIn(GEOMETRY, task.metadata)
                        self.assertEqual(task.metadata[AUDIT]["gather_count"], branches)

    def test_saved_execution_facts_keep_selection_and_tensor_geometry(self):
        for architecture in ARCHITECTURES:
            for logits in (0, 1, 7):
                with self.subTest(architecture=architecture, logits=logits):
                    schedule = planner.compile_serving_cohort_schedule(_scenario(architecture), _cohort(128, logits))
                    facts = _facts(schedule)
                    audited = [task for task in schedule.tasks if AUDIT in task.metadata]
                    self.assertTrue(audited)
                    for task in audited:
                        self.assertEqual(facts[task.task_id]["metadata"][AUDIT], task.metadata[AUDIT])
                    for task in schedule.tasks:
                        if GEOMETRY in task.metadata:
                            self.assertEqual(facts[task.task_id]["metadata"][GEOMETRY],
                                             json.loads(json.dumps(task.metadata[GEOMETRY])))

    def test_multi_request_selection_uses_physical_row_indices(self):
        cohort = BatchCohort("sparse", "prefill", 0.0, (
            BatchItem("first-request", "prefill", 64, 0, kv_append_tokens=64,
                      kv_materialized_tokens=64, logit_tokens=1),
            BatchItem("second-request", "prefill", 64, 0, kv_append_tokens=64,
                      kv_materialized_tokens=64, logit_tokens=2),
        ))
        for architecture in ARCHITECTURES[:2]:
            with self.subTest(architecture=architecture):
                schedule = planner.compile_serving_cohort_schedule(_scenario(architecture), cohort)
                audited = [task.metadata[AUDIT] for task in schedule.tasks if AUDIT in task.metadata]
                self.assertTrue(audited)
                for audit in audited:
                    self.assertEqual(audit["token_rows"], 128)
                    self.assertEqual(audit["logit_rows"], 3)
                    self.assertEqual(audit["selected_indices"], [63, 126, 127])
                self.assert_projection_rows(schedule, "last", 3)

    def test_supported_decode_template_replay_matches_fresh_compilation(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                scenario = _scenario(architecture)
                source = _cohort(1, 1, name="cold", context=9, phase="decode")
                target = _cohort(1, 1, name="replay", context=9, phase="decode")
                expected = planner.compile_serving_cohort_schedule(scenario, target)
                context = planner.CompilationContext(scenario, compiled_serving_invocation_segments=True)
                with patch.object(planner, "_compile_parallel_layer_body", wraps=planner._compile_parallel_layer_body) as compile_body:
                    with planner._compilation_scope(scenario, context):
                        planner.compile_serving_cohort_schedule(scenario, source)
                        first_calls = compile_body.call_count
                        actual = planner.compile_serving_cohort_schedule(scenario, target)
                self.assertGreater(first_calls, 0)
                self.assertEqual(compile_body.call_count, first_calls)
                self.assertEqual(actual, expected)
                self.assertTrue(any(AUDIT in task.metadata for task in actual.tasks))
                self.assertEqual(_facts(actual), _facts(expected))

    def test_hybrid_dynamic_invocation_replay_keeps_selection_after_context_change(self):
        scenario = _scenario("qwen3_5_hybrid_transformer")
        source = _cohort(1, 1, name="cold", context=9, phase="decode")
        target = _cohort(1, 1, name="replay", context=14, phase="decode")
        expected = planner.compile_serving_cohort_schedule(scenario, target)
        context = planner.CompilationContext(scenario, eager_full_attention_segments=False,
                                               compiled_serving_invocation_segments=True)
        with patch.object(planner, "_compile_parallel_iteration", wraps=planner._compile_parallel_iteration) as compile_body:
            with planner._compilation_scope(scenario, context):
                planner.compile_serving_cohort_schedule(scenario, source)
                first_calls = compile_body.call_count
                actual = planner.compile_serving_cohort_schedule(scenario, target)
        self.assertEqual(first_calls, 1)
        self.assertEqual(compile_body.call_count, first_calls)
        self.assertEqual(actual, expected)
        self.assertTrue(any(AUDIT in task.metadata for task in actual.tasks))
        self.assertEqual(_facts(actual), _facts(expected))

    def test_missing_declaration_keeps_legacy_path_without_selection_events(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                schedule = planner.compile_serving_cohort_schedule(
                    _scenario(architecture, declared=False), _cohort(128, 1))
                self.assertFalse(_tasks(schedule, "output_row_selection"))
                self.assertFalse(_tasks(schedule, "output_row_indices"))
                self.assertFalse(any(AUDIT in task.metadata for task in schedule.tasks))
                self.assert_projection_rows(schedule, "last", 128)
                norm = _tasks(schedule, "final_norm_apply", phase="gpu_elementwise")
                self.assertEqual(len(norm), 1)
                self.assertEqual(norm[0].metadata["cost_model"]["write_bytes"], 32 * 4)

    def test_lm_head_explicit_cpu_placement_is_honored(self):
        scenario = _scenario("llama_decoder")
        placement = replace(
            scenario.placement,
            op_to_component={**scenario.placement.op_to_component, "lm_head": "cpu0"},
        )
        scenario = replace(scenario, placement=placement)
        schedule = planner.compile_serving_cohort_schedule(
            scenario, _cohort(8, 1)
        )
        heads = _tasks(schedule, "lm_head_projection", phase="cpu_gemm")
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads[0].metadata["target_component"], "cpu0")
        self.assertFalse(_tasks(schedule, "lm_head_projection", phase="gpu_gemm"))

    def test_long_prefill_keeps_autoregressive_lm_head_to_final_row(self):
        """A long prompt still computes only the final prompt logit row.

        llama.cpp's ordinary completion path has ``logits_all=false``: the
        transformer evaluates all prompt rows, but the output projection and
        sampling consume only the final row.  Keep that distinction explicit
        so a prompt length (for example TinyLlama's 29-token trace) is not
        accidentally used as the lm-head GEMM batch.
        """
        scenario = _scenario("llama_decoder")
        scenario = replace(
            scenario,
            workload=replace(scenario.workload, prompt_tokens=29, output_tokens=2),
        )
        schedule = planner.compile_scenario(scenario)
        heads = [
            task
            for task in schedule.tasks
            if task.metadata.get("event_kind") == "lm_head_projection"
            and task.metadata.get("phase") == "gpu_gemm"
            and (task.name.startswith("prefill.") or task.name.startswith("decode0001."))
        ]
        self.assertEqual(
            {"prefill" if task.name.startswith("prefill.") else "decode0001" for task in heads},
            {"prefill", "decode0001"},
        )
        for task in heads:
            audit = task.metadata[AUDIT]
            self.assertEqual(audit["logit_rows"], 1)
            self.assertEqual(audit["ffn_rows"], 1)
            self.assertEqual(audit["final_norm_rows"], 1)
            self.assertEqual(task.metadata["cost_model"]["activation_bytes"], 32 * 4)
        prefill = next(task for task in heads if task.name.startswith("prefill."))
        self.assertEqual(prefill.metadata[AUDIT]["token_rows"], 29)
        self.assertEqual(prefill.metadata[AUDIT]["selected_indices"], [28])
        decode = next(task for task in heads if task.name.startswith("decode0001."))
        self.assertEqual(decode.metadata[AUDIT]["token_rows"], 1)
        self.assertEqual(decode.metadata[AUDIT]["selected_indices"], [0])


if __name__ == "__main__":
    unittest.main()
