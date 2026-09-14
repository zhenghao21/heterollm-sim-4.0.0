import unittest
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from heterollm_sim.control_plane_planner import (
    RuntimeExecutionTarget,
    _Candidate,
    _FusionOpportunity,
    _build_fusion_solve_units,
    _expand_solve_assignment,
    _solve_builtin,
    _solve_ortools,
    plan_runtime_placement,
)
from heterollm_sim.ir import ParallelSpec, RankMappingSpec
from heterollm_sim.planner import compile_scenario
from heterollm_sim.reference import build_reference_scenario


_DENSE0_FUSION_PARTICIPANTS = {
    "qkv_rope": (
        "dense0.attention",
        "dense0.attention.rope",
    ),
    "flash_attention": (
        "dense0.attention.qk",
        "dense0.attention.softmax.reduce",
        "dense0.attention.softmax.normalize",
        "dense0.attention.pv",
    ),
    "gemm_epilogue_activation": (
        "dense0.mlp",
        "dense0.mlp.activation",
    ),
    "residual_norm": (
        "dense0.attention.residual",
        "dense0.post_attention_norm.reduce",
        "dense0.post_attention_norm.apply",
    ),
}


def _scenario_with_fusion_policy(scenario, **overrides):
    return replace(
        scenario,
        fusion_policy=replace(scenario.fusion_policy, **overrides),
    )


def _rank_gpu_id(scenario, rank_id=0):
    return scenario.placement.parallel.rank_mapping[rank_id].component_id


def _task_names(schedule):
    return {task.name for task in schedule.tasks}


def _assert_has_task(test_case, task_names, name):
    test_case.assertTrue(name in task_names, "missing task {}".format(name))


def _assert_no_task(test_case, task_names, name):
    test_case.assertFalse(
        name in task_names,
        "unexpected task {}".format(name),
    )


def _task(schedule, name):
    return next(task for task in schedule.tasks if task.name == name)


def _prefill_dense0_rank0_tasks(schedule):
    return tuple(
        task
        for task in schedule.tasks
        if task.name.startswith("prefill.dense0.rank000")
    )


def _fusion_analysis(test_case, result):
    metadata = result.placement.metadata["control_plane"]["decision"]
    test_case.assertTrue(
        "fusion_analysis" in metadata,
        (
            "control_plane decision must include fusion_analysis audit "
            "records; present keys: {}"
        ).format(sorted(metadata)),
    )
    raw = metadata["fusion_analysis"]
    records = raw.values() if isinstance(raw, Mapping) else raw
    flattened = []
    for item in records:
        if isinstance(item, Mapping):
            flattened.append(item)
        else:
            flattened.extend(item)
    return tuple(flattened)


def _fusion_record(test_case, records, group, *, layer_id="dense0", rank_id=0):
    matches = [
        record
        for record in records
        if record.get("layer_id") == layer_id
        and record.get("rank_id") == rank_id
        and record.get("fusion_group") == group
    ]
    if not matches:
        test_case.fail(
            "missing fusion_analysis record for {} rank {}".format(
                group, rank_id
            )
        )
    test_case.assertEqual(len(matches), 1)
    return matches[0]


def _gpu_colocated_dense0_scenario():
    scenario = build_reference_scenario()
    rank_gpu = _rank_gpu_id(scenario)
    op_mapping = dict(scenario.placement.op_to_component)
    for op_keys in _DENSE0_FUSION_PARTICIPANTS.values():
        for op_key in op_keys:
            op_mapping[op_key] = rank_gpu
    return replace(
        scenario,
        placement=replace(scenario.placement, op_to_component=op_mapping),
    )


def _scenario_without_usable_cim():
    scenario = build_reference_scenario()
    components = tuple(
        replace(component, capacity_bytes=1)
        if "cim" in component.kind.lower()
        else component
        for component in scenario.hardware.components
    )
    return replace(
        scenario,
        hardware=replace(scenario.hardware, components=components),
    )


class ControlPlaneFusionTests(unittest.TestCase):
    def test_enabled_fusion_colocates_dense0_groups_on_rank_gpu_and_audits(self):
        scenario = _scenario_without_usable_cim()
        result = plan_runtime_placement(scenario)
        rank_gpu = _rank_gpu_id(scenario)
        fusion_records = _fusion_analysis(self, result)

        self.assertTrue(result.fully_placed, result.unplaced)
        for group, op_keys in _DENSE0_FUSION_PARTICIPANTS.items():
            with self.subTest(group=group):
                self.assertEqual(
                    {
                        result.placement.op_to_component[key]
                        for key in op_keys
                    },
                    {rank_gpu},
                )
                audit = _fusion_record(self, fusion_records, group)
                self.assertTrue(audit["fusion_enabled"])
                self.assertEqual(
                    audit["fusion_decision"],
                    "enabled_same_gpu_sram_resident",
                )
                self.assertEqual(
                    audit["fusion_target_component"],
                    rank_gpu,
                )
                self.assertTrue(audit["colocation_required"])
                self.assertEqual(
                    set(audit["co_located_op_keys"]),
                    set(op_keys),
                )
                self.assertGreater(audit["fusion_working_set_bytes"], 0)
                self.assertGreaterEqual(
                    audit["fusion_sram_limit_bytes"],
                    audit["fusion_working_set_bytes"],
                )

    def test_disabled_fusion_policy_does_not_add_colocation_incentives(self):
        scenario = _scenario_with_fusion_policy(
            _scenario_without_usable_cim(),
            qkv_rope=False,
            flash_attention=False,
            gemm_epilogue_activation=False,
            residual_norm=False,
        )
        result = plan_runtime_placement(scenario)
        fusion_records = _fusion_analysis(self, result)

        self.assertTrue(result.fully_placed, result.unplaced)
        for group in _DENSE0_FUSION_PARTICIPANTS:
            with self.subTest(group=group):
                audit = _fusion_record(self, fusion_records, group)
                self.assertFalse(audit["fusion_enabled"])
                self.assertEqual(
                    audit["fusion_decision"],
                    "disabled_by_policy",
                )
                self.assertFalse(audit["colocation_required"])

    def test_fusion_working_set_limit_does_not_force_colocation_and_is_audited(self):
        scenario = _scenario_with_fusion_policy(
            _scenario_without_usable_cim(),
            max_fused_working_set_bytes=1,
        )
        result = plan_runtime_placement(scenario)
        fusion_records = _fusion_analysis(self, result)

        self.assertTrue(result.fully_placed, result.unplaced)
        for group in _DENSE0_FUSION_PARTICIPANTS:
            with self.subTest(group=group):
                audit = _fusion_record(self, fusion_records, group)
                self.assertFalse(audit["fusion_enabled"])
                self.assertEqual(
                    audit["fusion_decision"],
                    "working_set_exceeds_sram_limit",
                )
                self.assertFalse(audit["colocation_required"])
                self.assertEqual(audit["fusion_sram_limit_bytes"], 1)
                self.assertGreater(audit["fusion_working_set_bytes"], 1)

    def test_mapped_compile_uses_fused_tasks_or_fallbacks_from_fusion_policy(self):
        scenario = _scenario_without_usable_cim()
        mapped = plan_runtime_placement(scenario)
        fused_schedule = compile_scenario(mapped.apply(scenario))
        fused_names = _task_names(fused_schedule)

        _assert_has_task(
            self,
            fused_names,
            "prefill.dense0.rank000.qkv.gpu_gemm",
        )
        self.assertEqual(
            _task(
                fused_schedule,
                "prefill.dense0.rank000.qkv.gpu_gemm",
            ).metadata["cost_model"]["epilogue_name"],
            "rope",
        )
        _assert_no_task(
            self,
            fused_names,
            "prefill.dense0.rank000.rope.gpu_elementwise",
        )
        _assert_has_task(
            self,
            fused_names,
            "prefill.dense0.rank000.attention_flash.gpu_fused_attention",
        )
        _assert_no_task(
            self,
            fused_names,
            "prefill.dense0.rank000.attention_qk.gpu_gemm",
        )
        _assert_no_task(
            self,
            fused_names,
            "prefill.dense0.rank000.attention_pv.gpu_gemm",
        )
        _assert_has_task(
            self,
            fused_names,
            "prefill.dense0.rank000.mlp_up_gate.gpu_gemm",
        )
        self.assertEqual(
            _task(
                fused_schedule,
                "prefill.dense0.rank000.mlp_up_gate.gpu_gemm",
            ).metadata["cost_model"]["epilogue_name"],
            "swiglu",
        )
        _assert_no_task(
            self,
            fused_names,
            "prefill.dense0.rank000.mlp_activation.gpu_elementwise",
        )
        _assert_has_task(
            self,
            fused_names,
            "prefill.dense0.rank000.fused_residual_norm.gpu_elementwise",
        )
        _assert_no_task(
            self,
            fused_names,
            "prefill.dense0.rank000.post_attention_norm_reduce.gpu_reduction",
        )
        _assert_no_task(
            self,
            fused_names,
            "prefill.dense0.rank000.post_attention_norm_apply.gpu_elementwise",
        )

        constrained = _scenario_with_fusion_policy(
            _scenario_without_usable_cim(),
            max_fused_working_set_bytes=1,
        )
        constrained_mapped = plan_runtime_placement(constrained)
        fallback_schedule = compile_scenario(
            constrained_mapped.apply(constrained)
        )
        fallback_names = _task_names(fallback_schedule)

        _assert_has_task(
            self,
            fallback_names,
            "prefill.dense0.rank000.rope.gpu_elementwise",
        )
        _assert_no_task(
            self,
            fallback_names,
            "prefill.dense0.rank000.attention_flash.gpu_fused_attention",
        )
        _assert_has_task(
            self,
            fallback_names,
            "prefill.dense0.rank000.attention_qk.gpu_gemm",
        )
        _assert_has_task(
            self,
            fallback_names,
            "prefill.dense0.rank000.attention_pv.gpu_gemm",
        )
        _assert_has_task(
            self,
            fallback_names,
            "prefill.dense0.rank000.mlp_activation.gpu_elementwise",
        )
        _assert_no_task(
            self,
            fallback_names,
            "prefill.dense0.rank000.fused_residual_norm.gpu_elementwise",
        )
        _assert_has_task(
            self,
            fallback_names,
            "prefill.dense0.rank000.post_attention_norm_reduce.gpu_reduction",
        )
        _assert_has_task(
            self,
            fallback_names,
            "prefill.dense0.rank000.post_attention_norm_apply.gpu_elementwise",
        )

    def test_joint_objective_rewards_legal_fusion_but_keeps_split_alternative(self):
        gpu_only = _scenario_without_usable_cim()
        fused = plan_runtime_placement(gpu_only)
        disabled_scenario = _scenario_with_fusion_policy(
            gpu_only,
            qkv_rope=False,
            flash_attention=False,
            gemm_epilogue_activation=False,
            residual_norm=False,
        )
        split = plan_runtime_placement(disabled_scenario)

        self.assertEqual(
            fused.placement.op_to_component,
            split.placement.op_to_component,
        )
        self.assertLess(fused.objective_value, split.objective_value)
        credits = [
            record["fusion_credit_ns"]
            for record in _fusion_analysis(self, fused)
            if record["fusion_enabled"]
        ]
        self.assertTrue(credits)
        self.assertTrue(all(credit > 0.0 for credit in credits))

        heterogeneous = build_reference_scenario()
        mapped = plan_runtime_placement(heterogeneous)
        records = _fusion_analysis(self, mapped)
        qkv = _fusion_record(self, records, "qkv_rope")
        activation = _fusion_record(
            self, records, "gemm_epilogue_activation"
        )
        self.assertEqual(
            mapped.placement.op_to_component["dense0.attention"],
            "cim0",
        )
        self.assertFalse(qkv["fusion_enabled"])
        self.assertEqual(
            qkv["fusion_decision"],
            "target_is_not_gpu",
        )
        self.assertEqual(
            mapped.placement.op_to_component["dense0.mlp"],
            "cim0",
        )
        self.assertFalse(activation["fusion_enabled"])

    def test_bundle_preserves_partial_mapping_per_original_requirement(self):
        scenario = build_reference_scenario()
        rank_target = RuntimeExecutionTarget(
            rank_id=0,
            tp_rank=0,
            pp_rank=0,
            ep_rank=0,
            compute_component_id="gpu0",
            component_id="gpu0",
        )

        def candidate(index):
            return _Candidate(
                index,
                "gpu0",
                None,
                1.0,
                (),
                rank_execution_targets=(rank_target,),
            )

        opportunity = _FusionOpportunity(
            opportunity_id="dense0:flash_attention:test",
            group="flash_attention",
            variant="test",
            layer_id="dense0",
            member_indices=(0, 1, 2, 3),
            member_op_keys=("qk", "reduce", "normalize", "pv"),
            working_set_bytes=1,
            eliminated_launches=3,
        )
        units = _build_fusion_solve_units(
            scenario,
            (),
            ((candidate(0),), (candidate(1),), (), (candidate(3),)),
            (opportunity,),
        )

        self.assertEqual(len(units), 1)
        self.assertEqual(
            min(choice.missing_count for choice in units[0].candidates),
            1,
        )
        solve = _solve_builtin(
            (units[0].candidates,),
            capacities={},
            base_usage={},
            time_limit_s=1.0,
            missing_weights=(4,),
        )
        expanded = _expand_solve_assignment(
            solve.assignment, units, requirement_count=4
        )
        self.assertEqual(sum(item is None for item in expanded), 1)
        self.assertIsNone(expanded[2])

    def test_planner_fusion_fails_closed_when_participant_maps_to_cpu(self):
        cases = (
            ("qkv_rope", "dense0.attention.rope"),
            ("flash_attention", "dense0.attention.softmax.reduce"),
            ("flash_attention", "dense0.attention.softmax.normalize"),
            ("residual_norm", "dense0.attention.residual"),
            ("gemm_epilogue_activation", "dense0.mlp.activation"),
        )

        for group, cpu_op_key in cases:
            with self.subTest(group=group, cpu_op_key=cpu_op_key):
                scenario = _gpu_colocated_dense0_scenario()
                placement = replace(
                    scenario.placement,
                    op_to_component={
                        **scenario.placement.op_to_component,
                        cpu_op_key: "cpu0",
                    },
                )
                schedule = compile_scenario(
                    replace(scenario, placement=placement)
                )
                dense_tasks = _prefill_dense0_rank0_tasks(schedule)

                enabled = [
                    task.name
                    for task in dense_tasks
                    if task.metadata.get("fusion_group") == group
                    and task.metadata.get("fusion_enabled")
                ]
                explicit_cpu_tasks = [
                    task.name
                    for task in dense_tasks
                    if task.metadata.get("operator_id") == cpu_op_key
                    and task.metadata.get("target_component") == "cpu0"
                ]

                self.assertFalse(
                    enabled,
                    "fusion must be disabled when {} maps to cpu0; got {}".format(
                        cpu_op_key,
                        enabled,
                    ),
                )
                self.assertTrue(
                    explicit_cpu_tasks,
                    "{} must be lowered as an explicit cpu0 fallback task".format(
                        cpu_op_key
                    ),
                )

    def test_tp_residual_norm_uses_full_rank_visible_working_set_consistently(self):
        scenario = _gpu_colocated_dense0_scenario()
        parallel = ParallelSpec(
            tp_degree=2,
            rank_mapping=(
                RankMappingSpec(0, "gpu0", 0, 0, 0, "hbm0"),
                RankMappingSpec(1, "gpu0", 1, 0, 0, "hbm0"),
            ),
            layer_to_stage={"dense0": 0, "moe1": 0},
        )
        scenario = replace(
            scenario,
            placement=replace(
                scenario.placement, parallel=parallel
            ),
            fusion_policy=replace(
                scenario.fusion_policy,
                # The compiled prefill chunk is 32 tokens at hidden=512.
                # This sits between the incorrect TP-local (49,152 B) and
                # full rank-visible (98,304 B)
                # residual+norm working sets.
                max_fused_working_set_bytes=75_000,
            ),
        )
        task_names = _task_names(compile_scenario(scenario))

        _assert_has_task(
            self,
            task_names,
            "prefill.dense0.rank000.attention_residual.gpu_elementwise",
        )
        _assert_has_task(
            self,
            task_names,
            "prefill.dense0.rank000.post_attention_norm_reduce.gpu_reduction",
        )
        _assert_no_task(
            self,
            task_names,
            "prefill.dense0.rank000.fused_residual_norm.gpu_elementwise",
        )

    def test_ortools_timeout_keeps_better_heuristic_incumbent(self):
        class Expression:
            def __add__(self, _other):
                return self

            __radd__ = __add__

            def __mul__(self, _other):
                return self

            __rmul__ = __mul__

            def __eq__(self, _other):
                return self

            def __le__(self, _other):
                return self

        class Variable(Expression):
            def __init__(self, name):
                self.name = name

        class Model:
            def NewBoolVar(self, name):
                return Variable(name)

            def Add(self, _constraint):
                return None

            def Minimize(self, _objective):
                return None

        class Solver:
            def __init__(self):
                self.parameters = SimpleNamespace()

            def Solve(self, _model):
                return 2

            def Value(self, variable):
                return int(variable.name == "x_0_1")

            def BestObjectiveBound(self):
                return 0.0

        fake_cp_model = SimpleNamespace(
            CpModel=Model,
            CpSolver=Solver,
            OPTIMAL=4,
            FEASIBLE=2,
        )
        candidates = (
            (
                _Candidate(0, "cheap", None, 1.0, ()),
                _Candidate(0, "expensive", None, 10.0, ()),
            ),
        )

        with patch(
            "heterollm_sim.control_plane_planner.importlib.import_module",
            return_value=fake_cp_model,
        ):
            solve = _solve_ortools(
                candidates,
                capacities={},
                base_usage={},
                time_limit_s=0.01,
            )

        self.assertFalse(solve.completed)
        self.assertEqual(solve.assignment[0].component_id, "cheap")
        self.assertEqual(solve.objective_value, 1.0)


if __name__ == "__main__":
    unittest.main()
