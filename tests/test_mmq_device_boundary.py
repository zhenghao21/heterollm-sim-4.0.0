"""CPU-declared, GPU-executed MMQ boundaries; compilation only, never hardware."""
from dataclasses import replace
import unittest

from heterollm_sim import planner
from heterollm_sim.cost_models import HostGemmOffloadCapability
from tests.test_mmq_planner import scenario


def _host_declared_mmq_case():
    case = scenario()
    placement = replace(
        case.placement,
        op_to_component={
            **case.placement.op_to_component,
            "attention": "cpu0",
            "full0.norm": "cpu0",
        },
        tensor_to_component={
            **case.placement.tensor_to_component,
            "full0.attention_weights": "hostmem0",
        },
    )
    profiles = {kind: dict(registry) for kind, registry in case.component_profiles.items()}
    profile_id = next(iter(profiles["gpu"]))
    profiles["gpu"][profile_id] = replace(
        profiles["gpu"][profile_id],
        host_gemm_offload=HostGemmOffloadCapability(minimum_m=1, evidence="native MMQ device-boundary test"),
    )
    return replace(case, placement=placement, component_profiles=profiles)


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


class MMQDeviceBoundaryTests(unittest.TestCase):
    def test_host_declared_attention_mmq_transfers_raw_f32_once_then_runs_all_stages_on_gpu(self):
        case = _host_declared_mmq_case()
        schedule = planner.compile_scenario(case)
        q_main = next(task for task in schedule.tasks if task.metadata.get("phase") == "gpu_gemm"
                      and task.metadata.get("projection_id") == "attention.q")
        work = q_main.metadata["mmq_source_work"]
        rows, width = work["m"], work["k"]
        raw_bytes = 4 * rows * width
        transfers = [task for task in schedule.tasks
                     if task.metadata.get("event_kind") == "operator_input_transfer"
                     and task.metadata.get("projection_id") == "attention.q"]
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].metadata["bytes"], raw_bytes)
        self.assertEqual(transfers[0].metadata["target_component"], "gpu0")

        conversion = next(task for task in schedule.tasks if task.metadata.get("phase") == "gpu_elementwise"
                          and task.metadata.get("projection_id") == "attention.q"
                          and task.metadata.get("mmq_source_work", {}).get("stage") == "conversion")
        fixup_launch = next(task for task in schedule.tasks if task.metadata.get("phase") == "kernel_launch"
                            and task.metadata.get("projection_id") == "attention.q"
                            and task.metadata.get("mmq_source_work", {}).get("stage") == "fixup")
        for task in (conversion, q_main, fixup_launch):
            self.assertEqual(task.metadata["target_component"], "gpu0")
        self.assertTrue(q_main.metadata["host_gemm_offload_applied"])
        self.assertEqual(q_main.metadata["placement_component"], "cpu0")
        self.assertEqual(q_main.metadata["execution_component"], "gpu0")
        self.assertIn(transfers[0].task_id, _ancestors(schedule, conversion))
        self.assertIn(conversion.task_id, _ancestors(schedule, q_main))
        self.assertIn(q_main.task_id, _ancestors(schedule, fixup_launch))

        self.assertEqual(q_main.metadata["cost_model"]["output_bytes"], 4 * work["m"] * work["n"])
        self.assertEqual(work["partial_writer_count"], 0)
        self.assertTrue(work["fixup_launch"])
        self.assertEqual(fixup_launch.metadata["cost_model"]["operations"], 0)
        projections_ready = next(task for task in schedule.tasks
                                 if task.metadata.get("event_kind") == "attention_projection_join")
        self.assertIn(fixup_launch.task_id, _ancestors(schedule, projections_ready))


if __name__ == "__main__":
    unittest.main()
