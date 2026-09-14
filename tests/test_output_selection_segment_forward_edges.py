"""Late-bound input edges must survive task-segment capture without reordering."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim import planner


def _graph(request, prefix, *, padding=0, forward=True):
    builder = planner._TaskBuilder(planner.RequestSpec(request, 0, 1, 1))
    for index in range(padding):
        builder.add("padding{}".format(index), planner.TaskCategory.POLICY)
    external = builder.add("external", planner.TaskCategory.POLICY)
    first = len(builder.tasks)
    counter = builder.counter
    upload = builder.add(prefix + ".upload", planner.TaskCategory.COMMUNICATION,
                         dependencies=(external,), advance=False,
                         metadata={"allocation_id": prefix + ".indices"})
    gate = builder.add(prefix + ".gate", planner.TaskCategory.MEMORY,
                       dependencies=(external,), advance=False)
    if forward:
        builder.tasks[first] = replace(builder.tasks[first], dependencies=(external, gate))
    terminal = builder.add(prefix + ".consume", planner.TaskCategory.COMPUTE,
                           dependencies=(upload, gate),
                           metadata={"invocation_id": request + ":" + prefix,
                                     "allocation_id": prefix + ".indices"})
    builder._rank_value_components[terminal] = {0: "gpu0"}
    builder._last_coherent_dma_task = upload
    capture = dict(first_task_index=first, source_prefix=prefix,
                   source_counter_before=counter, source_dependencies=(external,),
                   source_initial_previous=external, source_initial_dma=None,
                   terminal_task_id=terminal)
    return builder, capture


class OutputSelectionSegmentForwardEdgesTests(unittest.TestCase):
    def test_forward_edge_replay_preserves_order_ids_metadata_and_builder_state(self):
        source, arguments = _graph("source", "source-prefix")
        template = planner._TaskSegmentTemplate.capture(source, **arguments)
        self.assertIsNotNone(template)

        expected, expected_arguments = _graph("target", "target-prefix", padding=2)
        actual = planner._TaskBuilder(planner.RequestSpec("target", 0, 1, 1))
        for index in range(2):
            actual.add("padding{}".format(index), planner.TaskCategory.POLICY)
        external = actual.add("external", planner.TaskCategory.POLICY)
        result = template.replay(actual, prefix="target-prefix", dependencies=(external,))

        self.assertEqual(result, expected_arguments["terminal_task_id"])
        self.assertEqual(actual.tasks, expected.tasks)
        self.assertEqual(actual.counter, expected.counter)
        self.assertEqual(actual.previous, expected.previous)
        self.assertEqual(actual._last_coherent_dma_task, expected._last_coherent_dma_task)
        self.assertEqual(actual._rank_value_components, expected._rank_value_components)
        self.assertEqual(template.dependency_refs[0], (arguments["source_dependencies"][0], 1))

    def test_capture_rejects_cycle_self_loop_duplicate_and_unbound_dependency(self):
        for invalid in ("cycle", "self_loop", "duplicate", "unbound"):
            with self.subTest(invalid=invalid):
                source, arguments = _graph("source", "source-prefix")
                first = arguments["first_task_index"]
                upload, gate = source.tasks[first:first + 2]
                if invalid == "cycle":
                    source.tasks[first + 1] = replace(gate, dependencies=(upload.task_id,))
                elif invalid == "self_loop":
                    source.tasks[first] = replace(upload, dependencies=(upload.task_id,))
                elif invalid == "duplicate":
                    source.tasks.append(upload)
                else:
                    source.tasks[first] = replace(upload, dependencies=("outside-unbound",))
                self.assertIsNone(planner._TaskSegmentTemplate.capture(source, **arguments))

    def test_ordered_capture_keeps_fast_path_and_identical_replay(self):
        source, arguments = _graph("source", "source-prefix", forward=False)
        with patch.object(planner, "TopologicalSorter", side_effect=AssertionError(
                "already ordered tasks must not invoke a graph traversal")):
            template = planner._TaskSegmentTemplate.capture(source, **arguments)
        self.assertIsNotNone(template)
        expected, expected_arguments = _graph("target", "target-prefix", forward=False)
        actual = planner._TaskBuilder(planner.RequestSpec("target", 0, 1, 1))
        external = actual.add("external", planner.TaskCategory.POLICY)
        result = template.replay(actual, prefix="target-prefix", dependencies=(external,))
        self.assertEqual(result, expected_arguments["terminal_task_id"])
        self.assertEqual(actual.tasks, expected.tasks)


if __name__ == "__main__":
    unittest.main()
