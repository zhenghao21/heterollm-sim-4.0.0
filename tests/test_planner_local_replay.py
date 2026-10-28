"""The local stage replay keeps every executed record after its dependencies."""
from heterollm_sim.contracts import ResourceDemand, TaskCategory
from heterollm_sim.planner import _replay_local_record_timeline
from heterollm_sim.scalable_serving import TaskExecutionRecord


def _record(task_id, start, end, dependencies=(), resource=None):
    demands = (ResourceDemand(resource, end - start),) if resource else ()
    return TaskExecutionRecord(task_id, start, end, TaskCategory.COMMUNICATION, tuple(dependencies), {}, demands)


def test_zero_service_ties_replay_after_their_dependencies():
    # A counter past 99999 sorts before a shorter one: the zero-service
    # dispatch completion and the weight access it gates tie at one instant.
    compute = _record("g.000001.compute", 0., 40., resource="gpu0")
    dispatch = _record("g.98605.dispatch.complete", 40., 40., [compute.task_id])
    access = _record("g.100004.weights", 40., 40., [dispatch.task_id])
    later = _record("g.100005.expert", 40., 55., [access.task_id], resource="gpu1")
    ordered, ends, intervals = _replay_local_record_timeline([later, access, dispatch, compute])
    assert [record.task_id for record in ordered] == [compute.task_id, dispatch.task_id, access.task_id,
                                                      later.task_id]
    assert intervals[access.task_id] == (40., 40.) and intervals[later.task_id] == (40., 55.)
    assert ends[later.task_id] == 55.
