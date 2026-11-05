"""One request through the numeric FR-FCFS controller, for tests of its state machine."""
from heterollm_sim._compiled_hbm import execute_fr_fcfs_numeric


def execute_hbm_numeric(core, request):
    """The request's result, or None when the numeric controller declines without a change."""
    outcome = execute_fr_fcfs_numeric(core, (request,))
    return None if outcome is None else outcome[0][0]
