"""Aggregate ordinary SRAM facts from executed native physical records."""
from __future__ import annotations


def summarize_sram_task_traffic(tasks):
    # The existing iterator owns wrapper-role filtering and per-task/owner
    # deduplication for mixed memory tasks. No descriptors are executed here.
    from .planner import _physical_execution_rows
    fields = ("logical_bytes", "logical_read_bytes", "logical_write_bytes",
              "physical_bytes", "physical_read_bytes", "physical_write_bytes", "burst_count")
    totals = dict.fromkeys(fields, 0)
    owners, resources, task_ids = {}, {}, set()
    rows = 0
    for task, owner, raw, kind in _physical_execution_rows(tasks):
        if kind not in {"SRAM", "MEMORYKIND.SRAM"}:
            continue
        rows += 1
        task_ids.add(task.task_id)
        row = owners.setdefault(str(owner), {**dict.fromkeys(fields, 0), "transaction_count": 0})
        for key in fields:
            value = raw.get(key)
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            for destination in (totals, row):
                destination[key] = (destination[key] + value
                    if valid and destination[key] is not None else None)
        count = raw.get("operation_count")
        row["transaction_count"] = (row["transaction_count"] + count
            if type(count) is int and row["transaction_count"] is not None else None)
        for resource, amount in raw.get("resource_bytes", {}).items():
            resources[str(resource)] = resources.get(str(resource), 0) + amount
    return {"schema": "heterollm.sram-traffic/v1", "task_count": len(task_ids),
            "recorded_owner_rows": rows, **totals, "owners": owners,
            "resource_bytes": dict(sorted(resources.items())),
            "scope": "executed native SRAM physical owner records; transaction_count counts AccessRequest operations, not bursts"}


__all__ = ["summarize_sram_task_traffic"]
