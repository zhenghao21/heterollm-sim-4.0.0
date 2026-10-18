"""Persistent exclusions from a scan, separate from its physical manifest."""
from __future__ import annotations

import json
from pathlib import Path


def load_cancellations(output_dir: Path, manifest: dict) -> dict[str, dict]:
    path = Path(output_dir) / "queue_plan.json"
    if not path.exists():
        return {}
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != "b200-scan-plan/v1":
        raise ValueError("invalid scan queue plan schema")
    known = {point["point_id"]: point for point in manifest["points"]}
    cancellations = {}
    for row in plan["cancelled_points"]:
        identifier = row["point_id"]
        if identifier not in known or identifier in cancellations:
            raise ValueError("cancelled points must be unique manifest members")
        if any(row.get(key) != value for key, value in known[identifier].items()):
            raise ValueError("cancelled point differs from the physical manifest")
        if (row.get("status") != "cancelled" or not row.get("reason")
                or not row.get("cancelled_at") or row.get("output_tokens") == 1024):
            raise ValueError("invalid long-output cancellation")
        saved_path = Path(output_dir) / "points" / (identifier + ".json")
        if saved_path.exists():
            saved = json.loads(saved_path.read_text(encoding="utf-8-sig"))
            if saved.get("status") != "cancelled":
                raise ValueError("cancellation conflicts with an existing result: " + identifier)
            if saved.get("ttft_ns") is not None or saved.get("tpot_ns") is not None:
                raise ValueError("cancelled point must have empty metrics: " + identifier)
        cancellations[identifier] = row
    return cancellations
