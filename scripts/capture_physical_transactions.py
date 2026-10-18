"""Optional exact transaction observations for implementation diagnostics.

This module is loaded only when explicitly requested by a benchmark. Traced
runs cannot establish speedup: serialization observes every returned request,
page completion and resolved physical task without altering the controllers.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import wraps
import gzip
import importlib
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _open_trace(path, mode):
    path = Path(path)
    return (gzip.open(path, mode + "t", encoding="utf-8", compresslevel=1)
            if path.suffix == ".gz" else path.open(mode, encoding="utf-8"))


def exact_value(value):
    """Keep float bits, scalar types and native array bytes in diagnostic JSON."""
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "value": exact_value(value.value)}
    if type(value) is bool:
        return {"bool": value}
    if value is None or type(value) in (int, str):
        return value
    if type(value) is float:
        return {"float64": value.hex()}
    if isinstance(value, np.ndarray):
        return {"array": value.dtype.str, "shape": list(value.shape),
                "bytes": value.tobytes(order="C").hex()}
    if isinstance(value, np.generic):
        return {"scalar": value.dtype.str, "bytes": value.tobytes().hex()}
    if is_dataclass(value):
        return {"type": type(value).__qualname__, "fields": {
            item.name: exact_value(getattr(value, item.name)) for item in fields(value)}}
    if isinstance(value, Mapping):
        # Mapping order is not a transaction observable. Preserve key types
        # too, including the tuple keys used by controller calendars.
        pairs = [[exact_value(key), exact_value(item)] for key, item in value.items()]
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return {"mapping": pairs}
    if type(value) in (tuple, list):
        return {type(value).__name__: [exact_value(item) for item in value]}
    if type(value) in (set, frozenset):
        items = [exact_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return {type(value).__name__: items}
    raise TypeError(f"unsupported transaction observation: {type(value).__qualname__}")


class PhysicalTransactionCapture:
    """Observe public physical execution boundaries in a diagnostic process."""

    def __init__(self, path):
        self.path = Path(path)
        self.count = 0
        self._dram_depth = 0

    def write(self, kind, value):
        self._output.write(json.dumps({"kind": kind, "value": exact_value(value)},
                                      ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1

    @staticmethod
    def owner(core):
        return tuple((name, core.config.metadata.get(name)) for name in (
            "bank_resource_prefix", "array_resource_prefix", "host_resource_id"))

    def __enter__(self):
        from heterollm_sim import data_motion
        from heterollm_sim import _compiled_dram
        from heterollm_sim.dram_core import DramCore
        from heterollm_sim.nand_core import NandCore

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stack = ExitStack()
        self._output = self._stack.enter_context(_open_trace(self.path, "w"))

        def physical(original):
            @wraps(original)
            def observed(task, runtime, arrival_ns, **kwargs):
                result = original(task, runtime, arrival_ns, **kwargs)
                self.write("physical_task", (task, arrival_ns, result))
                return result
            return observed

        def dram(original, many):
            @wraps(original)
            def observed(core, requests, **kwargs):
                if many:
                    requests = tuple(requests)
                outer = self._dram_depth == 0
                self._dram_depth += 1
                try:
                    result = original(core, requests, **kwargs)
                finally:
                    self._dram_depth -= 1
                if outer:
                    rows = zip(requests, result) if many else ((requests, result),)
                    for request, completion in rows:
                        self.write("dram_request", (self.owner(core), request, completion))
                return result
            return observed

        def nand(original, kind):
            @wraps(original)
            def observed(core, *args, **kwargs):
                result = original(core, *args, **kwargs)
                if result is not None:
                    self.write(kind, (self.owner(core), args, kwargs, result))
                return result
            return observed

        def host_scope_execute(original):
            @wraps(original)
            def observed(core, request, *args, **kwargs):
                result = original(core, request, *args, **kwargs)
                if result is not None and kwargs.get("_scope") is not None and self._dram_depth == 0:
                    self.write("dram_request", (self.owner(core), request, result))
                return result
            return observed

        self._stack.enter_context(patch.object(data_motion, "resolve_physical_task",
                                               physical(data_motion.resolve_physical_task)))
        for name, many in (("submit", False), ("submit_many", True)):
            self._stack.enter_context(patch.object(DramCore, name, dram(getattr(DramCore, name), many)))
        for name in ("submit", "submit_read_pages", "submit_write_pages"):
            self._stack.enter_context(patch.object(NandCore, name, nand(getattr(NandCore, name), name)))
        # New implementations can execute HOST requests through a task-local
        # resident controller. Observe the same returned request, without
        # forcing publication or routing back through the public submitter.
        io_name = "heterollm_sim._io_execution"
        if importlib.util.find_spec(io_name) is not None:
            scope_module = importlib.import_module(io_name)
        elif getattr(_compiled_dram, "HostTaskScope", None) is not None:
            scope_module = _compiled_dram
        else:
            scope_module = None
        if scope_module is not None:
            self._stack.enter_context(patch.object(scope_module, "execute_compiled",
                                                  host_scope_execute(scope_module.execute_compiled)))
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)


def compare_transaction_files(baseline_path, candidate_path):
    """Compare the complete ordered trace, stopping at its first difference."""
    count = 0
    with _open_trace(baseline_path, "r") as before, _open_trace(candidate_path, "r") as after:
        for count, (left, right) in enumerate(zip_longest(before, after), 1):
            if left == right:
                continue
            if left is None or right is None or json.loads(left) != json.loads(right):
                return {"exactly_equal": False, "first_different_record": count,
                        "baseline_kind": json.loads(left)["kind"] if left else None,
                        "candidate_kind": json.loads(right)["kind"] if right else None}
    return {"exactly_equal": True, "record_count": count}
