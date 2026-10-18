"""Task-local physical instructions and ordered execution output.

Preparation resolves allocations before any access is submitted. The ledger
keeps the original transaction order, without inventing logical transactions
for HOST-staged HBF acceptance. Public metadata is materialized once per task.
Neither object owns a rollback boundary; the event kernel or standalone
physical resolver continues to own that boundary.
"""
from __future__ import annotations

from .memory_types import Operation

try:
    import cython
except ImportError:
    class _PythonMode:
        bint = bool
        @staticmethod
        def cclass(cls):
            return cls
        @staticmethod
        def ccall(function):
            return function
        @staticmethod
        def declare(*args, **kwargs):
            return None
    cython = _PythonMode()


@cython.cclass
class PreparedPhysicalTask:
    metadata = cython.declare(object, visibility="public")
    config = cython.declare(object, visibility="public")
    owners = cython.declare(object, visibility="public")
    instructions = cython.declare(object, visibility="public")
    staging_bindings = cython.declare(object, visibility="public")

    def __init__(self, metadata, config, owners, instructions, staging_bindings):
        self.metadata = metadata
        self.config = config
        self.owners = owners
        self.instructions = instructions
        self.staging_bindings = staging_bindings


@cython.cclass
class PhysicalTaskLedger:
    records = cython.declare(list, visibility="public")
    staging_stats = cython.declare(dict, visibility="public")
    completion_ns = cython.declare(object, visibility="public")
    all_erases = cython.declare(cython.bint, visibility="public")

    def __init__(self):
        self.records = []
        self.staging_stats = {}
        self.completion_ns = None
        self.all_erases = True

    @cython.ccall
    def accept(self, operation, completion_ns):
        previous = self.completion_ns
        self.completion_ns = completion_ns if previous is None else max(previous, completion_ns)
        self.all_erases = self.all_erases and operation is Operation.ERASE
