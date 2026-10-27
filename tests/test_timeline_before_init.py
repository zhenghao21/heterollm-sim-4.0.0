"""The init-only accelerator retains ordinary Python constructor dispatch."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import FunctionType

import pytest

from heterollm_sim import _resource_ledger, memory_transfer

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "tests/test_resource_ledger.py").is_file())

# The complete old constructor, also used by the fresh-process benchmark.
REFERENCE_INIT = '''def __init__(self, transaction, name):
    self.transaction, self.name = transaction, name
'''
_reference_namespace = {}
exec(REFERENCE_INIT, _reference_namespace)
reference_init = _reference_namespace["__init__"]


@pytest.fixture(params=("source", "compiled"))
def init(request):
    if request.param == "compiled":
        if Path(_resource_ledger.__file__).suffix == ".py":
            pytest.skip("optional compiled ledger is not installed")
        return _resource_ledger.timeline_before_init
    spec = importlib.util.spec_from_file_location("before_init_python",
        ROOT / "src/heterollm_sim/_resource_ledger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.timeline_before_init


def constructor_observation(implementation, mode):
    calls, created = [], []
    transaction = object()
    class Child(memory_transfer._TimelineBefore):
        __init__ = implementation
        def __new__(cls, transaction, name):
            calls.append("new")
            view = super().__new__(cls)
            created.append(view)
            return view
        def __setattr__(self, key, value):
            calls.append((key, value is transaction if key == "transaction" else value))
            if mode == "raise-" + key:
                raise ValueError("authored " + key)
            object.__setattr__(self, key, value)
            if mode == "rebind-setattr" and key == "transaction":
                def replacement(self, key, value):
                    calls.append(("replacement", key, value))
                    object.__setattr__(self, key, value)
                Child.__setattr__ = replacement
    try:
        view = Child(transaction, "lane_available")
        result = ("return", view.transaction is transaction, view.name)
    except Exception as error:
        result = ("raise", type(error), str(error))
    values = vars(created[0])
    state = tuple((key, value is transaction if key == "transaction" else value)
                  for key, value in values.items())
    return result, calls, state


@pytest.mark.parametrize("mode", ("normal", "raise-transaction", "raise-name", "rebind-setattr"))
def test_new_and_setattr_keep_order_values_exception_prefix_and_live_rebinding(init, mode):
    assert constructor_observation(init, mode) == constructor_observation(reference_init, mode)


def test_data_descriptors_and_subclass_constructor_overrides(init):
    def observe(implementation):
        calls = []
        transaction = object()
        class View(memory_transfer._TimelineBefore):
            __init__ = implementation
            transaction = property(fset=lambda self, value: calls.append(("transaction", value is transaction)))
            name = property(fset=lambda self, value: calls.append(("name", value)))
        View(transaction, "lane_available")
        class OwnInit(View):
            def __init__(self, transaction, name):
                calls.append(("own-init", transaction is transaction_marker, name))
        transaction_marker = transaction
        OwnInit(transaction, "own")
        foreign = object()
        class ForeignNew(View):
            def __new__(cls, transaction, name):
                calls.append("foreign-new")
                return foreign
        assert ForeignNew(transaction, "ignored") is foreign
        return calls
    assert observe(init) == observe(reference_init)


def test_keyword_arguments_and_bound_init_keep_the_same_attributes(init):
    for implementation in (reference_init, init):
        class View(memory_transfer._TimelineBefore):
            __init__ = implementation
        transaction = object()
        view = View(transaction=transaction, name="lane_available")
        assert view.transaction is transaction and view.name == "lane_available"
        replacement = object()
        view.__init__(transaction=replacement, name="busy_ns")
        assert view.transaction is replacement and view.name == "busy_ns"


def test_real_before_lanes_installs_only_init_and_retains_python_readers():
    cls = memory_transfer._TimelineBefore
    assert cls.__init__ is _resource_ledger.timeline_before_init
    assert isinstance(cls.get, FunctionType) and isinstance(cls.__getitem__, FunctionType)
    assert cls.get.__module__ == cls.__getitem__.__module__ == memory_transfer.__name__
    assert dict(memory_transfer._OWNED_TIMELINE_BEFORE_METHODS)["__init__"] is cls.__init__
    timeline = memory_transfer.ResourceTimeline()
    timeline.lane_available["r"] = [1., 3.]
    token = timeline._begin_owned_transaction()
    try:
        view = token.before_lanes()
        assert type(view) is cls and view.transaction is token and view.name == "lane_available"
        assert view.get("r") == (1., 3.) and view["r"] == (1., 3.)
        assert view.get("missing", "default") == "default"
    finally:
        token.finish()
