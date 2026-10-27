"""Privately produced, immutable native timeline rollback columns.

Public snapshot calls stay ordinary dictionaries. Only external-import rollback
snapshots request these seven read-only, scalar/tuple-valued columns.
Their historical allocation tables also share native self-copying extents.
"""
import copy
import copyreg
from collections.abc import Mapping
from types import MappingProxyType, ModuleType
from weakref import WeakKeyDictionary
from ._resource_ledger import memo_keys_are_exact_int

from .memory_allocator import (
    _NATIVE_ALLOCATION_COPY_CONTRACT as _ALLOCATION_COPY_CONTRACT,
    PhysicalAddressAllocator as _CACHE_ALLOCATOR_CLASS)


_ALLOCATION_HISTORY_CACHE = WeakKeyDictionary()


_COLUMNS = ("ready_ns", "directions", "busy_ns", "bytes_moved", "last_intervals",
            "lane_available", "lane_directions")
_DEEPCOPY = copy.deepcopy
_DICT_DEEPCOPY = copy._deepcopy_dict
_DISPATCH = copy._deepcopy_dispatch
_HANDLERS = tuple(_DISPATCH.items())
_REDUCERS = copy.dispatch_table
_SET_REDUCERS = tuple((kind, _REDUCERS.get(kind)) for kind in (set, frozenset))
_BUILTINS = _DEEPCOPY.__builtins__
_FUNCTIONS = tuple((module, name, function, function.__code__, function.__defaults__,
                    function.__kwdefaults__, tuple((function.__kwdefaults__ or {}).items()))
    for module, names in ((copy, ("deepcopy", "_deepcopy_atomic", "_deepcopy_dict", "_deepcopy_tuple",
                                 "_deepcopy_list", "_deepcopy_method", "_reconstruct", "_keep_alive")),
                          (copyreg, ("_slotnames", "__newobj__", "__newobj_ex__", "_reconstructor")))
    for name in names for function in (vars(module)[name],))
_BUILTIN_NAMES = ("type", "getattr", "hasattr", "issubclass", "isinstance", "id", "len", "zip",
                  "tuple", "list", "dict", "set", "str", "object", "iter", "next", "range",
                  "enumerate", "bool", "hash")
_GLOBALS = tuple((module, name, vars(module).get(name, _BUILTINS[name]))
                 for module in (copy, copyreg) for name in _BUILTIN_NAMES)


def _copy_bindings_native():
    # Read raw namespaces first. Deleted globals must not call module __getattr__
    # while deciding whether ordinary deepcopy must handle the column.
    if type(copy) is not ModuleType or type(copyreg) is not ModuleType:
        return False
    for module in (copy, copyreg):
        if any(type(key) is not str for key in vars(module)):
            return False
    namespace = vars(copy)
    if (namespace.get("deepcopy") is not _DEEPCOPY
            or namespace.get("_deepcopy_dispatch") is not _DISPATCH
            or namespace.get("dispatch_table") is not _REDUCERS
            or type(_DISPATCH) is not dict or type(_REDUCERS) is not dict
            or any(type(kind) is not type for kind in _DISPATCH)
            or any(type(kind) is not type for kind in _REDUCERS)
            or any(_DISPATCH.get(kind) is not handler for kind, handler in _HANDLERS)
            or any(_REDUCERS.get(kind) is not handler for kind, handler in _SET_REDUCERS)):
        return False
    for module, name, function, code, defaults, keywords, keyword_items in _FUNCTIONS:
        if (vars(module).get(name) is not function or function.__code__ is not code
                or function.__defaults__ is not defaults or function.__kwdefaults__ is not keywords
                or function.__builtins__ is not _BUILTINS):
            return False
        if keywords is not None and (type(keywords) is not dict
                or any(type(key) is not str for key in keywords)
                or len(keywords) != len(keyword_items)
                or any(keywords.get(key) is not value for key, value in keyword_items)):
            return False
    if type(_BUILTINS) is not dict or any(type(key) is not str for key in _BUILTINS):
        return False
    for module, name, original in _GLOBALS:
        if vars(module).get(name, _BUILTINS.get(name)) is not original:
            return False
    return True


class _ImmutableTimelineColumn(tuple, Mapping):
    """Owned native values with immutable storage and O(1) mapping lookups."""
    __slots__ = ()
    __hash__ = None
    __eq__ = Mapping.__eq__

    def __ne__(self, other):
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __new__(cls, owned, leaf_ids):
        return tuple.__new__(cls, (MappingProxyType(owned), leaf_ids))

    def __getitem__(self, key):
        return tuple.__getitem__(self, 0)[key]

    def __iter__(self):
        return iter(tuple.__getitem__(self, 0))

    def __len__(self):
        return len(tuple.__getitem__(self, 0))

    def __contains__(self, key):
        return key in tuple.__getitem__(self, 0)

    def keys(self):
        return tuple.__getitem__(self, 0).keys()

    def items(self):
        return tuple.__getitem__(self, 0).items()

    def values(self):
        return tuple.__getitem__(self, 0).values()

    def _plain(self):
        return tuple.__getitem__(self, 0).copy()

    def __deepcopy__(self, memo):
        if (type(memo) is dict and memo_keys_are_exact_int(memo)
                and (id(memo) not in memo or type(memo[id(memo)]) is list)
                and memo.keys().isdisjoint(tuple.__getitem__(self, 1))
                and _copy_bindings_native()):
            return self
        return _copy_private_mapping(self, memo)


def _native_value(value, leaf_ids, tuples):
    kind = type(value)
    if (kind is str or kind is int or kind is float or kind is bool
            or kind is bytes or kind is complex or value is None):
        leaf_ids.add(id(value))
        return True
    if kind is tuple:
        identity = id(value)
        if identity in tuples:
            return True
        tuples.add(identity)
        leaf_ids.add(identity)
        return all(_native_value(item, leaf_ids, tuples) for item in value)
    return False


def _freeze_timeline_columns(state):
    """Reuse each already captured column; unknown columns remain plain dicts."""
    if type(state) is not dict or not _copy_bindings_native():
        return state
    for name in _COLUMNS:
        values = state.get(name)
        if type(values) is not dict:
            continue
        leaf_ids, tuples = set(), set()
        try:
            native = all(type(key) is str and _native_value(key, leaf_ids, tuples)
                         and _native_value(value, leaf_ids, tuples)
                         for key, value in values.items())
        except RecursionError:
            native = False
        if native:
            # The fresh owned map has no external mutable alias. Keys and all
            # values are held alive by the proxy; every recorded ID stays live.
            state[name] = _ImmutableTimelineColumn(values.copy(), frozenset(leaf_ids))
    return state



def _allocation_bindings_native():
    kind, mro, function, code, defaults, keywords, descriptor = _ALLOCATION_COPY_CONTRACT
    if type(kind) is not type or type.__getattribute__(kind, '__mro__') is not mro:
        return False
    namespace = type.__getattribute__(kind, '__dict__')
    if (any(type(name) is not str for name in namespace)
            or namespace.get('__deepcopy__') is not function
            or function.__code__ is not code or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not keywords
            or namespace.get('__dict__') is not descriptor
            or namespace.get('__getattribute__', object.__getattribute__)
               is not object.__getattribute__):
        return False
    dispatch = vars(copy).get('_deepcopy_dispatch')
    return type(dispatch) is dict and all((type(key) is type for key in dispatch)) and (dispatch.get(kind) is None)

def _copy_private_mapping(value, memo):
    dispatch = vars(copy).get('_deepcopy_dispatch')
    if type(dispatch) is dict and all((type(kind) is type for kind in dispatch)):
        copier = dispatch.get(dict)
        if copier is _DICT_DEEPCOPY:
            return _DICT_DEEPCOPY(value, memo)
        if copier is not None:
            return copier(value, memo)
    return _DEEPCOPY(tuple.__getitem__(value, 0).copy(), memo)

class _ImmutableAllocationColumn(_ImmutableTimelineColumn):
    """Only the historical table is shared; extents already copy as themselves."""
    __slots__ = ()

    def __deepcopy__(self, memo):
        if _allocation_bindings_native():
            return _ImmutableTimelineColumn.__deepcopy__(self, memo)
        return _copy_private_mapping(self, memo)

def _native_cache_allocators(allocators):
    return (type(allocators) is dict
            and all(type(owner) is str for owner in allocators)
            and type(_CACHE_ALLOCATOR_CLASS) is type
            and type.__getattribute__(_CACHE_ALLOCATOR_CLASS, '__hash__') is object.__hash__
            and type.__getattribute__(_CACHE_ALLOCATOR_CLASS, '__eq__') is object.__eq__)

def _freeze_allocation_state(state, allocators=None):
    if type(state) is not dict or not _copy_bindings_native() or (not _allocation_bindings_native()):
        return state
    cache_enabled = _native_cache_allocators(allocators)
    native_class = _ALLOCATION_COPY_CONTRACT[0]
    for owner, values in state.items():
        if type(owner) is not str:
            continue
        allocator = allocators.get(owner) if cache_enabled else None
        if type(allocator) is not _CACHE_ALLOCATOR_CLASS:
            allocator = None
        if type(values) is not dict or not values:
            if allocator is not None:
                _ALLOCATION_HISTORY_CACHE.pop(allocator, None)
            continue
        cached = _ALLOCATION_HISTORY_CACHE.get(allocator) if allocator is not None else None
        same = (type(cached) is _ImmutableAllocationColumn and len(values) == len(cached)
                and all(key is previous_key and value is previous_value
                        for (key, value), (previous_key, previous_value)
                        in zip(values.items(), tuple.__getitem__(cached, 0).items())))
        leaf_ids = None if same else set()
        native = True
        cache_safe = allocator is not None
        for key, value in values.items():
            if (type(key) is not tuple or len(key) != 2 or type(key[0]) is not str
                    or type(key[1]) is not int or type(value) is not native_class):
                native = False
                break
            attributes = object.__getattribute__(value, '__dict__')
            if type(attributes) is not dict:
                native = False
                break
            for name, item in attributes.items():
                if type(name) is not str or name == '__deepcopy__':
                    native = False
                    break
                if cache_safe:
                    kind = type(item)
                    if not (item is None or kind is int or kind is str or kind is bool
                            or kind is float or kind is bytes or kind is complex):
                        cache_safe = False
            if not native:
                break
            if leaf_ids is not None:
                leaf_ids.update((id(key), id(key[0]), id(key[1]), id(value)))
        if not native:
            if allocator is not None:
                _ALLOCATION_HISTORY_CACHE.pop(allocator, None)
            continue
        if same and cache_safe:
            state[owner] = cached
            continue
        if leaf_ids is None:
            leaf_ids = set()
            for key, value in values.items():
                leaf_ids.update((id(key), id(key[0]), id(key[1]), id(value)))
        column = _ImmutableAllocationColumn(values.copy(), frozenset(leaf_ids))
        state[owner] = column
        if cache_safe:
            _ALLOCATION_HISTORY_CACHE[allocator] = column
        elif allocator is not None:
            _ALLOCATION_HISTORY_CACHE.pop(allocator, None)
    return state
