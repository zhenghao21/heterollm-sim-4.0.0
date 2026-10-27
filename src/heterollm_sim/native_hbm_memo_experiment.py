"""Opt-in experiment: memoize complete, bit-identical HBM numeric transfers.

The numeric dispatch has an explicit context bridge. On a hit the complete
numeric post-state is copied back, then its caller commits/publishes it.
No time normalization, partial state key or namespace-derived equivalence
is used. Normal execution never imports or enables this context manager.

No functions are replaced. The native-hook ownership guard separately
validates the controlled cache and its current native cancellation check;
unknown cache/check implementations retain the ordinary safe fallback.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import math
import struct
import sys
import zlib
from threading import RLock
from types import MethodType

import numpy as np

from .execution_control import ExecutionControl, native_callback_free_control


_ACTIVE = ContextVar("native_hbm_numeric_memo_experiment", default=None)
MAX_MEMO_BYTES = 2 * 1024**3
IMAGE_FORMAT_VERSION = 1
_RAW_CODEC = 0
_ZLIB_CODEC = 1
_COMPRESS_MIN_BYTES = 4096
_EMPTY_LRU_BYTES = sys.getsizeof(OrderedDict())
_LOCK_TYPE = type(RLock())
_NATIVE_CANCEL = ExecutionControl.raise_if_cancelled


def _owned_size(value):
    """Conservatively count each cache entry, including Python containers."""
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        size += sum(_owned_size(item) for item in value)
    return size


def _array_key(array):
    return (array.dtype.str, array.shape, array.strides, _encode_image(array.tobytes(order="C")))


def _encode_image(contents):
    """Retain all bits, with a deterministic codec rather than a digest key."""
    if len(contents) >= _COMPRESS_MIN_BYTES:
        return (IMAGE_FORMAT_VERSION, _ZLIB_CODEC, len(contents), zlib.compress(contents, level=1))
    return (IMAGE_FORMAT_VERSION, _RAW_CODEC, len(contents), contents)


def _decode_image(image):
    if (type(image) is not tuple or len(image) != 4
            or type(image[0]) is not int or image[0] != IMAGE_FORMAT_VERSION
            or type(image[1]) is not int or image[1] not in (_RAW_CODEC, _ZLIB_CODEC)
            or type(image[2]) is not int or image[2] < 0 or type(image[3]) is not bytes):
        raise ValueError("invalid numeric memo image frame")
    contents = image[3] if image[1] == _RAW_CODEC else zlib.decompress(image[3])
    if len(contents) != image[2]:
        raise ValueError("numeric memo image length mismatch")
    return contents


def _image_sizes(images):
    raw_bytes = stored_bytes = 0
    for image in images:
        raw_bytes += image[2]
        stored_bytes += len(image[3])
    return raw_bytes, stored_bytes


class NumericMemoExperiment:
    def __init__(self, *, max_bytes, min_request_bytes, check):
        if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
                or not 0 <= max_bytes <= MAX_MEMO_BYTES):
            raise ValueError("max_bytes must be an integer between zero and 2 GiB")
        if (isinstance(min_request_bytes, bool) or not isinstance(min_request_bytes, int)
                or min_request_bytes < 0):
            raise ValueError("min_request_bytes must be a non-negative integer")
        self.max_bytes, self.min_request_bytes, self.check = max_bytes, min_request_bytes, check
        self._entries, self._lock = OrderedDict(), RLock()
        self.stats = {"calls": 0, "hits": 0, "misses": 0, "declined": 0,
                      "evictions": 0, "entries": 0, "resident_bytes": 0,
                      "resident_payload_bytes": 0, "peak_resident_bytes": 0,
                      "image_format_version": IMAGE_FORMAT_VERSION,
                      "raw_image_bytes_encoded": 0, "stored_image_bytes_encoded": 0,
                      "raw_resident_image_bytes": 0, "stored_resident_image_bytes": 0,
                      "peak_raw_resident_image_bytes": 0, "peak_stored_resident_image_bytes": 0,
                      "replay_failures": 0,
                      "matched_request_bytes": 0, "native_hook_guard_preserved": True}

    def _check(self):
        if self.check is not None:
            self.check()

    def _refresh_resident(self):
        # Count the actual retained compressed bytes and their containers.
        # The empty cache object exists even when disabled; only its growth
        # belongs to retained entries. Transient serialization is separate.
        self.stats["entries"] = len(self._entries)
        self.stats["resident_bytes"] = (self.stats["resident_payload_bytes"]
            + sys.getsizeof(self._entries) - _EMPTY_LRU_BYTES) if self._entries else 0

    def _forget_entry(self, key, expected=None):
        entry = self._entries.get(key)
        if entry is None or (expected is not None and entry is not expected):
            return
        del self._entries[key]
        self.stats["resident_payload_bytes"] -= entry[2]
        self.stats["raw_resident_image_bytes"] -= entry[3]
        self.stats["stored_resident_image_bytes"] -= entry[4]
        if not self._entries:
            self._entries.clear()  # Release any now-empty OrderedDict table.
        self._refresh_resident()

    def _make_key(self, first, stop, arrival, read, g, t, arrays, execution_policy):
        if (type(first) is not int or type(stop) is not int or type(read) is not bool
                or type(arrival) is not float or not math.isfinite(arrival)
                or len(arrays) != 14 or type(g) is not np.ndarray or type(t) is not np.ndarray):
            return None
        all_arrays = (g, t, *arrays)
        if any(type(array) is not np.ndarray or array.dtype.hasobject
               or not array.flags.c_contiguous for array in all_arrays):
            return None
        if any(not array.flags.writeable for array in arrays):
            return None
        # C-contiguous arrays have one byte extent. Declining aliases avoids
        # replaying a post-image under a different write-sharing relationship.
        extents = sorted((array.ctypes.data, array.ctypes.data + array.nbytes)
                         for array in all_arrays if array.nbytes)
        if any(left[1] > right[0] for left, right in zip(extents, extents[1:])):
            return None
        if type(execution_policy) is not tuple or any(type(value) not in (str, int, bool) for value in execution_policy):
            return None
        return ((first, stop, struct.pack("=d", arrival), read, execution_policy),
                tuple(_array_key(array) for array in all_arrays))

    def execute(self, original, first, stop, arrival, read, g, t, arrays, scratch, *, execution_policy=()):
        self._check()
        with self._lock:
            self.stats["calls"] += 1
        try:
            request_bytes = (stop - first) * int(g[9])
        except (TypeError, ValueError, IndexError, OverflowError):
            with self._lock:
                self.stats["declined"] += 1
            return original(first, stop, arrival, read, g, t, arrays, scratch)
        if request_bytes < self.min_request_bytes or not self.max_bytes:
            with self._lock:
                self.stats["declined"] += 1
            return original(first, stop, arrival, read, g, t, arrays, scratch)
        try:
            key = self._make_key(first, stop, arrival, read, g, t, arrays, execution_policy)
        except (MemoryError, TypeError, ValueError, IndexError):
            key = None
        if key is None:
            with self._lock:
                self.stats["declined"] += 1
            return original(first, stop, arrival, read, g, t, arrays, scratch)
        key_raw_bytes, key_stored_bytes = _image_sizes(item[3] for item in key[1])
        with self._lock:
            self.stats["raw_image_bytes_encoded"] += key_raw_bytes
            self.stats["stored_image_bytes_encoded"] += key_stored_bytes
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
        if entry is not None:
            self._check()
            post_state, result, _size, _raw_bytes, _stored_bytes = entry
            # Shapes, dtype and complete input bits matched. Memory errors or
            # cancellation during replay propagate to the caller's ordinary
            # rollback path; rerunning on a partially copied state is unsafe.
            try:
                for array, image in zip(arrays, post_state):
                    contents = _decode_image(image)
                    np.copyto(array, np.frombuffer(contents, dtype=array.dtype).reshape(array.shape), casting="no")
            except Exception:
                # Do not rerun a partially copied transition. The unchanged
                # owner transaction restores it; a bad image is not reusable.
                with self._lock:
                    self._forget_entry(key, entry)
                    self.stats["replay_failures"] += 1
                raise
            with self._lock:
                self.stats["hits"] += 1
                self.stats["matched_request_bytes"] += request_bytes
            self._check()
            return copy.deepcopy(result)
        with self._lock:
            self.stats["misses"] += 1
        # Scratch is reset by every active channel and carries no numeric
        # history into a request. It remains leased by the unchanged caller.
        result = original(first, stop, arrival, read, g, t, arrays, scratch)
        self._check()
        if not (isinstance(result, tuple) and len(result) == 5
                and math.isfinite(float(result[0])) and type(result[4]) is np.ndarray):
            return result
        try:
            post = tuple(_encode_image(array.tobytes(order="C")) for array in arrays)
            post_raw_bytes, post_stored_bytes = _image_sizes(post)
            saved_result = copy.deepcopy(result)
            raw_bytes, stored_bytes = key_raw_bytes + post_raw_bytes, key_stored_bytes + post_stored_bytes
            entry = (post, saved_result, 0, raw_bytes, stored_bytes)
            size = _owned_size(key) + _owned_size(entry)
            # An integer may need a larger Python representation than zero.
            size = _owned_size(key) + _owned_size((post, saved_result, size, raw_bytes, stored_bytes))
            entry = (post, saved_result, size, raw_bytes, stored_bytes)
            with self._lock:
                self.stats["raw_image_bytes_encoded"] += post_raw_bytes
                self.stats["stored_image_bytes_encoded"] += post_stored_bytes
            if size > self.max_bytes:
                return result
            with self._lock:
                self._forget_entry(key)
                self._entries[key] = entry
                self.stats["resident_payload_bytes"] += size
                self.stats["raw_resident_image_bytes"] += raw_bytes
                self.stats["stored_resident_image_bytes"] += stored_bytes
                self._refresh_resident()
                while self._entries and self.stats["resident_bytes"] > self.max_bytes:
                    self._forget_entry(next(iter(self._entries)))
                    self.stats["evictions"] += 1
                self.stats["peak_resident_bytes"] = max(self.stats["peak_resident_bytes"], self.stats["resident_bytes"])
                for kind in ("raw", "stored"):
                    peak, current = f"peak_{kind}_resident_image_bytes", f"{kind}_resident_image_bytes"
                    self.stats[peak] = max(self.stats[peak], self.stats[current])
        except MemoryError:
            # Execution already succeeded; failure to retain a memo entry is
            # immaterial to the returned native result and controller state.
            pass
        return result

    def clear(self):
        with self._lock:
            self._entries.clear()
            self.stats.update(entries=0, resident_bytes=0, resident_payload_bytes=0,
                              raw_resident_image_bytes=0, stored_resident_image_bytes=0)


_NATIVE_MEMO_CLASS = NumericMemoExperiment
_NATIVE_MEMO_NAMESPACE = tuple(vars(NumericMemoExperiment).items())
_NATIVE_MEMO_FUNCTIONS = (("_owned_size", _owned_size), ("_array_key", _array_key),
                        ("_encode_image", _encode_image), ("_decode_image", _decode_image),
                        ("_image_sizes", _image_sizes), ("zlib", zlib),
                        ("native_callback_free_control", native_callback_free_control))
_NATIVE_MEMO_LIBRARY_FUNCTIONS = ((np, "copyto", np.copyto), (np, "frombuffer", np.frombuffer),
    (copy, "deepcopy", copy.deepcopy), (sys, "getsizeof", sys.getsizeof),
    (struct, "pack", struct.pack), (math, "isfinite", math.isfinite),
    (zlib, "compress", zlib.compress), (zlib, "decompress", zlib.decompress))
_NATIVE_MEMO_CONSTANTS = (("IMAGE_FORMAT_VERSION", 1), ("_RAW_CODEC", 0), ("_ZLIB_CODEC", 1),
                        ("_COMPRESS_MIN_BYTES", 4096), ("_EMPTY_LRU_BYTES", _EMPTY_LRU_BYTES))
_NATIVE_MEMO_FIELDS = frozenset(("max_bytes", "min_request_bytes", "check", "_entries", "_lock", "stats"))


def native_memo_hooks(memo):
    """Prove the opt-in bridge cannot invoke replaceable Python callbacks.

    The exact cache implementation writes only the numeric arrays supplied
    to execute and its private containers. Native cancellation/deadline checks
    can raise, but cannot edit physical owners or their resource calendars.
    The native numeric caller already snapshots every possible written
    channel and the whole touched bitmap, including memo replay failures.
    """
    if type(memo) is not _NATIVE_MEMO_CLASS or NumericMemoExperiment is not _NATIVE_MEMO_CLASS:
        return False
    namespace = vars(NumericMemoExperiment)
    if len(namespace) != len(_NATIVE_MEMO_NAMESPACE) or any(
            namespace.get(name) is not original for name, original in _NATIVE_MEMO_NAMESPACE):
        return False
    if (any(globals().get(name) is not original for name, original in _NATIVE_MEMO_FUNCTIONS)
            or any(type(globals().get(name)) is not int or globals().get(name) != value
                   for name, value in _NATIVE_MEMO_CONSTANTS)
            or any(getattr(module, name, None) is not original
                   for module, name, original in _NATIVE_MEMO_LIBRARY_FUNCTIONS)):
        return False
    fields = vars(memo)
    if (set(fields) != _NATIVE_MEMO_FIELDS or type(fields["_entries"]) is not OrderedDict
            or type(fields["_lock"]) is not _LOCK_TYPE or type(fields["stats"]) is not dict
            or type(fields["max_bytes"]) is not int or not 0 <= fields["max_bytes"] <= MAX_MEMO_BYTES
            or type(fields["min_request_bytes"]) is not int or fields["min_request_bytes"] < 0
            or any(type(key) is not str or type(value) not in (int, bool)
                   for key, value in fields["stats"].items())):
        return False
    check = fields["check"]
    return (type(check) is MethodType and check.__func__ is _NATIVE_CANCEL
            and native_callback_free_control(check.__self__))


@contextmanager
def hbm_numeric_memo_experiment(*, max_bytes=MAX_MEMO_BYTES,
                                min_request_bytes=128 * 1024, check=None):
    """Enable one context-local exact cache and yield its statistics object.

    ``check`` may be the active ExecutionControl.raise_if_cancelled; it is
    called before lookup/replay and after native misses. Deadline/cancellation
    are current control decisions, not numeric cache-key inputs. Calls in
    other threads/contexts go directly to the original numeric dispatcher.

    Images keep every raw bit, using zlib level 1 above a fixed size. The
    LRU's compressed payload/container accounting is capped by max_bytes;
    transient serialization/copy memory and native scratch are not the LRU.
    """
    if check is None:
        check = ExecutionControl().raise_if_cancelled
    memo = NumericMemoExperiment(max_bytes=max_bytes, min_request_bytes=min_request_bytes, check=check)
    token = _ACTIVE.set(memo)
    try:
        yield memo
    finally:
        _ACTIVE.reset(token)
        memo.clear()


__all__ = ["hbm_numeric_memo_experiment", "NumericMemoExperiment"]
