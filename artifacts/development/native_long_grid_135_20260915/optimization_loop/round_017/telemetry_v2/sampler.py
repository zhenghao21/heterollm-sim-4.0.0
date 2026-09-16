"""Scoped high-resolution waitable timer + SM read windows. No inference or GPU writes.

QPC targets are monotonic absolute deadlines. Win32 receives one-shot relative
100ns due times derived afresh from each deadline; UTC absolute timers are not
misused as QPC timers. No timeBeginPeriod, process priority, clock or power change.
"""
from __future__ import annotations
from dataclasses import dataclass
import ctypes as ct
from ctypes import wintypes as wt
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable

NS_PER_SECOND = 1_000_000_000
CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
SYNCHRONIZE = 0x00100000
TIMER_MODIFY_STATE = 0x00000002
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF
INFINITE = 0xFFFFFFFF


def require_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class DeadlineGrid:
    origin_qpc: int
    frequency: int
    period_ns: int = 5_000_000

    def __post_init__(self):
        require_int(self.origin_qpc, 'origin QPC')
        require_int(self.frequency, 'QPC frequency')
        require_int(self.period_ns, 'period nanoseconds')
        if self.period_ns * self.frequency < NS_PER_SECOND:
            raise ValueError('period smaller than one QPC tick')

    def deadline(self, index: int) -> int:
        if type(index) is not int or index < 0:
            raise ValueError('index must be nonnegative integer')
        return self.origin_qpc + ceil_div(index * self.period_ns * self.frequency, NS_PER_SECOND)

    def next_future_index(self, after_qpc: int, current_index: int) -> int:
        # Strictly future deadline, not a burst of catch-up samples.
        return max(current_index + 1, (after_qpc - self.origin_qpc) * NS_PER_SECOND // (self.period_ns * self.frequency) + 1)


class Win32DeadlineTimer:
    """Owns unnamed, non-inheritable stop+high-resolution timer handles only."""
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Win32 high-resolution timer requires Windows')
        self.lib = ct.WinDLL('kernel32', use_last_error=True)
        self.lib.QueryPerformanceCounter.argtypes = [ct.POINTER(ct.c_longlong)]
        self.lib.QueryPerformanceCounter.restype = wt.BOOL
        self.lib.QueryPerformanceFrequency.argtypes = [ct.POINTER(ct.c_longlong)]
        self.lib.QueryPerformanceFrequency.restype = wt.BOOL
        self.lib.CreateEventW.argtypes = [ct.c_void_p, wt.BOOL, wt.BOOL, wt.LPCWSTR]
        self.lib.CreateEventW.restype = wt.HANDLE
        self.lib.CreateWaitableTimerExW.argtypes = [ct.c_void_p, wt.LPCWSTR, wt.DWORD, wt.DWORD]
        self.lib.CreateWaitableTimerExW.restype = wt.HANDLE
        self.lib.SetWaitableTimerEx.argtypes = [wt.HANDLE, ct.POINTER(ct.c_longlong), wt.LONG, ct.c_void_p, ct.c_void_p, ct.c_void_p, wt.ULONG]
        self.lib.SetWaitableTimerEx.restype = wt.BOOL
        self.lib.WaitForMultipleObjects.argtypes = [wt.DWORD, ct.POINTER(wt.HANDLE), wt.BOOL, wt.DWORD]
        self.lib.WaitForMultipleObjects.restype = wt.DWORD
        self.lib.SetEvent.argtypes = [wt.HANDLE]; self.lib.SetEvent.restype = wt.BOOL
        self.lib.CloseHandle.argtypes = [wt.HANDLE]; self.lib.CloseHandle.restype = wt.BOOL
        self.lib.CancelWaitableTimer.argtypes = [wt.HANDLE]; self.lib.CancelWaitableTimer.restype = wt.BOOL
        self.state_lock = threading.Lock(); self.closed = False; self.waiting = False
        self.stop_handle = None; self.timer_handle = None; self.close_errors = []
        f = ct.c_longlong()
        self._check(self.lib.QueryPerformanceFrequency(ct.byref(f)), 'QueryPerformanceFrequency')
        self.frequency = require_int(f.value, 'QPC frequency')
        self.stop_handle = self.lib.CreateEventW(None, True, False, None)
        self._check(self.stop_handle, 'CreateEventW')
        self.timer_handle = self.lib.CreateWaitableTimerExW(None, None, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                                           SYNCHRONIZE | TIMER_MODIFY_STATE)
        if not self.timer_handle:
            error = ct.get_last_error()
            self.lib.CloseHandle(self.stop_handle); self.stop_handle = None; self.closed = True
            raise OSError(error, 'CREATE_WAITABLE_TIMER_HIGH_RESOLUTION unavailable; no silent coarse fallback')
        self.metadata = {'backend': 'CreateWaitableTimerExW', 'flags': CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                         'periodic_timer_used': False, 'tolerable_delay_ms': 0,
                         'QPC_deadline_policy': 'absolute monotonic phase; rearmed one-shot relative100ns',
                         'timeBeginPeriod_called': False, 'global_timer_resolution_changed': False}

    @staticmethod
    def _check(ok, name):
        if not ok:
            raise OSError(ct.get_last_error(), name + ' failed')

    def now(self) -> int:
        v = ct.c_longlong(); self._check(self.lib.QueryPerformanceCounter(ct.byref(v)), 'QueryPerformanceCounter')
        return v.value

    def signal_stop(self):
        with self.state_lock:
            if not self.closed:
                self._check(self.lib.SetEvent(self.stop_handle), 'SetEvent(stop)')

    def wait_until(self, deadline_qpc: int) -> dict[str, Any]:
        with self.state_lock:
            if self.closed or self.waiting:
                raise RuntimeError('timer closed or more than one waiter')
            self.waiting = True
        begin = self.now(); arms = []
        try:
            handles = (wt.HANDLE * 2)(self.stop_handle, self.timer_handle)
            while True:
                now = self.now()
                # Poll stop even for an already-due target; stop wins over overdue work.
                state = self.lib.WaitForMultipleObjects(2, handles, False, 0)
                if state == WAIT_OBJECT_0:
                    return {'status': 'stopped', 'wait_begin_qpc': begin, 'wait_end_qpc': self.now(), 'arms': arms}
                if state == WAIT_FAILED:
                    raise OSError(ct.get_last_error(), 'WaitForMultipleObjects(stop poll) failed')
                if now >= deadline_qpc:
                    return {'status': 'deadline_reached', 'wait_begin_qpc': begin, 'wait_end_qpc': now, 'arms': arms}
                # SetWaitableTimerEx accepts UTC absolute or relative100ns, not QPC absolute.
                due_100ns = max(1, ceil_div((deadline_qpc - now) * 10_000_000, self.frequency))
                due = ct.c_longlong(-due_100ns)
                arm_start = self.now()
                self._check(self.lib.SetWaitableTimerEx(self.timer_handle, ct.byref(due), 0, None, None, None, 0), 'SetWaitableTimerEx')
                arm_end = self.now()
                state = self.lib.WaitForMultipleObjects(2, handles, False, INFINITE)
                wake = self.now()
                arms.append({'arm_start_qpc': arm_start, 'arm_end_qpc': arm_end, 'relative_due_100ns': due_100ns,
                             'wake_qpc': wake, 'wait_return': state})
                if state == WAIT_OBJECT_0:
                    return {'status': 'stopped', 'wait_begin_qpc': begin, 'wait_end_qpc': wake, 'arms': arms}
                if state != WAIT_OBJECT_0 + 1:
                    raise OSError(ct.get_last_error(), 'timer wait failed: ' + str(state))
                # Rare early wake is rearmed; no busy-wait spin.
        finally:
            with self.state_lock: self.waiting = False

    def close(self):
        with self.state_lock:
            if self.closed: return
            if self.waiting: raise RuntimeError('cannot close handles while worker is waiting')
            if self.timer_handle and not self.lib.CancelWaitableTimer(self.timer_handle):
                self.close_errors.append({'api': 'CancelWaitableTimer', 'error': ct.get_last_error()})
            for name in ('timer_handle', 'stop_handle'):
                handle = getattr(self, name)
                if handle and not self.lib.CloseHandle(handle):
                    self.close_errors.append({'api': 'CloseHandle:' + name, 'error': ct.get_last_error()})
                setattr(self, name, None)
            self.closed = True
            if self.close_errors: raise OSError('timer handle cleanup failed: ' + repr(self.close_errors))


class NVMLSMReader:
    """Read-only device identity and one SM clock call per sample; no nvml setters."""
    def __init__(self, device_index: int = 0):
        if os.name != 'nt': raise OSError('NVML diagnostic is Windows only')
        if type(device_index) is not int or device_index < 0: raise ValueError('device index invalid')
        self.path = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/nvml.dll'
        self.lib = ct.CDLL(str(self.path)); self.closed = False; self.initialized = False
        self.lib.nvmlInit_v2.argtypes = []; self.lib.nvmlInit_v2.restype = ct.c_int
        self.lib.nvmlShutdown.argtypes = []; self.lib.nvmlShutdown.restype = ct.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ct.c_uint, ct.POINTER(ct.c_void_p)]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ct.c_int
        self.lib.nvmlDeviceGetClockInfo.argtypes = [ct.c_void_p, ct.c_uint, ct.POINTER(ct.c_uint)]
        self.lib.nvmlDeviceGetClockInfo.restype = ct.c_int
        self._check(self.lib.nvmlInit_v2(), 'nvmlInit_v2'); self.initialized = True
        try:
            self.handle = ct.c_void_p()
            self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(device_index, ct.byref(self.handle)), 'nvmlDeviceGetHandleByIndex_v2')
            self.identity = {'device_index': device_index, 'uuid': self._string('nvmlDeviceGetUUID', True),
                'name': self._string('nvmlDeviceGetName', True), 'driver_version': self._string('nvmlSystemGetDriverVersion', False),
                'nvml_library': file_ref(self.path), 'clock_writes_performed': False}
        except BaseException:
            self.close(); raise

    @staticmethod
    def _check(code, name):
        if code != 0: raise RuntimeError(name + ' NVML status ' + str(code))

    def _string(self, name, device):
        function = getattr(self.lib, name)
        function.argtypes = ([ct.c_void_p] if device else []) + [ct.c_char_p, ct.c_uint]
        function.restype = ct.c_int; value = ct.create_string_buffer(128)
        args = ([self.handle] if device else []) + [value, 128]
        self._check(function(*args), name)
        return value.value.decode('utf-8')

    def read_sm(self):
        if self.closed: raise RuntimeError('read after NVML shutdown')
        value = ct.c_uint()
        status = self.lib.nvmlDeviceGetClockInfo(self.handle, 1, ct.byref(value))
        return {'status': status, 'value': value.value if status == 0 else None, 'unit': 'MHz'}

    def close(self):
        if not self.closed:
            self.closed = True
            if self.initialized:
                self._check(self.lib.nvmlShutdown(), 'nvmlShutdown')
                self.initialized = False


def file_ref(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(); before = path.stat(); count = 0; digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1 << 20): digest.update(block); count += len(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or count != before.st_size:
        raise ValueError('identity input changed during read')
    return {'path': str(path), 'bytes': count, 'sha256': digest.hexdigest()}


def sample_loop(timer, reader, *, duration_ns: int, period_ns: int = 5_000_000,
                ready: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Portable algorithm used by deterministic host tests and the Win32 worker."""
    require_int(duration_ns, 'duration nanoseconds'); require_int(period_ns, 'period nanoseconds')
    origin = timer.now(); grid = DeadlineGrid(origin, timer.frequency, period_ns)
    end_target = origin + ceil_div(duration_ns * timer.frequency, NS_PER_SECOND)
    index, samples, missed, waits = 0, [], [], []
    ready()
    while grid.deadline(index) < end_target:
        deadline = grid.deadline(index)
        wait = timer.wait_until(deadline); wait['deadline_qpc'] = deadline; waits.append(wait)
        if wait['status'] == 'stopped': break
        if wait['status'] != 'deadline_reached': raise RuntimeError('unknown timer wait status')
        sample_begin = timer.now()
        if sample_begin >= end_target:
            missed.append({'first_index': index, 'next_index': ceil_div(duration_ns, period_ns), 'reason': 'wake_after_measurement_window'})
            break
        sm_begin = timer.now()
        try: reading = reader.read_sm()
        except Exception as error: reading = {'status': 'exception', 'value': None, 'error': type(error).__name__ + ': ' + str(error), 'unit': 'MHz'}
        sm_end = timer.now(); sample_end = timer.now()
        samples.append({'index': index, 'deadline_qpc': deadline, 'sample_begin_qpc': sample_begin,
            'sm_read_begin_qpc': sm_begin, 'sm_read_end_qpc': sm_end, 'sample_end_qpc': sample_end,
            'sm_mhz': reading, 'wake_lateness_qpc': sample_begin - deadline})
        next_index = grid.next_future_index(sample_end, index)
        if next_index != index + 1:
            missed.append({'first_index': index + 1, 'next_index': min(next_index, ceil_div(duration_ns, period_ns)),
                           'reason': 'sampling_or_wake_overrun_no_catchup_burst'})
        index = next_index
    return {'schema': 'precise-sm-telemetry/v1', 'qpc_frequency': timer.frequency, 'period_ns': period_ns,
        'duration_ns': duration_ns, 'origin_qpc': origin, 'window_end_qpc': end_target,
        'finished_qpc': timer.now(), 'requested_slots': ceil_div(duration_ns, period_ns),
        'samples': samples, 'waits': waits, 'missed_deadlines': missed,
        'timer_policy': dict(timer.metadata), 'identity': dict(reader.identity),
        'inference_launched': False, 'clock_written': False, 'cost_calibration_eligible': False}


class SamplerSession:
    """Non-daemon worker owns NVML; resources are never closed after a timed-out join."""
    def __init__(self, *, timer, reader_factory, duration_ns=2_000_000_000, period_ns=5_000_000):
        self.timer = timer; self.reader_factory = reader_factory
        self.duration_ns = duration_ns; self.period_ns = period_ns
        self.ready = threading.Event(); self.done = threading.Event()
        self.result = None; self.errors = []; self.reader_closed = False; self.handles_closed = False
        self.thread = threading.Thread(target=self._worker, name='precision-sm-sampler', daemon=False)

    def _worker(self):
        reader = None
        try:
            reader = self.reader_factory()
            self.result = sample_loop(self.timer, reader, duration_ns=self.duration_ns, period_ns=self.period_ns, ready=self.ready.set)
        except BaseException as error:
            self.errors.append({'phase': 'sampling', 'error': type(error).__name__ + ': ' + str(error)})
        finally:
            # This is the only thread that may close NVML; every read has returned.
            if reader is not None:
                try: reader.close(); self.reader_closed = True
                except BaseException as error: self.errors.append({'phase': 'NVML_shutdown', 'error': str(error)})
            self.done.set(); self.ready.set()

    def start(self):
        self.thread.start(); return self

    def stop(self):
        self.timer.signal_stop()

    def join(self, timeout_seconds=None):
        self.thread.join(timeout_seconds)
        if self.thread.is_alive():
            raise TimeoutError('sampler still active; NVML and timer handles remain owned, no completed receipt')
        return self.result

    def close(self):
        if self.thread.is_alive():
            raise RuntimeError('sampler must exit before closing timer handles')
        if not self.handles_closed:
            self.timer.close(); self.handles_closed = True

    def lifecycle(self):
        return {'thread_alive': self.thread.is_alive(), 'thread_exited': self.done.is_set(),
                'NVML_shutdown_after_reads': self.reader_closed, 'timer_handles_closed_after_join': self.handles_closed,
                'errors': list(self.errors), 'forced_termination': False}
