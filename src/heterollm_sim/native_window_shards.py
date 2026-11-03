"""One point's stationary windows shared by several processes.

Every window runs on its own copy of the unchanged cohort boundary, with
arguments that depend on no other window, and each process builds that boundary
through the same deterministic setup. So all processes go through the window
list running the windows nobody has claimed yet and saving each run. The
primary process then goes through the list once more and takes the saved runs
in the usual order: all that follows a run (the records, the re-sampling check,
the samples) is the one-process code, whichever process ran it.

The primary runs a window itself when its owner died or failed, in the window's
place in the list; a window's error is then raised exactly where one process
raises it. A run that proves the store reservation too small makes the windows
after it useless, as the list stops there or earlier: they are cancelled and
the next pass starts from the reservation the primary publishes.

Files of a pass, under ``pass<k>/``: ``reservation.pickle``, the store share
its windows run with; ``<pid>.owner``, a process's name, hard-linked as
``<position>.claim`` so that a claim is atomic and names its owner;
``<position>.record``, a saved run; ``<position>.cancel``, polled by the window
running there; ``<position>.limit``, no window after that position is needed.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import pickle
import time
import traceback

from .execution_control import ExecutionCancelledError, NativeFastWallBudgetError, WindowCancelledError

_POLL_SECONDS = .2


class WindowShards:
    """This process's side of a point's shared windows.

    ``alive(pid)`` tells the primary whether a window's owner still runs. A
    helper never returns from ``sample``: it goes from pass to pass until
    its control stops it or its primary ends the process.
    """

    def __init__(self, directory, *, alive=None, helper=False):
        self.directory = Path(directory)
        self.alive = alive
        self.helper = helper
        self.executing = False
        self.passes = self.position = self.lost = 0
        self.window_seconds = 0.

    def sample(self, prepared, control, sample, records):
        """The samples ``sample(records)`` builds; it walks the window list once."""
        started = time.perf_counter()
        while True:
            self._begin_pass(prepared, control)
            sample([])
            if not self.helper:
                break
        self.executing, self.position = False, 0
        return {**sample(records), "wall_seconds": time.perf_counter() - started}

    def window(self, sample_id, control, execute, exceeded):
        """The next window's run; None while the list is only being executed."""
        position, self.position = self.position, self.position + 1
        if self.executing:
            self._execute(position, control, execute, exceeded)
            return None
        return self._collect(position, sample_id, control, execute)

    def _begin_pass(self, prepared, control):
        if self.passes and not self.helper:
            # The list stopped early: nothing of the last pass is needed now.
            self._stop_after(-1)
        self.passes += 1
        self.position, self.executing = 0, True
        reservation = self._path("reservation", "pickle")
        if self.helper:
            while not reservation.exists():
                control.raise_if_cancelled()
                time.sleep(_POLL_SECONDS)
            prepared["page_staging_occupancy"] = pickle.loads(reservation.read_bytes())
        else:
            reservation.parent.mkdir(parents=True)
            self._publish(reservation, prepared.get("page_staging_occupancy"))
        self._path(os.getpid(), "owner").write_text(str(os.getpid()), encoding="ascii")

    def _execute(self, position, control, execute, exceeded):
        if self._limit() < position or not self._claim(position):
            return
        claim = self._path(position, "claim")
        # A limit set just before this claim did not see it to cancel it.
        if self._limit() < position:
            claim.unlink()
            return
        try:
            record = execute(replace(control, window_cancel_path=str(self._path(position, "cancel"))))
        except WindowCancelledError:
            claim.unlink()
            return
        except (ExecutionCancelledError, NativeFastWallBudgetError):
            raise
        except Exception:
            # The primary runs it again in its place, where its error counts.
            traceback.print_exc()
            claim.unlink()
            return
        self._publish(self._path(position, "record"), record)
        if exceeded(record):
            self._stop_after(position)

    def _collect(self, position, sample_id, control, execute):
        path = self._path(position, "record")
        while not path.exists():
            control.raise_if_cancelled()
            owner = self._owner(position)
            if owner is not None and self.alive(owner):
                time.sleep(_POLL_SECONDS)
                continue
            if owner is not None:
                self._path(position, "claim").unlink(missing_ok=True)
            if self._claim(position):
                self.lost += 1
                self._publish(path, execute(control))
        record = pickle.loads(path.read_bytes())
        if record["sample_id"] != sample_id:
            raise RuntimeError("native_window_shards_list_differs_between_processes")
        self.window_seconds += record["wall_seconds"]
        return record

    def _stop_after(self, position):
        # The limit first, the cancels after: a later claim checks the limit.
        self._path(position, "limit").touch()
        for claim in self._pass.glob("*.claim"):
            if int(claim.stem) > position:
                self._path(claim.stem, "cancel").touch()

    def _limit(self):
        return min((int(path.stem) for path in self._pass.glob("*.limit")), default=math.inf)

    def _claim(self, position):
        try:
            os.link(self._path(os.getpid(), "owner"), self._path(position, "claim"))
        except FileExistsError:
            return False
        return True

    def _owner(self, position):
        try:
            return int(self._path(position, "claim").read_text(encoding="ascii"))
        except FileNotFoundError:
            return None

    def _publish(self, path, value):
        partial = path.with_name(f"{path.name}.{os.getpid()}.part")
        partial.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(partial, path)

    @property
    def _pass(self):
        return self.directory / f"pass{self.passes}"

    def _path(self, name, suffix):
        return self._pass / f"{name}.{suffix}"
