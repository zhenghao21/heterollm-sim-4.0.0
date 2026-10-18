"""Typed loops for the unchanged full-page HBF metadata state machine.

The controller enables these helpers only when this file is compiled and its
ordinary dictionaries, queues and methods are present. Source-mode simulation
keeps the original loops; it does not add a Python helper call for every page.
No NAND command, completion dependency, journal entry or page object is omitted.
"""
from __future__ import annotations

import numpy as np

try:
    import cython
except ImportError:
    class _PythonMode:
        compiled = False

        @staticmethod
        def cfunc(function):
            return function

        @staticmethod
        def cast(_type, value):
            return value

        inline = cfunc

    cython = _PythonMode()


COMPILED = cython.compiled
PAGE = 4096


@cython.cfunc
@cython.inline
def _remember_set(mapping: dict, journals: list, key: object,
                  value: object, missing: object) -> cython.void:
    """The insertion-only _JournalDict setter, including every outer journal."""
    token: dict
    values: dict
    for token in journals:
        values = token["values"]
        if key not in values:
            values[key] = mapping.get(key, missing)
    if key not in mapping:
        owner: object = cython.cast(object, mapping)
        positions: object = owner._positions
        if positions is not None:
            positions[key] = owner._next_position
            owner._next_position += 1
    # A typed dictionary assignment calls PyDict_SetItem directly. The
    # ordinary setter's journal work was performed above in the same order.
    dict.__setitem__(mapping, key, value)


@cython.cfunc
@cython.inline
def _remember_one_page(pages: dict, token: dict, key: object,
                       page_type: object) -> cython.void:
    saved_pages: dict
    page: object
    state: object
    saved: object
    ready: object
    saved_pages = token["pages"]
    if key not in saved_pages:
        page = pages.get(key)
        if page is None:
            saved_pages[key] = None
        else:
            state = page._read_state
            saved = None if state is None else token["read_ready"].get(id(state))
            ready = (float(saved[1][page._read_index])
                     if saved is not None and page._read_index < len(saved[1])
                     else page.ready_ns)
            ready = token["page_ready"].pop(key, ready)
            saved_pages[key] = page_type(page.valid.copy(), page.dirty.copy(),
                                        page.slot, page.flash, ready)


@cython.cfunc
@cython.inline
def _remember_page(pages: dict, transactions: list, key: object,
                   page_type: object) -> cython.void:
    token: dict
    for token in transactions:
        _remember_one_page(pages, token, key, page_type)


@cython.cfunc
@cython.inline
def _remember_owners(owners: dict, transactions: list, block: object,
                     gc_dirty: object) -> cython.void:
    token: dict
    saved: dict
    previous: object
    if gc_dirty is not None:
        gc_dirty.add(block)
    for token in transactions:
        saved = token["owners"]
        if block not in saved:
            previous = owners.get(block)
            saved[block] = None if previous is None else previous.copy()


@cython.cfunc
@cython.inline
def _remember_slots(transactions: list, slots: object) -> cython.void:
    token: dict
    state: dict
    for token in transactions:
        state = token["state"]
        if "_slots" not in state:
            state["_slots"] = slots.copy()


def count_missing_pages(pages: dict, buffer_id, generation,
                        first: cython.Py_ssize_t, stop: cython.Py_ssize_t):
    index: cython.Py_ssize_t
    count: cython.Py_ssize_t = 0
    for index in range(first, stop):
        count += (buffer_id, generation, index) not in pages
    return count


def prepare_full_write(controller, buffer_id, generation,
                       first: cython.Py_ssize_t, stop: cython.Py_ssize_t,
                       time_ns, missing):
    """Allocate exactly the original round-robin, collection-free page run.

    The caller already checked the complete run's free capacity and owns its
    rollback transaction. A block's page zero sees its old block dependency;
    later pages use their original independent arrivals.
    """
    pages: dict = controller._pages
    used: dict = cython.cast(dict, controller._used)
    used_journals: list = controller._used._journals
    active = controller._active
    free = controller._free_blocks
    block_ready: dict = cython.cast(dict, controller._block_ready)
    lanes = controller._lanes
    pages_per_block = controller.config.pages_per_block
    locations: list = []
    addresses: list = []
    arrivals: list = []
    old_ready: list = []
    keys: list = []
    all_new: cython.bint = True
    index: cython.Py_ssize_t
    key: object
    page: object
    ready: object
    block: object
    number: object
    arrival: object
    bank: object
    lane: object
    for index in range(first, stop):
        key = (buffer_id, generation, index)
        page = pages.get(key)
        keys.append(key)
        all_new = all_new and page is None
        ready = max(time_ns, page.ready_ns if page is not None else 0.0)
        while len(active) < lanes and free:
            block = free.popleft()
            _remember_set(used, used_journals, block, 0, missing)
            active.append(block)
        # The caller's available-page check proves an active block exists.
        block = active.popleft()
        number = used[block]
        _remember_set(used, used_journals, block, number + 1, missing)
        if used[block] < pages_per_block:
            active.append(block)
        arrival = max(ready, block_ready.get(block, 0.0)) if number == 0 else ready
        locations.append((block, number))
        bank, lane = divmod(block, lanes)
        addresses.append((bank * pages_per_block * lanes + number * lanes + lane) * PAGE)
        arrivals.append(arrival)
        old_ready.append(ready)
    return locations, addresses, arrivals, old_ready, keys, bool(all_new)


def commit_full_write(controller, buffer_id, generation,
                      first: cython.Py_ssize_t, locations: list,
                      old_ready: list, completions: list, time_ns,
                      page_type, missing, full_ranges, empty_ranges,
                      keys: list,
                      page_transactions: list, partial_page_transactions: list):
    """Publish the original page objects, mappings and first-write journals."""
    pages: dict = controller._pages
    owners: dict = controller._owners
    indexes: dict = controller._buffer_pages.setdefault((buffer_id, generation), {})
    transactions: list = controller._transaction_stack
    gc_dirty: object = controller._gc_dirty
    block_ready: dict = cython.cast(dict, controller._block_ready)
    block_journals: list = controller._block_ready._journals
    slot_ready: dict = cython.cast(dict, controller._slot_ready)
    slot_journals: list = controller._slot_ready._journals
    slots = controller._slots
    resident = controller._resident
    read_state = controller._read_states.get((buffer_id, generation))
    position: cython.Py_ssize_t
    count: cython.Py_ssize_t = min(len(locations), len(old_ready), len(completions))
    key: object
    page: object
    physical: object
    ready: object
    time: object
    previous: object
    token: dict
    lower: object
    upper: object
    page_index: cython.Py_ssize_t
    completion: object = time_ns
    for position in range(count):
        page_index = first + position
        key = keys[position]
        physical = locations[position]
        ready, time = old_ready[position], completions[position]
        _remember_page(pages, page_transactions, key, page_type)
        for token, lower, upper in partial_page_transactions:
            if page_index < lower or page_index >= upper:
                _remember_one_page(pages, token, key, page_type)
        page = pages.get(key)
        if page is None:
            page = page_type(full_ranges, empty_ranges, None, physical, time)
            pages[key] = page
            indexes[first + position] = None
        else:
            if page.flash is not None:
                _remember_owners(owners, transactions, page.flash[0], gc_dirty)
                owners.get(page.flash[0], {}).pop(page.flash[1], None)
            if page.slot is not None:
                _remember_set(slot_ready, slot_journals, page.slot, ready, missing)
                _remember_slots(transactions, slots)
                slots.append(page.slot)
                resident.pop(key)
            page.flash = physical
            page.slot = None
            page._valid = full_ranges
            page._dirty = empty_ranges
            page.ready_ns = time
        _remember_owners(owners, transactions, physical[0], gc_dirty)
        owners.setdefault(physical[0], {})[physical[1]] = key
        if read_state is not None:
            controller._refresh_read_state(key, page)
            read_state = controller._read_states.get((buffer_id, generation))
        previous = block_ready.get(physical[0], 0.0)
        _remember_set(block_ready, block_journals, physical[0], max(time, previous), missing)
        completion = max(completion, time)
    return completion


@cython.cfunc
@cython.inline
def _read_array(values: object, dtype: object) -> cython.bint:
    return (type(values) is np.ndarray and values.ndim == 1
            and values.dtype == dtype and values.flags.c_contiguous
            and values.flags.writeable and values.flags.aligned)


@cython.cfunc
@cython.inline
def _plain_ranges(ranges: object, full_ranges: object,
                  empty_ranges: object) -> cython.bint:
    """Whether equality with the one full range cannot execute author code."""
    return (ranges is full_ranges or ranges is empty_ranges
            or (type(ranges) is list
                and (len(ranges) != 1
                     or (type(ranges[0]) is tuple and len(ranges[0]) == 2
                         and type(ranges[0][0]) is int
                         and type(ranges[0][1]) is int))))


@cython.cfunc
@cython.inline
def _plain_read_keys(owners: dict, indexes: dict,
                     size: cython.Py_ssize_t) -> cython.bint:
    """A lookup cannot collide with an authored owner or page-index hook."""
    key: object
    for key in owners:
        if not (type(key) is tuple and len(key) == 2
                and type(key[0]) is str and type(key[1]) is int):
            return False
    for key in indexes:
        if not (type(key) is int and 0 <= key < size):
            return False
    return True


def initialize_read_state(controller, owner, indexes: dict, state,
                          full_ranges, empty_ranges, page_type, state_type):
    """Bind existing page objects to one fresh dense read state, in order."""
    pages: dict = controller._pages
    addresses = state.addresses
    ready = state.ready
    valid = state.valid
    lanes = controller._lanes
    pages_per_block = controller.config.pages_per_block
    index: object
    page: object
    full: object
    address: object
    block: object
    number: object
    bank: object
    lane: object
    ranges: object
    dirty: object
    clean_address: object
    page_state: object
    time: object
    offset: cython.Py_ssize_t
    size: cython.Py_ssize_t = 0
    address_values: cython.longlong[::1] = None
    valid_values: cython.uchar[::1] = None
    ready_values: cython.double[::1] = None
    valid_bytes: object = None
    typed: cython.bint = (type(state) is state_type
        and type(lanes) is int and type(pages_per_block) is int
        and _read_array(addresses, np.dtype(np.int64))
        and _read_array(valid, np.dtype(np.bool_))
        and _read_array(ready, np.dtype(np.float64)))
    if typed:
        size = addresses.shape[0]
        typed = (valid.shape[0] == size and ready.shape[0] == size
                 and _plain_read_keys(controller._buffer_pages, indexes, size))
    if typed:
        address_values = addresses
        valid_bytes = valid.view(np.uint8)
        valid_values = valid_bytes
        ready_values = ready
    for index in indexes:
        page = pages[(*owner, index)]
        if typed and type(page) is not page_type:
            typed = False
            address_values = valid_values = ready_values = valid_bytes = None
        address = -1
        if page.flash is not None:
            if typed and not (type(page.flash) is tuple and len(page.flash) == 2
                              and type(page.flash[0]) is int
                              and type(page.flash[1]) is int):
                typed = False
                address_values = valid_values = ready_values = valid_bytes = None
            block, number = page.flash
            bank, lane = divmod(block, lanes)
            address = (bank * pages_per_block * lanes + number * lanes + lane) * PAGE
        if type(page) is page_type:
            # The caller verified these exact Page property descriptors. A
            # million clean pages seeded with explicit lists need the same
            # equality/truth operations, not two Python getter calls each.
            ranges = page._valid
            if typed and not _plain_ranges(ranges, full_ranges, empty_ranges):
                typed = False
                address_values = valid_values = ready_values = valid_bytes = None
            full = ranges is full_ranges or ranges == [(0, PAGE)]
            clean_address = -1
            if full:
                dirty = page._dirty
                if typed and not (dirty is empty_ranges or type(dirty) is list):
                    typed = False
                    address_values = valid_values = ready_values = valid_bytes = None
                if (dirty is empty_ranges or not dirty) and page.slot is None:
                    clean_address = address
        else:
            # An authored Page subclass can replace either property. Do not
            # inspect its backing representation or bypass those callbacks.
            full = page.valid == [(0, PAGE)]
            clean_address = address if full and not page.dirty and page.slot is None else -1
        if typed and not -(1 << 63) <= clean_address < (1 << 63):
            typed = False
            address_values = valid_values = ready_values = valid_bytes = None
        if typed:
            offset = index
            address_values[offset] = clean_address
            valid_values[offset] = full
        else:
            addresses[index] = clean_address
            valid[index] = full
        page_state = page._read_state
        if page_state is not state:
            # A former binding can contain an authored array/index/float
            # conversion. Release exported buffers before invoking it, just
            # as for custom range equality or dirty truth above.
            if typed and page_state is not None:
                typed = False
                address_values = valid_values = ready_values = valid_bytes = None
            time = page.ready_ns
            if typed and type(time) is not float:
                typed = False
                address_values = valid_values = ready_values = valid_bytes = None
            if typed:
                ready_values[offset] = time
            else:
                ready[index] = time
            page._read_state, page._read_index = state, index
