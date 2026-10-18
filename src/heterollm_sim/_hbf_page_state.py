"""Typed loops for the unchanged full-page HBF metadata state machine.

The controller enables these helpers only when this file is compiled and its
ordinary dictionaries, queues and methods are present. Source-mode simulation
keeps the original loops; it does not add a Python helper call for every page.
No NAND command, completion dependency, journal entry or page object is omitted.
"""
from __future__ import annotations

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
def _remember_page(pages: dict, transactions: list, key: object,
                   page_type: object) -> cython.void:
    token: dict
    saved_pages: dict
    page: object
    state: object
    saved: object
    ready: object
    for token in transactions:
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
    return locations, addresses, arrivals, old_ready


def commit_full_write(controller, buffer_id, generation,
                      first: cython.Py_ssize_t, locations: list,
                      old_ready: list, completions: list, time_ns,
                      page_type, missing):
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
    completion: object = time_ns
    for position in range(count):
        key = (buffer_id, generation, first + position)
        physical = locations[position]
        ready, time = old_ready[position], completions[position]
        _remember_page(pages, transactions, key, page_type)
        page = pages.get(key)
        if page is None:
            page = page_type([(0, PAGE)], [], None, physical, time)
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
            page.valid = [(0, PAGE)]
            page.dirty = []
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


def initialize_read_state(controller, owner, indexes: dict, state):
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
    for index in indexes:
        page = pages[(*owner, index)]
        address = -1
        if page.flash is not None:
            block, number = page.flash
            bank, lane = divmod(block, lanes)
            address = (bank * pages_per_block * lanes + number * lanes + lane) * PAGE
        full = page.valid == [(0, PAGE)]
        addresses[index] = address if full and not page.dirty and page.slot is None else -1
        valid[index] = full
        if page._read_state is not state:
            ready[index] = page.ready_ns
            page._read_state, page._read_index = state, index
