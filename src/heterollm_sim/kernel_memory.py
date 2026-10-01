"""Resolve explicit L2 accesses at DES dispatch, not at graph construction.

The cache state belongs to the event-kernel instance. Memory phases are ordered by the L2 service resource; compute tails may
overlap. Fills become reusable only after the preceding memory phase ends. Unnamed traffic gets unique buffers.
"""
from dataclasses import asdict, replace
import math

from .cache_state import CacheAccess, ExplicitCacheState
from .contracts import ResourceDemand
from .cost_models import HBMProfile


def resolve_l2_task(task, states):
    contract = task.metadata.get('stateful_l2')
    if contract is None:
        return task
    owner = contract['owner']
    signature = (contract['capacity_bytes'], contract['line_bytes'], contract['write_back'], contract['write_allocate'])
    existing = states.get(owner)
    if existing is not None and existing[0] != signature:
        raise ValueError('L2 configuration changed within one simulation')
    if existing is None:
        cache = ExplicitCacheState(signature[0], signature[1], write_back=signature[2], write_allocate=signature[3])
    else:
        cache = existing[1]
    accesses = tuple(CacheAccess(**{**item, 'buffer_id':
                        task.task_id + item['buffer_id'][10:] if item['buffer_id'].startswith('@tasklocal') else item['buffer_id']})
                     for item in contract['accesses'])
    hbm = HBMProfile(**contract['hbm'])
    results = cache.access_many(accesses)
    states[owner] = (signature, cache)
    reads = sum(r.backing_read_bytes for r in results)
    writes = sum(r.backing_write_bytes for r in results)
    requested = sum(a.size_bytes for a in accesses)
    memory_ns = hbm.memory_service(reads, writes, bandwidth_gb_s=contract['bandwidth_gb_s'])['service_ns']
    cache_ns = max(requested / contract['cache_bandwidth_gb_s'],
                   math.ceil(sum(r.touched_lines for r in results) / contract['cache_parallelism']) * contract['cache_latency_ns'])
    demands = []
    for d in task.demands:
        if d.resource_id == contract['memory_resource']:
            d = replace(d, service_ns=memory_ns, bytes_moved=reads + writes, energy_pj=(reads + writes) * hbm.energy_pj_per_byte)
        elif d.resource_id == contract['cache_resource']:
            d = replace(d, service_ns=cache_ns, bytes_moved=requested)
        demands.append(d)
    # Measured device wall is invalid under a different cache protocol. The
    # planner prevents combining that surface with mutable L2 for now.
    duration = max((d.service_ns for d in demands if d.resource_id not in contract['envelope_resources']), default=0)
    duration += contract.get('dependency_ns', 0)
    memory_completion_ns = max(memory_ns, cache_ns)
    demands = tuple(replace(d, service_ns=(memory_completion_ns if d.resource_id == contract.get('order_resource') else duration))
                    if d.resource_id in contract['envelope_resources'] else d for d in demands)
    report = {'model': 'line_lru_dispatch_order', 'hit_lines': sum(r.hit_lines for r in results),
              'miss_lines': sum(r.miss_lines for r in results), 'hbm_read_bytes': reads, 'hbm_write_bytes': writes,
              'dirty_eviction_bytes': sum(r.dirty_eviction_bytes for r in results),
              'accesses': tuple(asdict(a) for a in accesses),
              'cache_state_after': asdict(cache.snapshot()),
              'concurrency_policy': 'memory_phase_order_compute_tail_overlap',
              'memory_completion_offset_ns': memory_completion_ns,
              'prediction_confidence': 'low', 'validated': False}
    prediction = task.metadata.get('kernel_prediction', {})
    if prediction:
        prediction = {**prediction, 'prediction': {**prediction.get('prediction', {}),
                      'prediction_ns': duration, 'model': 'analytical', 'reason': 'stateful_l2_runtime_recost'}}
    return replace(task, demands=demands, metadata={**task.metadata, **({'kernel_prediction': prediction} if prediction else {}), 'l2_execution': report,
                   'analytical_service_ns': duration, 'analytical_bytes': sum(d.bytes_moved for d in demands)})


def attach_l2_contract(task, *, gpu, hbm, memory_resource, cache_resource, owner, accesses):
    """Validate and attach a dynamic cost contract; no mutable cache state here."""
    level = gpu.cache_hierarchy.levels[-1]
    if level.capacity_bytes % level.line_bytes:
        raise ValueError('stateful L2 capacity must be a multiple of line size')
    accesses = tuple(accesses)
    if not accesses:
        return task
    for access in accesses:
        if not isinstance(access, CacheAccess):
            raise ValueError('explicit CacheAccess required')
    model = task.metadata.get('phase_metadata', {}).get('kernel_model', task.metadata.get('cost_model', {}).get('kernel_model', {}))
    if model.get('prediction', {}).get('model', 'analytical') != 'analytical':
        raise ValueError('measured kernel cache protocol cannot be combined with stateful L2')
    bandwidth = model.get('achieved_bandwidth_gb_s', hbm.effective_bandwidth_gb_s)
    if memory_resource not in {d.resource_id for d in task.demands}:
        raise ValueError('stateful L2 requires an existing backing-memory demand')
    envelope = tuple(d.resource_id for d in task.demands if d.resource_id.endswith('.device_stream'))
    # The shared order resource lasts through memory completion, not compute.
    # This retains deterministic fill visibility while allowing compute tails
    # to overlap other kernels' memory service (not cycle-level concurrent fills).
    order = owner + '.access_order'
    duration = max((d.service_ns for d in task.demands), default=0)
    demands = tuple(d for d in task.demands if d.resource_id != order)
    if cache_resource not in {d.resource_id for d in demands}:
        demands += (ResourceDemand(cache_resource, 0),)
    demands += (ResourceDemand(order, duration),)
    return replace(task, demands=demands, metadata={**task.metadata, 'stateful_l2': {
        'owner': owner, 'capacity_bytes': level.capacity_bytes, 'line_bytes': level.line_bytes,
        'write_back': gpu.cache_hierarchy.write_back, 'write_allocate': gpu.cache_hierarchy.write_allocate,
        'cache_resource': cache_resource, 'memory_resource': memory_resource,
        'cache_bandwidth_gb_s': level.bandwidth_gb_s, 'cache_latency_ns': level.hit_latency_ns,
        'cache_parallelism': level.transaction_parallelism, 'bandwidth_gb_s': bandwidth,
        'hbm': asdict(hbm), 'accesses': tuple(asdict(a) for a in accesses),
        'invocation_buffers': tuple(a.buffer_id for a in accesses if a.buffer_id.startswith('@invocation:')),
        'envelope_resources': (*envelope, order), 'order_resource': order,
        'dependency_ns': model.get('pipeline_ns', {}).get('dependency', 0),
    }})

def paged_buffer_accesses(*, buffer_id, page_ids, tokens_per_page, bytes_per_token,
                          first_token, token_count, operation='read', allocation_generation=0):
    """Translate logical KV token range to physical pages; aliases stay aliases.

    Callers must supply the allocator's page table. Generation prevents a freed
    allocation from hitting stale lines after its page IDs are recycled.
    """
    for name, value, minimum in (
        ('tokens_per_page', tokens_per_page, 1), ('bytes_per_token', bytes_per_token, 1),
        ('first_token', first_token, 0), ('token_count', token_count, 0),
        ('allocation_generation', allocation_generation, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(name + ' has invalid integer value')
    if not isinstance(buffer_id, str) or not buffer_id or operation not in ('read','write'):
        raise ValueError('invalid buffer identity or access operation')
    if any(type(page) is not int or page < 0 for page in page_ids):
        raise ValueError('physical page IDs must be nonnegative integers')
    end = first_token + token_count
    if end > len(page_ids) * tokens_per_page:
        raise ValueError('KV token range exceeds physical page table')
    result = []
    while first_token < end:
        logical_page, offset = divmod(first_token, tokens_per_page)
        count = min(end-first_token, tokens_per_page-offset)
        result.append(CacheAccess(f'{buffer_id}:generation:{allocation_generation}:page:{page_ids[logical_page]}',
                                  offset*bytes_per_token, count*bytes_per_token, operation,
                                  tokens_per_page*bytes_per_token))
        first_token += count
    return tuple(result)


def attention_stream_k_accesses(audit, invocation_id, *, fixup=False):
    """Bind uniform tiles, including tail queries, to CUDA scratch addresses.

    Tile order is query tile first, then GQA head tile and KV head, matching
    fattn-common.cuh. Partial result writes include padding; fixup reads do not.
    IDs are invocation-local, not request-persistent allocations.
    """
    schedule = audit.get('stream_k')
    traffic = audit.get('scratch_traffic')
    if not schedule or not traffic or not schedule['fixup_required']:
        return None
    blocks = schedule['main_blocks']
    tiles = schedule['output_tiles']
    parts = schedule['blocks_per_tile']
    allocation = traffic['allocation_bytes']
    columns, remainder = divmod(allocation, blocks * (16 + 4 * 64))
    query_tokens = schedule['query_tokens']
    query_tile = schedule['query_tile']
    heads_per_tile = schedule['heads_per_tile']
    query_tiles = math.ceil(query_tokens / query_tile)
    if (remainder or columns != query_tile * heads_per_tile
            or tiles % query_tiles
            or traffic['output_bytes'] != (tiles // query_tiles) * query_tokens * heads_per_tile * 256):
        raise ValueError('stream-K tile layout differs from traffic geometry')
    scratch = '@invocation:' + invocation_id + ':scratch'
    output = '@invocation:' + invocation_id + ':output'
    operation = 'read' if fixup else 'write'
    result = []
    if not fixup and audit['read_bytes']:
        result.append(CacheAccess('@tasklocal:attention_inputs', 0, audit['read_bytes'], 'read'))
    for tile in range(tiles):
        first = tile * parts
        last = first + parts - 1
        # Last block owns final output and bank-0 metadata; earlier partials
        # occupy bank 1 and the data region after both metadata banks.
        live_queries = min(query_tile, query_tokens - (tile % query_tiles) * query_tile)
        live_columns = live_queries * heads_per_tile if fixup else columns
        result.append(CacheAccess(scratch, last * columns * 8, live_columns * 8,
                                  operation, allocation))
        # Each partial block has its own padded stride; a short final tile
        # cannot be represented as one packed range spanning those blocks.
        for block in range(first, last):
            result.append(CacheAccess(scratch, (blocks + block) * columns * 8,
                                      live_columns * 8, operation, allocation))
            result.append(CacheAccess(scratch, blocks * columns * 16 + block * columns * 256,
                                      live_columns * 256, operation, allocation))
    size = traffic['output_bytes']
    if fixup:
        result.append(CacheAccess(output, 0, size, 'read', size))
    result.append(CacheAccess(output, 0, size, 'write', size))
    for op, key in (('read', 'read_bytes'), ('write', 'write_bytes')):
        if sum(a.size_bytes for a in result if a.operation == op) != audit[key]:
            raise ValueError('stream-K physical ranges differ from stage traffic')
    return tuple(result)


def bind_l2_invocation(metadata, namespace):
    """Rebind only declared compiler temporaries, never weights or KV pages.

    All tasks in one online replay receive the same namespace. Clone only the
    mutated contract so cached templates remain reusable and read-only.
    """
    contract = metadata.get('stateful_l2')
    if not contract or not contract.get('invocation_buffers') or namespace is None:
        return metadata
    buffers = set(contract['invocation_buffers'])
    accesses = tuple({**access, 'buffer_id': namespace + ':' + access['buffer_id']}
                     if access['buffer_id'] in buffers else access
                     for access in contract['accesses'])
    return {**metadata, 'stateful_l2': {**contract, 'accesses': accesses,
            'invocation_buffers': (), 'invocation_namespace': namespace}}
