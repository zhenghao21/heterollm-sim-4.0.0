"""One native completion POST for an explicitly simultaneous prompt cohort.

This changes the request submission policy. It does not reproduce independent
HTTP arrivals, and its client timestamps belong to a shared transport request.
The locked server tokenizes all prompts and enqueues their tasks under one lock.
"""
from __future__ import annotations
import json
import time
from urllib.request import Request, urlopen


def post_cohort_stream_json(url: str, payload: dict, parallel: int):
    if parallel < 1 or not isinstance(payload.get('prompt'), str):
        raise ValueError('cohort requires a positive parallel count and one text prompt')
    if int(payload.get('n', 1)) != 1 or int(payload.get('n_cmpl', 1)) != 1 or payload.get('id_slot', -1) != -1:
        raise ValueError('cohort does not allow child completions or a shared fixed slot')
    body = dict(payload, prompt=[payload['prompt']] * parallel, stream=True, n_cmpl=1)
    request = Request(url, data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'})
    groups = {i: [] for i in range(parallel)}
    control_events = []
    started = time.perf_counter()
    with urlopen(request, timeout=180) as response:
        for line in response:
            text = line.decode('utf-8').strip()
            if not text or text.startswith(':'):
                continue
            if not text.startswith('data:'):
                raise ValueError('cohort expected SSE data')
            text = text[5:].strip()
            if text == '[DONE]':
                break
            item = json.loads(text)
            received = time.perf_counter()
            # Locked server emits null for is_begin/header-only notifications.
            if item is None:
                control_events.append({'event': None, 'received_monotonic_s': received})
                continue
            if not isinstance(item, dict) or 'error' in item:
                raise ValueError(f'cohort server error: {item}')
            index = item.get('index')
            if isinstance(index, bool) or not isinstance(index, int) or index not in groups:
                raise ValueError(f'invalid cohort result index: {index}')
            groups[index].append((item, received))
    ended = time.perf_counter()
    result = []
    for index, events in groups.items():
        if not events or sum('timings' in item and bool(item.get('stop')) for item, _ in events) != 1:
            raise ValueError(f'missing or duplicate final timing for cohort index {index}')
        merged = {}
        for item, _ in events:
            merged.update(item)
        merged['content'] = ''.join(str(item.get('content') or '') for item, _ in events)
        merged['raw_stream_events'] = [item for item, _ in events]
        merged['shared_stream_control_events'] = control_events
        merged['raw_stream_event_times_monotonic_s'] = [t for _, t in events]
        token_times = [t for item, t in events if item.get('tokens')]
        content_times = [t for item, t in events if item.get('content')]
        selected = token_times or content_times
        first = selected[0] if selected else None
        last = selected[-1] if selected else None
        elapsed = lambda t: None if t is None else (t - started) * 1000.0
        boundary = {
            'mode': 'sse', 'submission_mode': 'atomic_prompt_list',
            'transport_scope': 'shared_post', 'cohort_index': index,
            'request_start': 'client_before_shared_http_post',
            'request_start_monotonic_s': started,
            'last_token_monotonic_s': last, 'stream_end_monotonic_s': ended,
            'first_event_ms': elapsed(events[0][1]),
            'first_content_ms': elapsed(content_times[0]) if content_times else None,
            'request_to_first_token_ms': elapsed(first),
            'request_to_last_token_ms': elapsed(last), 'request_to_end_ms': elapsed(last),
            'stream_end_ms': elapsed(ended),
            'stream_control_overhead_ms': (ended - last) * 1000.0 if last is not None else None,
            'token_chunk_times_ms': [elapsed(t) for t in token_times],
            'first_token_source': 'stream_first_token_ids' if token_times else 'stream_first_nonempty_content',
            'status': 'measured' if selected else 'unavailable', 'chunk_count': len(events),
        }
        result.append((merged, boundary))
    complete = all(b['last_token_monotonic_s'] is not None for _, b in result)
    makespan = (max(b['last_token_monotonic_s'] for _, b in result) - started) * 1000.0 if complete else None
    for _, boundary in result:
        boundary['batch_client_makespan_ms'] = makespan
        boundary['batch_client_wall_ms'] = (ended - started) * 1000.0
    return result
