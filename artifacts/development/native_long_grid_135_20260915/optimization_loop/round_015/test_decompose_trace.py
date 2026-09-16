"""Portable interval tests; never execute a GPU, native benchmark or simulator."""
import importlib.util
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('r15_decompose', Path(__file__).with_name('decompose_trace.py'))
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


@pytest.mark.parametrize('intervals,expected', [
    ([], []), ([(0, 10), (2, 4), (7, 15)], [(0, 15)]),
    ([(10, 20), (0, 10), (30, 30)], [(0, 20)]),
    ([(9, 12), (0, 3)], [(0, 3), (9, 12)]),
])
def test_union_merges_overlapping_touching_and_unsorted_intervals(intervals, expected):
    assert trace.intervals_union(intervals) == expected
    assert trace.union_ns(intervals) == sum(end - start for start, end in expected)


@pytest.mark.parametrize('intervals', [[(10, 9)], [(True, 10)], [(0., 10)], [(0, None)]])
def test_bad_intervals_fail_closed(intervals):
    with pytest.raises(ValueError):trace.intervals_union(intervals)


def test_gaps_clips_overlapping_events_and_preserves_boundary_gaps():
    assert trace.gaps([(-10, 3), (2, 6), (10, 14), (13, 16), (30, 40)], 0, 20) == [(6, 10), (16, 20)]
    assert trace.gaps([], 0, 20) == [(0, 20)]
    assert trace.gaps([(-5, 30)], 0, 20) == []


def test_host_and_gpu_union_is_not_the_sum():
    host = [(0, 5), (5, 25)]
    gpu = [(10, 15), (20, 30)]
    assert trace.union_ns(host) + trace.union_ns(gpu) == 40
    assert trace.union_ns([*host, *gpu]) == 30
    assert trace.intersection_ns(host, gpu) == 10


def test_sync_overlap_partition_closes_with_parallel_kernels():
    kernels = [(20, 40), (30, 50), (70, 80)]
    sync = [(10, 90)]
    prefix, tail = [(0, 20)], [(80, 100)]
    holes = trace.gaps(kernels, 20, 80)
    assert holes == [(50, 70)]
    assert trace.union_ns(kernels) == 40
    assert sum(trace.intersection_ns(sync, values) for values in (prefix, kernels, holes, tail)) == trace.union_ns(sync)
    assert 20 + trace.union_ns(kernels) + trace.union_ns(holes) + 20 == 100


def test_overlap_computation_deduplicates_each_side():
    assert trace.intersection_ns([(0, 10), (5, 15)], [(2, 8), (3, 14)]) == 12
    assert trace.intersection_ns([(0, 4)], [(4, 8)]) == 0


def test_process_part_of_kernel_api_correlation():
    pid = 0x1000000
    api = {'globalTid': pid | 17, 'correlationId': 7}
    kernel = {'globalPid': pid, 'correlationId': 7}
    wrong_process = {'globalPid': 0x2000000, 'correlationId': 7}
    assert trace.correlation_key(api, True) == trace.correlation_key(kernel)
    assert trace.correlation_key(api, True) != trace.correlation_key(wrong_process)


def test_source_rows_not_silently_dropped_or_changed():
    rows = [{'evidence_rowid': 1, 'start': 1, 'end': 3}]
    trace.verify_rows(rows, rows, 'test')
    with pytest.raises(ValueError):trace.verify_rows([], rows, 'test')
    with pytest.raises(ValueError):trace.verify_rows([dict(rows[0], end=4)], rows, 'test')
    with pytest.raises(ValueError):trace.verify_rows(rows * 2, rows, 'test')


def test_nearest_rank_p95_keeps_long_formal_calls():
    assert trace.stats([*range(1, 19), 100, 1000])['p95_nearest_rank_ns'] == 100
    assert trace.stats([*range(1, 19), 100, 1000])['maximum_ns'] == 1000


@pytest.mark.parametrize('existing,inside', [(True, False), (False, True)])
def test_cli_never_overwrites_or_writes_inside_frozen_trace(tmp_path, monkeypatch, existing, inside):
    frozen = tmp_path / 'trace';frozen.mkdir()
    output = frozen / 'new.json' if inside else tmp_path / 'new.json'
    if existing:output.write_text('retained')
    monkeypatch.setattr(sys, 'argv', ['decompose', '--trace-root', str(frozen), '--output', str(output), '--markdown', str(tmp_path / 'new.md')])
    with pytest.raises(ValueError, match='new and outside frozen trace'):trace.main()
    if existing:assert output.read_text() == 'retained'
