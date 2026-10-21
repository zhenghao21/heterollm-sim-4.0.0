"""PD transfers, continuous batching and measured traffic use real event callbacks."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heterollm_sim.paper_serving_control import (
    DecodeExecution, PendingAdmission, ServingConfig, ServingRequest,
    THREE_YEARS_SECONDS, run_serving)


def callbacks(import_ns=3., decode_ns=5.):
    imports = []
    batches = []

    def importer(request, now):
        imports.append((request, now))
        return PendingAdmission(request, now + import_ns, request.context_tokens * 2,
                                request.context_tokens * 2, request.context_tokens * 2,
                                {"pd_transfer": import_ns / 2, "pd_write": import_ns / 2})

    def decoder(batch, pending, now):
        batches.append((batch, pending, now))
        return DecodeExecution(now + decode_ns, len(batch) * 4, len(batch) * 2,
                               len(batch) * 3, {"kernel": decode_ns})

    return importer, decoder, imports, batches


def config(**kwargs):
    return replace(ServingConfig(2, 10, 5, warmup_steps=0, measurement_steps=8,
                                prefetch_slots=2, import_lookahead_tokens=2), **kwargs)


def test_prepared_full_batch_executes_every_event_and_pays_startup():
    importer, decoder, imports, batches = callbacks()
    result = run_serving(config(), decoder, importer)
    assert result.startup_end_ns == 3
    assert result.startup_import_bytes == (10 + 12) * 2
    assert result.startup_program_bytes == 44
    assert result.startup_costs_ns == {"pd_transfer": 3, "pd_write": 3}
    assert len(batches) == 8
    assert all(len(batch) == 2 for batch, _, _ in batches)
    assert result.decode_tokens == 16
    assert result.completed_requests == 4
    assert result.admitted_requests == 3
    assert result.full_batch_step_fraction == 1
    assert result.admission_stall_ns == 0
    assert result.mean_tpot_ns == result.mean_iteration_ns == 5
    assert result.max_tpot_ns == result.max_iteration_ns == 5
    assert result.decode_tokens_per_second == 16 / (40 / 1e9)
    assert result.output_equivalent_tokens_per_second == 20 / (40 / 1e9)
    assert result.pd_import_bytes == result.pd_program_bytes == 60
    assert result.decode_program_bytes == 64
    assert result.logical_decode_write_bytes == 32
    assert result.decode_transfer_bytes == 48
    assert [row.generated_decode_tokens for row in batches[0][0]] == [0, 2]
    assert [row.context_tokens for row in batches[0][0]] == [10, 12]
    assert imports[2][1] == 3  # An actual pre-import, not free completion-time replacement.
    assert result.step_records[0]["pending"] == 1
    assert "no cycle" in result.assumptions[0]


def test_reactive_import_stalls_are_counted_in_system_runtime():
    importer, decoder, _, batches = callbacks(import_ns=11)
    result = run_serving(config(prefetch_slots=0, import_lookahead_tokens=0), decoder, importer)
    assert result.full_batch_step_fraction == 1
    assert result.admission_stall_ns > 0
    assert result.mean_iteration_ns > 5
    assert result.max_tpot_ns > 5
    assert any(row["decode_start_ns"] > row["start_ns"] for row in result.step_records)
    assert all(len(batch) == 2 for batch, _, _ in batches)


def test_partial_policy_exposes_batch_gaps_instead_of_claiming_full_batch():
    importer, decoder, _, batches = callbacks(import_ns=20)
    result = run_serving(config(prefetch_slots=0, full_batch_policy="partial"), decoder, importer)
    assert result.full_batch_step_fraction < 1
    assert result.mean_batch_size < 2
    assert any(len(batch) < 2 for batch, _, _ in batches)


def test_callbacks_share_import_resources_with_decode():
    write_free = 0.

    def importer(request, now):
        nonlocal write_free
        write_free = max(now, write_free) + 7
        return PendingAdmission(request, write_free, 10, 12, 10, {"write": 7})

    def decoder(batch, pending, now):
        # A callback-selected half-duplex link competes with prefetch writes.
        return DecodeExecution(max(now, write_free) + 5, costs={"decode": 5})

    result = run_serving(config(measurement_steps=4), decoder, importer)
    assert result.startup_end_ns == 14
    assert result.step_records[0]["end_ns"] - result.step_records[0]["start_ns"] == 12
    assert result.max_tpot_ns >= 12


def test_window_does_not_recount_warmup_tokens_or_startup_writes():
    importer, decoder, _, batches = callbacks()
    result = run_serving(config(warmup_steps=4, measurement_steps=8), decoder, importer)
    assert len(batches) == 12
    assert result.warmup_steps_completed == 4
    assert result.measured_steps == 8
    assert result.decode_tokens == 16
    assert result.measurement_start_ns == 23
    assert result.measurement_end_ns == 63
    assert result.mean_tpot_ns == 5
    assert result.pd_import_bytes == 80
    assert result.startup_import_bytes == 44


def test_three_year_pec_uses_physical_writes_and_explicit_decimal_capacity():
    importer, decoder, _, _ = callbacks()
    result = run_serving(config(hbf_capacity_bytes=1000), decoder, importer)
    expected = (result.pd_program_completed_bytes + result.decode_program_bytes) / (40 / 1e9)
    assert result.projected_three_year_physical_bytes == expected * THREE_YEARS_SECONDS
    assert result.projected_three_year_average_pec == expected * THREE_YEARS_SECONDS / 1000
    assert "ideal uniform wear" in result.assumptions[-2]


def test_pec_excludes_final_prefetches_that_have_not_completed():
    importer, decoder, _, _ = callbacks(import_ns=20)
    result = run_serving(config(measurement_steps=1, import_lookahead_tokens=4,
                                hbf_capacity_bytes=1000), decoder, importer)
    assert result.pd_program_issued_bytes == 40
    assert result.pending_unfinished_program_start_bytes == 0
    assert result.pending_unfinished_program_end_bytes == 40
    assert result.pd_program_completed_bytes == 0
    expected = result.decode_program_bytes / (5 / 1e9) * THREE_YEARS_SECONDS
    assert result.projected_three_year_physical_bytes == expected


def test_pec_excludes_old_ready_import_when_admitted_inside_measurement():
    def importer(request, now):
        latency = 100 if request.request_id >= 3 else 3
        return PendingAdmission(request, now + latency, 20, 20, 20)

    _, decoder, _, _ = callbacks()
    result = run_serving(config(warmup_steps=2, measurement_steps=1), decoder, importer)
    assert result.pd_program_bytes == 20  # Admitted now, but PD already finished during warmup.
    assert result.pending_unfinished_program_start_bytes == 0
    assert result.pd_program_issued_bytes == result.pending_unfinished_program_end_bytes == 20
    assert result.pd_program_completed_bytes == 0


def test_pd_completion_window_is_corrected_for_inflight_warmup_imports():
    importer, decoder, _, _ = callbacks(import_ns=20)
    result = run_serving(config(warmup_steps=1, measurement_steps=3,
                                import_lookahead_tokens=4), decoder, importer)
    assert result.pending_unfinished_program_start_bytes == 40
    assert result.pd_program_completed_bytes == (
        result.pd_program_issued_bytes + result.pending_unfinished_program_start_bytes
        - result.pending_unfinished_program_end_bytes)
    assert result.pd_program_completed_bytes > result.pd_program_bytes


def test_hbm_has_no_nand_pec_when_capacity_is_zero():
    importer, decoder, _, _ = callbacks()
    assert run_serving(config(), decoder, importer).projected_three_year_average_pec is None


def test_prefill_first_token_convention_is_explicit():
    importer, decoder, _, _ = callbacks()
    result = run_serving(config(batch_size=1, prefetch_slots=1, stagger_initial_ages=False,
                                output_tokens=3, measurement_steps=4,
                                prefill_first_output_token=False), decoder, importer)
    assert result.completed_requests == 1
    assert result.config["prefill_first_output_token"] is False


def test_exogenous_idle_time_is_not_free_and_finite_arrivals_exhaust():
    importer, decoder, imports, batches = callbacks()
    result = run_serving(config(arrival_policy="exogenous", arrival_times_ns=(10., 30.),
                                batch_size=1, prefetch_slots=0, measurement_steps=20,
                                output_tokens=3), decoder, importer)
    assert result.status == "arrivals_exhausted"
    assert [at for _, at in imports] == [10, 30]
    assert result.decode_tokens == 4
    assert result.completed_requests == 2
    assert result.measurement_start_ns == 0
    assert result.measurement_end_ns == 43
    assert result.admission_stall_ns == 23
    assert len(batches) == 4
    assert result.decode_tokens_per_second == 4 / (43 / 1e9)


def test_exogenous_backlog_cannot_create_unbounded_resident_kv():
    importer, decoder, imports, batches = callbacks()
    result = run_serving(config(batch_size=1, prefetch_slots=1, output_tokens=3,
                                measurement_steps=6, arrival_policy="exogenous",
                                arrival_times_ns=(0.,) * 100), decoder, importer)
    assert result.measured_steps == 6
    assert len(imports) < 100
    assert all(len(batch) + len(pending) <= 2 for batch, pending, _ in batches)


def test_empty_arrival_source_has_no_invented_decode_or_throughput():
    importer, decoder, _, batches = callbacks()
    result = run_serving(config(arrival_policy="exogenous"), decoder, importer)
    assert result.status == "arrivals_exhausted"
    assert not batches
    assert result.decode_tokens_per_second == result.mean_tpot_ns == 0
    assert result.measured_steps == 0


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_free_or_invalid_import_completion_is_rejected(value):
    def importer(request, now):
        return PendingAdmission(request, value, 10, 10, 10)

    with pytest.raises(ValueError):
        run_serving(config(), lambda *args: DecodeExecution(10), importer)


def test_import_cannot_replace_request_or_skip_kv_transfer_accounting():
    def wrong_request(request, now):
        return PendingAdmission(replace(request, request_id=99), now + 1, 10, 10, 10)

    with pytest.raises(ValueError, match="same request"):
        run_serving(config(), lambda *args: DecodeExecution(10), wrong_request)
    with pytest.raises(ValueError, match="non-zero KV"):
        PendingAdmission(ServingRequest(0, 10, 3), 1, 0, 0, 0)


@pytest.mark.parametrize("end", [3., 2., float("nan"), float("inf")])
def test_decode_must_advance_callback_completion(end):
    importer, _, _, _ = callbacks()
    with pytest.raises(ValueError):
        run_serving(config(), lambda *args: DecodeExecution(end), importer)


@pytest.mark.parametrize("changes", [
    {"batch_size": 0}, {"measurement_steps": 0}, {"warmup_steps": -1},
    {"prefetch_slots": 3}, {"prefill_first_output_token": "true"},
    {"output_tokens": 1}, {"arrival_policy": "invented"},
    {"full_batch_policy": "pretend"}, {"arrival_times_ns": (1.,)},
    {"arrival_policy": "exogenous", "arrival_times_ns": (2., 1.)},
])
def test_configuration_rejects_invalid_semantics(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_optional_records_do_not_change_execution_or_statistics():
    a_import, a_decode, _, _ = callbacks()
    b_import, b_decode, _, _ = callbacks()
    first = run_serving(config(), a_decode, a_import).as_dict()
    second = run_serving(config(retain_step_records=False), b_decode, b_import).as_dict()
    assert not second.pop("step_records")
    first.pop("step_records")
    first["config"]["retain_step_records"] = False
    assert first == second


def test_initial_and_successor_requests_keep_stable_ep_slots():
    importer, decoder, imports, batches = callbacks()
    run_serving(config(batch_size=3, prefetch_slots=3, output_tokens=7,
                       measurement_steps=18), decoder, importer)
    assert [request.slot_id for request, _ in imports[:3]] == [0, 1, 2]
    imported_slots = {request.request_id: request.slot_id for request, _ in imports}
    chains = {slot: [] for slot in range(3)}
    for batch, pending, _ in batches:
        assert len({row.slot_id for row in batch}) == len(batch)
        assert len({row.request.slot_id for row in pending}) == len(pending)
        for request in batch:
            assert request.slot_id == imported_slots[request.request_id]
            chain = chains[request.slot_id]
            if not chain or chain[-1] != request.request_id:
                chain.append(request.request_id)
    assert all(len(chain) >= 3 for chain in chains.values())
    # Active list positions change as earlier requests finish; ownership does not.
    assert any([row.slot_id for row in batch] != list(range(3)) for batch, _, _ in batches)
    assert any(any(row.request.slot_id in {active.slot_id for active in batch}
                       for row in pending) for batch, pending, _ in batches)


def test_ready_prefetch_cannot_steal_other_ep_slot_when_its_import_is_slow():
    batches = []
    imports = []

    def importer(request, now):
        latency = 100 if request.request_id >= 2 and request.slot_id == 0 else 3
        imports.append((request, now, now + latency))
        return PendingAdmission(request, now + latency, 20, 20, 20)

    def decoder(batch, pending, now):
        batches.append((batch, pending, now))
        assert len({row.slot_id for row in batch}) == len(batch)
        return DecodeExecution(now + 5)

    result = run_serving(config(measurement_steps=12, full_batch_policy="partial"),
                         decoder, importer)
    assert result.full_batch_step_fraction < 1
    assert any(len(batch) == 1 and batch[0].slot_id == 1 for batch, _, _ in batches)
    first_slot_zero_successor = next(row for row in imports
                                    if row[0].request_id >= 2 and row[0].slot_id == 0)
    for batch, _, now in batches:
        if now < first_slot_zero_successor[2]:
            assert not any(row.request_id == first_slot_zero_successor[0].request_id for row in batch)


def test_exogenous_backlog_replacements_inherit_known_slot_before_import():
    importer, decoder, imports, batches = callbacks()
    result = run_serving(config(batch_size=3, prefetch_slots=2, output_tokens=4,
                                measurement_steps=12, arrival_policy="exogenous",
                                arrival_times_ns=(0.,) * 20), decoder, importer)
    assert result.measured_steps == 12
    known = {row.request_id: row.slot_id for row, _ in imports}
    assert set(known.values()) == {0, 1, 2}
    for batch, pending, _ in batches:
        assert len({row.slot_id for row in batch}) == len(batch)
        assert len({row.request.slot_id for row in pending}) == len(pending)
        assert all(row.slot_id == known[row.request_id] for row in batch)
        assert all(row.request.slot_id == known[row.request.request_id] for row in pending)


@pytest.mark.parametrize("slot", [-1, True, 1.5])
def test_request_slot_must_be_nonnegative_integer(slot):
    with pytest.raises(ValueError, match="slot_id"):
        ServingRequest(0, 10, 4, slot_id=slot)


def test_complete_request_residence_excludes_staggered_startup_seeds():
    importer, decoder, _, _ = callbacks()
    result = run_serving(config(), decoder, importer)
    assert result.completed_requests == 4
    assert result.completed_request_elapsed_count == 2
    assert result.mean_completed_request_elapsed_ns == 20
    assert result.max_completed_request_elapsed_ns == 20


def test_no_complete_fresh_residence_is_none_instead_of_invented_zero():
    importer, decoder, _, _ = callbacks()
    result = run_serving(config(measurement_steps=2), decoder, importer)
    assert result.completed_requests == 1  # An aged startup request completes.
    assert result.completed_request_elapsed_count == 0
    assert result.mean_completed_request_elapsed_ns is None
    assert result.max_completed_request_elapsed_ns is None


def test_request_admitted_in_warmup_keeps_full_residence_until_measurement_completion():
    importer, decoder, _, _ = callbacks()
    result = run_serving(config(warmup_steps=4, measurement_steps=8), decoder, importer)
    assert result.completed_request_elapsed_count == 4
    assert result.mean_completed_request_elapsed_ns == 20
    assert result.max_completed_request_elapsed_ns == 20


def test_complete_request_residence_includes_other_slots_admission_stalls():
    importer, decoder, _, _ = callbacks(import_ns=11)
    result = run_serving(config(prefetch_slots=0, import_lookahead_tokens=0), decoder, importer)
    # Each fresh request executes four 5 ns decode steps and experiences one
    # 11 ns interruption while the other slot's successor is imported.
    assert result.completed_request_elapsed_count == 2
    assert result.mean_completed_request_elapsed_ns == 31
    assert result.max_completed_request_elapsed_ns == 31


def test_exogenous_request_residence_begins_at_paid_pd_ready_admission():
    importer, decoder, _, _ = callbacks()
    result = run_serving(config(arrival_policy="exogenous", arrival_times_ns=(10., 30.),
                                batch_size=1, prefetch_slots=0, measurement_steps=20,
                                output_tokens=3), decoder, importer)
    assert result.completed_request_elapsed_count == 2
    assert result.mean_completed_request_elapsed_ns == 10
    assert result.max_completed_request_elapsed_ns == 10
