# Host service controls — prepared only

No counter pilot, timing, GPU execution, native LLM or simulator change has been
performed by this successor's preparation. Only compile/link and pure-host logic
checks are permitted during the current R27 campaign. The pure-logic executable's
MSVC startup security cookie imports QPC (verified against the installed CRT source
and linker map); the test object itself references no experiment timing API. This
is not a clock-call instrumented claim about the entire CRT. The old import-gate
failure is preserved as build attempt 0001, and no probe was executed.

## Owner correction

The existing 250 ns per physical invocation group describes a host/API submission
queue semantic, but the current graph places its demand on `gpu0.command_queue`
(`reference.py:584`, `planner.py:7144-7152`). It is **not currently charged to a CPU
core**. The 1000 ns/kernel term is separately on `gpu.frontend`. The `128+12G`
command-build instruction assumption is another owner. A measured caller-submit
candidate is not automatically added to or substituted for any of these. Source
semantics, demand resource, and overlap must be reviewed separately.

## Bounded successor and original source reuse

`service_probe.cpp` directly includes the pinned predecessor `probe.cpp` with its
old main renamed and never called. Thus Graph construction, exact target DLL
identity checking, the CUPTI launch decoder and CPU numerical check are reused;
no source tree is copied, target DLL patched, CUDA kernel rebuilt or old artifact
changed. New functions/records are isolated here. `strict_gate.py` reuses the predecessor's strict environment/process functions, whose source
SHA remains pinned, and adds successor/wrapper/freeze runner names so relative
command lines also count as competing project work. Inherited dispatch/profiler overrides are rejected, secrets are
not inherited, live project native/simulation/probe work blocks execution, and no
absence of desktop GPU activity is inferred.

The operator is the same unfused F32 RMS_NORM 4096x1, 1/4/16 nodes. The finite group
ladder is 32/128/512/2048. It includes all original controls (N,128) and the equal-K
triplet (1,512), (4,128), (16,32), so [constant,K,G] is full-rank. Chain is development;
fanout is kept whole as holdout. Five independent processes, 31 samples/case, no
outlier removal/retries. Each case reuses an already allocated/warmed graph.
Construction and allocation are outside this caller-submit scope.

Each sample records empty-loop-before, submit, busy-sync, idle-sync and
empty-loop-after separately. Empty loops retain a volatile side effect. They are
observer/loop controls, not a fabricated driver API baseline. CPU demand during
synchronization is kept distinct from its wait wall; active CPU does not by itself
prove spin versus other API work. Actual CUDA scheduling flags are recorded.
There are no GPU events or wall-minus-GPU calculations in this successor.

`counter_pilot.cpp` is a separate executable without CUDA/GGML imports. Three
processes have at most 20 seconds of sampling budget each (60 seconds total),
65 controls/process: empty loop counts 0/128/512/2048, busy windows
0.25/1/4/16/64 ms, and Sleep 1/4/16/64 ms, five repetitions. A deadline overrun is a
failure, not a retry or fabricated <=60s success. OS scheduling can delay any
operation; the program records failure if its sampling deadline was exceeded.

Raw start/end user/kernel 100ns units and cycles are always retained. Each clock
snapshot is bracketed by QPC-before/QPC-after; the payload interior starts after
the first snapshot and ends before the second snapshot. Read envelopes remain raw.
Neither 100 ticks nor a smallest positive observed increment is an error bound.
QueryThreadCycleTime is never converted to ns. Raw counters alone never assert
CPU precision or automatically produce an admitted service coefficient.

Conditional development use remains allowed. The pilot reports accounting
visibility and reported user+kernel accounting/wall ratios per controlled window,
plus the difference between long busy-window ratios. Under the explicit assumption
that descheduling/interrupt effects are limited, stable long windows can support a
**conditional** caller-accounting cost candidate and help choose aggregation windows.
Sleep controls expose wait versus accounting behavior. These are observed-window
convergence/visibility findings, not a guaranteed clock-resolution upper bound or
formal <=5% service accuracy. The group ladder can similarly supply conditional
per-K/per-G development evidence once measured, with caller-only scope, retained
sync/wait data and independent holdout. Neither ETW nor this protocol forbids such
transparent development use; integrating the candidate into resource demands still
requires owner/overlap review and a new simulator freeze.

## Executable ETW ingestion boundary

The runnable `analyze.py attach-etw` interface consumes a completed service record
and a `host-service-etw-normalized/v1` document. It does **not** pretend to capture
or export ETL. An actual collector/exporter remains necessary and must establish:

- `probe_record_ref`, `source_etl_ref`, `exporter_ref`, `export_receipt_ref`, each with
  exact path/bytes/SHA. The exporter receipt binds its source ETL, exporter and the
  canonical normalized payload hash (document excluding `export_receipt_ref`).
- `clock_client_context:1`, matching `qpc_frequency`; integer `lost_events:0` and
  `lost_buffers:0`; complete scheduler and interrupt coverage; enclosing
  `coverage_start_qpc` and `coverage_end_qpc`.
- `scheduled_intervals`: complete `{pid,tid,cpu,start_qpc,end_qpc}` on-CPU intervals.
- `interrupt_intervals`: `{cpu,start_qpc,end_qpc,kind}` with kind DPC or ISR.

The adapter rejects overlaps on the same CPU or TID, reversed time, mismatched
clock/probe identity, loss or missing coverage. It intersects on-CPU intervals with
each phase and subtracts the **union** of DPC/ISR spans once. Same-process other
thread CPU is separately reported, never summed into the calling thread or
identified as a driver worker merely by name. The tick allowance is explicitly
only a QPC quantization allowance, not a guaranteed clock accuracy confidence bound.

Before candidate generation, a separate `host-service-etw-method-qualification/v1`
artifact is required. It binds the protocol, collector configuration, exporter
source and validation receipt. Validation must contain runtime busy/Sleep controls,
context-switch migration, DPC/ISR overlap, loss rejection, QPC clock identity and
paired observation-overhead checks with their raw evidence references. This
artifact is currently **not available**; no field is filled in as passed. The
ETW attachment interface can analyze bound evidence when provided. Calibration
admission is deliberately disabled in this frozen protocol: a caller-provided
`passed_checks` JSON, matching SHA values or QPC tick count cannot enable it. A
separately reviewed collector/exporter and measured method validator must be
implemented and frozen before a new version can allow candidate admission.
The `candidate` command therefore rejects in this preparation even if passed=true
is supplied. This is an explicit remaining implementation requirement.

Future command shape:

```text
python analyze.py pilot-summary --records <three pilot records> --output <new summary>
python analyze.py attach-etw --record <service record> --etw <normalized ETW.json> --output <new attached.json>
python analyze.py candidate --development <five attached chain files> --holdout <five attached fanout files> --direct <ten unprofiled service records> --method-qualification <reviewed method.json> --output <new candidate.json>
```

Analysis requires immutable runner start/child/finish receipts, the exact raw
record and command, pre/post build identities and process gates. It rejects a
standalone record, an incomplete terminal receipt and a record from outside the
fixed run root. The `profiling:false` field proves only the in-process CUPTI
policy; it does not prove the absence of an external ETW session. `direct` is a
requested observation cohort until the independent method establishes this state.

No target LLM timings enter this analysis. Intended candidate gates include full coverage,
5% within/between-process service stability, <=2% tick allowance and empty-control
fraction/drift, <=5% traced/untraced wall effect (diagnostic only, not a wall-to-CPU
conversion), <=5% tail normalized-cost convergence, and <=5% error on the complete
held-out topology. An unconverged ladder is reported as potential queue/backpressure
or domain change, not averaged into a constant. After the method admission implementation is completed, a nonnegative full-rank
`fixed + beta_K*K + beta_G*G` surface would be restricted to this measured
operator/hardware/runtime/joint domain. It does not generalize to all kernels, G=1,
other graph sizes or other hardware. Source/refitting to LLM latency remains forbidden.

## Preparation and future execution

```text
python build.py verify
python build.py build
python build.py host-test --manifest <explicit build_manifest.NNNN.json>
python -m pytest -q test_controls.py
python run_controls.py --check --manifest <explicit build_manifest.NNNN.json>
```

Only `host-service-logic-tests` is executed by host-test; its imports are checked
for absence of experiment timing references in the test object and CUDA imports;
the verified CRT startup QPC dependency is documented above. Counter/GPU programs remain unexecuted.
A future root-reviewed run uses `run_controls.py --execute`, an explicit mode
(pilot/path/service), topology and process index. Root is fixed to this directory's
`runs` to prevent a changed output path silently bypassing the pilot/retry budget.
Both path topologies must qualify before service measurements. Direct and ETW
service records use separate immutable subdirectories. ETW presence is only a
requested observation mode until the actual trace evidence establishes it.
Every execution calls the pinned R27 full262 verifier before and after the child,
and requires both frozen arms/262 terminal identities, not merely an idle process
list or completion-file existence. Process idle is checked again after the full
barrier validation. No scoring/prediction is executed by this check. Rejected
preflight/budget attempts receive an immutable failed terminal, as do child failures
and observation interruptions. Every run retains start, child PID, raw record/stdout/stderr, pre/post identities,
process gates and failed terminal. Waiting does not kill a child on observation
interruption. All official source links and parameter/clock constraints are kept
in the single `protocol.json`; no additional plan documents are created.

Revision 0004: the predecessor analyzer is pinned in protocol/build identity and verified before import, after import and after validation; exact verified source bytes are executed rather than a cached pyc. The idle gate includes the GPU timing collector, relative execution runners and Windows spawn parent ownership, rejecting unknown ancestry while allowing explicit host-only tests. Final reference failures produce rejected terminal receipts; any recorded execution/post-check/reference error forbids validated status. Revision 0003 sources and manifest are retained under prepared_revision.0003. No new timing/GPU execution is part of this revision.
