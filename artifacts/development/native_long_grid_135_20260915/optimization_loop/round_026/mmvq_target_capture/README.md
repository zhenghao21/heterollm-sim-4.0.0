# R26 original-DLL MMVQ launch recorder — host preparation

Status: compiled, 38 host-only checks passed, **GPU recorder never executed**.
Current build: `build_manifest.0004.json`; current test: `host_test.0003.json`.
The older unnumbered manifest describes development revision 0003, whose source
snapshot is retained in `development_revision.0003.zip`. Do not use it for current execution.

## Scope and boundary

One independently constructed Q5_0 / M1 / K4096 / N3072 MUL_MAT invokes the
unmodified locked ggml-cuda.dll, not the R4/R6 source wrapper. This is a dispatch
identity experiment only. It produces no duration, throughput or calibration
coefficient, performs no LLM request and reads no target LLM actual latency.

The target process checks loaded ggml-base and ggml-cuda module paths and full
SHA256 before and after graph execution. CUPTI is loaded by absolute path,
full-file SHA and API version 130401. CUDA UUID, capability 12.0, 84 SMs and warp
size 32 must match the pinned device. `source_contract.json` pins source ABI,
original module receipts, device-code evidence, type, shape and expected geometry.

The CUPTI callback uses the runtime domain and bounded storage only. It copies
parameter values before return, performs no CUDA calls and never dereferences a
device pointer. It accepts only the full source-qualified Q8_1 conversion symbol
and Q5_0 M1/nonfusion/nonsmall-K/nonhalve-iters main symbol. The conversion decoder
copies 9 parameters; main copies 19 including the 48-byte fusion object. Unknown
symbols or unsupported launch APIs remain records but cannot qualify. Launch API
exit status must be present and successful.

An accepted pair must match all tensor pointers and strides: graph input feeds
conversion, conversion output feeds main, graph weights feed main and main output
is the graph destination; ids/fusion are absent. Conversion precedes main on the
same context and stream, with distinct correlation IDs. Runtime memcpy/memset
calls during the graph interval are separately recorded and reject qualification.
Allocation and synchronization are recorded separately and must return success.
This does not assert zero physical memory traffic: kernel reads/writes and all
unobserved driver internals remain outside that API claim.

## Exact source-derived geometry

For sm_120 the MMVQ parameter table is GENERIC. Q5_0 K4096 has 128 quant blocks,
4 warps and 32 blocks per K-loop iteration, so small_k is false. halve_iters is
GB10-only, so false. M1 has one output row per CTA: grid=(3072,1,1), block=(32,4,1),
dynamic shared=0. Q8_1 conversion has K padded to 4096, 256 threads per block:
grid=(16,1,1), block=(256,1,1), dynamic shared=0. The static shared memory compiled
into the kernel is NOT the dynamic shared argument and is not asserted to be zero.

Main weight strides use Q5_0 blocks (128 per row, 393216 per channel/sample), input
strides use Q8_1 blocks (128), output strides use F32 elements (3072). Fast division
of one is uint3(1,0,1); nchannels_y is the unused uint3(0,0,0) for no-ids.

## Safe reproduction now

```
E:\anaconda\python.exe build.py verify
E:\anaconda\python.exe build.py host-test --manifest build_manifest.0004.json
```

The host test is a separate executable that imports **zero CUDA/CUPTI/GGML DLLs**,
confirmed by dumpbin. It tests synthetic callback records and refusal cases only.
Its success is not evidence of live CUPTI/header compatibility. Build.py provides
no GPU-run action. The target recorder requires explicit runtime authorization and
an exclusive new output file; root must provide an independently reviewed runner
that verifies this manifest, isolates DLL resolution/environment, reserves the
GPU and preserves pre/post receipts. No such runtime action was taken here.

## Preserved development failures

Build 0001 compiled both executables but could not write a manifest due to Python
encoding spelling `utf8-sig`. Build 0002 compiled both but rejected localized MSVC
include-log parsing. Logs and failure receipts remain. Build 0003 passed 34 tests;
review added bounded memory/allocation/synchronization API records. Build 0004
passed 38 tests, preserves all failure states, and is the current candidate.

## Remaining limitations

- Live runtime dispatch identity and actual CUDA/CUPTI callback compatibility are unverified.
- This experiment alone does not establish numerical kernel correctness or timing accuracy.
- Only the explicitly pinned contiguous, two-dimensional Q5_0 M1 case is accepted.
- Pointer equality is checked inside one process; compare relationships, not raw addresses,
  across a later target-vs-wrapper experiment.
- Runtime memory API detection is not a claim that driver-domain internal operations are absent.
