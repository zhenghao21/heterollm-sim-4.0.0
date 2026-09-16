# R26 ExC-aware original-DLL MMVQ recorder

Status: **compiled; 63 host-only tests passed (all original 38 retained); this
new GPU recorder has never been executed**. Current build is `build_manifest.json`;
current test receipt is `host_test.0001.json`. This directory is a new revision.
The old recorder, its manifest0004, and `target_capture_run.0001` remain unchanged.

The predecessor's actual run observed runtime callback ID430 (`cudaLaunchKernelExC`)
and the two expected symbols, but did not decode config or kernel arguments. Its
legacy grid=(1,1,1) and function/stream=0 fields were defaults, not observations.
`source_contract.json` pins that rejected record, the prior build/source files,
the same target DLLs, locked source and installed CUDA/CUPTI headers.

## New decoding boundary

The recorder supports exact installed-header APIs:

- `cudaLaunchKernel_v7000_params`: unchanged original support.
- `cudaLaunchKernelExC_v11060_params` (ID430): copy `cudaLaunchConfig_t` during
  ENTER, then copy grid/block, dynamic shared bytes, stream, kernel arguments and
  a bounded array of attributes before returning. No callback performs a CUDA
  call or dereferences a device pointer.

The full source-qualified 9-argument conversion and 19-argument MMVQ decoders,
correlation/context checks, pointer relationships, memory API refusal and return
code checks remain active. Unsupported APIs (including the unqualified ptsz
variant), contradictory API ID/name, unknown kernel symbols and missing data are
recorded and rejected.

`launch_metadata_json.h` is shared with host tests. When config/geometry has not
been copied, `geometry_observed` is false and function/stream/grid/block/shared
are JSON null. A captured stream of zero is represented as zero only when it was
actually observed. Legacy APIs have no attribute list in their parameter ABI;
the recorder does not synthesize one.

## Attribute contract

Locked `common.cuh::ggml_cuda_pdl_config` emits exactly one attribute:

```
id = cudaLaunchAttributeProgrammaticStreamSerialization (6)
val.programmaticStreamSerializationAllowed = 1
numAttrs = 1
```

Only that observed combination qualifies the current ExC source path. The SDK
allows other attributes and general nonzero PDL values; those permissions do not
prove that this locked source produced them. Missing/zero/duplicate attributes,
other SDK attributes, unknown IDs and any PDL value other than the exact source
value are retained and rejected. No default attribute is fabricated.

At most 16 attribute entries are copied. Larger lists preserve their actual
reported count and bounded prefix, set truncation, and fail qualification.
Unknown values are preserved as opaque union bytes without interpreting pointers.
For the known PDL attribute, only its active four-byte int is meaningful. The
source leaves inactive union bytes unspecified; the recorder retains them as
opaque data and never requires zero padding or uses it for qualification.

Each kernel is checked independently, so a legacy conversion and ExC main can
qualify if both paths match their contracts. PDL can affect GPU dependency
execution; CPU launch order is not a claim that GPU durations cannot overlap.
This recorder provides no timing value or performance coefficient.

## Reproduction without GPU

From this directory:

```
E:\anaconda\python.exe build.py verify
E:\anaconda\python.exe build.py host-test
```

`build.py` has no GPU-run action. The independent test executable imports zero
CUDA/CUPTI/GGML DLLs, verified from its import table. The 25 new checks cover ExC
identity, config/attribute/argument lifetime, null inputs, PDL source values,
unknown attribute serialization, bounded truncation, mixed APIs, geometry errors,
failed or absent EXIT, unknown symbols, and unobserved-geometry JSON. Both the
constructor proof and 64-bit SDK struct sizes/offsets are pinned and asserted.

Actual target callback compatibility remains unverified. A new reviewed runner
must select this new manifest and test receipt before any later GPU invocation;
neither the old failed run nor these host tests authorize performance calibration.
