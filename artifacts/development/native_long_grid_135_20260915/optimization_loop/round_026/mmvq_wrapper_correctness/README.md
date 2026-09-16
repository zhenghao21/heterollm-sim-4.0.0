# R26 synthetic MMVQ correctness driver — fixed GPU fixture passed, performance unqualified

Scope is exactly `Q5_0`, M=1, K=4096, N=3072, padded K=4096, one explicit
nonblocking stream, no fusion and no timing loop. This is a standalone wrapper
correctness preparation. It does not certify the locked backend's runtime dispatch
or admit a cost-model coefficient.

## Artifacts and isolation

- `correctness_driver.cpp` includes the original R6 `mmvq_probe_abi.h` directly.
  There is no independently copied prototype. Exact pointer-type assertions retain
  the convert `type/k/m/padded_k` four-integer signature.
- `build_driver.py` links the three unchanged R6 objects by absolute path. It does
  not recompile them, copy a source tree, touch R6 or invoke an executable.
- `mmvq_gpu_correctness.exe` executed once in `../wrapper_correctness_v2_run.0001`;
  the fixed GPU fixture passed. The predecessor runner was rejected before GPU startup.
- `mmvq_cpu_self_test.exe` is separately compiled with the GPU path excluded and
  links only the CPU base library and Windows libraries. Its import table contains
  no cudart, nvcuda or ggml-cuda. It also rejects any such module being loaded.
- `cpu_self_test_result.json` and five pytest checks cover the CPU reference,
  independently reconstructed Python Q8_1 bytes, bad-output detection, exclusive
  output creation and rejection of the GPU flag by the CPU-only binary.
- `build_inputs.json`, generated `build_identity.h`, `build_receipt.json`, compiler
  discovered include lists and logs retain exact source, object, toolchain, import
  library and expected runtime DLL hashes. Before any CUDA API call the driver
  checks the manifest SHA, its input files and the actual loaded GGML/CUDA modules.
- `verify_device_code.py` extracts cubins without launching the executable. Its
  receipt can establish retained device-code identity only.

The native-thread-control tree is an overlay, not another complete GGML tree.
Its recorded unchanged `ggml-base.dll` and the semantic build's `ggml-base.dll`
are byte-identical. The CPU quantizer is resolved from that locked DLL and its
actual loaded path/SHA is recorded; no missing native source path is invented.
The native DLL is never rebuilt or replaced by this work.

## Deterministic data and conversion reference

Every original F32 weight is nonzero. For index i, `mix(i XOR 0x51354d31)` is the
fixed 32-bit integer permutation in `cpu_reference.h`; its low bit selects the sign
and the remaining bits select a dyadic magnitude between 1/512 and 1023/512.
Weights pass through the locked `ggml_quantize_chunk(GGML_TYPE_Q5_0,...)` and all
packed rows are independently decoded and compared with the locked CPU dequantizer.

Every 32-element input block has `d = 2^(-10 + block_index mod 4)`. Fifteen pairs
contain `(+q*d,-q*d)` with q in 1..120. The last two values are `127*d` and
`-(64 + block_index mod 63)*d`. All 4096 input elements are nonzero; block sums are
nonzero and vary. These inputs have no rounding ties, denormals or overflow.

This construction is deliberate: the locked CUDA source writes Q8_1 `s` as
`half(sum(original_input))`, while Q5_0's dot implementation uses `s` in its offset
correction. For arbitrary F32 data this need not equal `d * sum(quantized_values)`.
Here both quantities are exactly equal and exactly representable in half. Thus a
CPU dot of dequantized Q5_0 and returned Q8_1 is the correct reference without an
unexplained quantization allowance. This fixture does **not** validate arbitrary
rounding-boundary inputs, zero blocks, padding, other shapes or other datatypes.

The CPU Q8_1 reference is constructed independently as little-endian half `d`,
half `s`, then 32 signed bytes. The driver compares **all 4608 converted bytes
exactly**, including scale, sum and quantized values, before it calls MMVQ. Python
independently reproduces the layout/hash during CPU-only tests.

## Predeclared floating-point acceptance

The output reference is the binary64 sum of products of locked-CPU-dequantized
Q5_0 weights and returned, dequantized Q8_1 inputs. Comparison to original F32
weights is not used, because that would mix quantization error with kernel error.
No GPU output was available when this rule was chosen.

For binary precision with unit roundoff u, define `gamma(n,u)=n*u/(1-n*u)`.
In locked `vecdotq.cuh`, Q5_0 has QI=4 and VDR=2: two integer partial dots per
32-element block, hence P=256 partials per output row. Each partial has at most
four floating operations (scale, offset scale, subtraction, final scale). Any
serial or tree reduction of P partials has at most P-1 additions on one path.
Therefore `n32=P+3=259` is a conservative forward-error path bound independent of
which warp/tree order the backend selects. Integer DP4A intermediates are bounded
by `16*31*127=62992`, well below integer overflow and exactly convertible to float.
All fixture arithmetic remains in the normal floating range.

For each row calculate, from the already quantized inputs only:

```text
A = sum over blocks |d5| * (|d8| * sum_j |u5_j*q8_j| + 16*|s8|)
S = sum over elements |dequant_w_j * dequant_x_j|
g32 = gamma(259, 2^-24)
g64 = gamma(4*K+64, 2^-53)
bound = nextafter_up((g32*A + g64*S) / (1-g64))
```

Here `u5` is the unsigned stored 5-bit value (the signed value plus 16). Using
unsigned magnitudes and the explicit offset term prevents cancellation from
shrinking the forward-error allowance. The binary64 term also covers construction
of the reference/amplitude; the final bound rounds upward. FMA contraction may
reduce rounding error and does not invalidate this conservative bound. A row
passes only if its finite output differs from the reference by no more than its
precomputed bound. NaN/Inf always fail. Tolerances are not measured maxima or fitted
relative percentages. The actual error, bound ratio and worst row are retained.

Relevant locked sources, hashed in the build inputs:

```text
source/llama.cpp-semantic/ggml/src/ggml-cuda/quantize.cu:54-105, 558-573
source/llama.cpp-semantic/ggml/src/ggml-cuda/vecdotq.cuh:172-200, 811-828
source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu:693-818
source/llama.cpp-semantic/ggml/src/ggml-common.h:115-116, 229-269
source/llama.cpp-semantic/ggml/src/ggml-quants.c:187-229, 500-524
```

## Runtime behavior and observed numerical result

The GPU executable requires explicit `--run-correctness --output ABSOLUTE_NEW_JSON`.
The JSON is opened with Windows `CREATE_NEW`; existing files are never replaced.
An initial non-passing journal is flushed before CUDA calls, and failures remain
recorded. All raw artifacts also use exclusive creation.

The completed run records actual device UUID/name/PCI address, compute capability,
SM count, memory/L2 properties, driver/runtime versions and stream handle. It uses
one H2D input upload, the shared conversion shim, D2H conversion validation, the
shared main shim and D2H output validation. Buffers have checked 256-byte leading
and trailing guard regions. Every CUDA call made by this driver, immediate launch
status, synchronization and explicit cleanup is checked. Errors in inherited R6
support remain part of that support's separate qualification scope.

Packed weights, original inputs, expected/returned Q8_1, output, CPU reference and
predeclared per-row bounds are retained as raw artifacts with SHA. The fixed
native test dataset is not read, target LLMs are not run and no timing coefficient
is changed. Passing this synthetic check would still leave target dispatch,
launch geometry, stream correlation, cache behavior and profiler overhead
unverified; those are the separate capture/qualification task.

## Reproduce only the authorized static and CPU work

From this directory with the project's Python environment:

```text
python build_driver.py
python -m pytest test_correctness_driver.py -q
python verify_device_code.py
```

The final command creates its extraction directory/receipt exclusively. Use a new
reviewed directory if rebuilding a different identity; do not overwrite historical
GPU results. OBJ, EXE and extracted cubins are local build artifacts and should not
be treated as an instruction to upload binaries.
