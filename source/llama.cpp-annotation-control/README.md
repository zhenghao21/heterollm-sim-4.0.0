# Annotation control overlay

This private local overlay retains the original `../llama.cpp-semantic` source, objects, libraries and runtime unchanged. It recompiles exactly three translation units, rebuilds the server-context archive, and links three DLLs and the server EXE into `build-annotation-control/bin`. Other runtime files are copied byte for byte.

Set `LLAMA_TRACE_ANNOTATIONS=1` before process startup to enable the existing CUDA operator and phase NVTX annotations. The default, `0`, skips metadata/layout/format construction before any annotation call. The value is cached once per translation unit and cannot be toggled inside a running process. Only the exact string `1` enables it.

The locked server build has `LLAMA_SERVER_HOST_TRACE` disabled. Its diagnostic string construction is now also compiled out. A future build with that compile flag enabled follows the same runtime environment switch. Actual engine timestamp sampling, token arrays and counters are unchanged.

## Reproduce

Run `E:/anaconda/python.exe source/llama.cpp-annotation-control/build_overlay.py` from the simulator root, then `E:/anaconda/python.exe source/llama.cpp-annotation-control/test_annotation_guard.py`. The build uses the existing MSVC/CUDA toolchain, original compile flags, PCH, readonly headers and preserved objects. It is an incremental overlay build, not a clean build.

`evidence/source_manifest.json`, `evidence/before`, `evidence/annotation_control.patch`, `evidence/header_snapshot.json`, `evidence/build_receipt.json`, `evidence/build.log` and `evidence/guard_validation.json` retain provenance and checks. The old runtime SHA map is verified before and after. A successful no-export EXE link need not produce an import library.

## Qualification

Version startup and scope/switch regression checks pass. No model performance measurement was run by the builder. New runtime identity and the annotation environment choice must be captured before comparison. The previous binary's semantic proof must not be reused for this runtime. Performance equivalence, observer overhead and variance remain to be measured by the experiment owner.
